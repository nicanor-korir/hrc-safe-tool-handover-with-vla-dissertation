"""
Scripted handover policy for the Path A simulation.

This is the stand-in for OpenVLA. The methodology commits to a learned
vision-language-action model, but full OpenVLA-7B integration is out of
scope for the time available, so the handover behaviour here is a
hand-written state machine. The four-condition ablation and the
graduated safety response are unchanged by this substitution, because
the monitoring streams read policy state and trajectory data rather than
a model hidden layer (Gu et al. 2025 frame failure detection over
internal policy signals, which we mirror with scripted internal state).

The machine moves through seven states.

  APPROACHING   reach above the chosen tool
  GRASPING      descend, secure the tool, confirm the grasp
  LIFTING       raise the tool clear of the table
  TRANSPORTING  carry it toward the handover zone
  PRESENTING    hold it handle-first for the human to take
  RELEASING     release the tool once intent is confirmed
  DONE          terminal, trial complete

All three tools are presented handle-first per Ortenzi et al. (2021).

Grasp model
-----------
The grasp is abstracted as a fixed kinematic attachment between the
gripper link and the tool, formed when the gripper arrives over the
tool. The Franka in this workspace is mounted level with the table, so
a friction-based pinch on the thin primitive tools is unreliable at the
floor of the arm's reach. Since the study measures the monitoring and
fusion streams rather than grasp mechanics, a constraint-based grasp is
the honest simplification. It makes the nominal pick deterministic so
that any failure seen in the data comes from the injected scenario, not
from grasp noise. The thesis records this as a Path A simplification.

Deliberate failure injection is built in for the four scenario groups
named in the methodology. Each group is selected by the failure_mode
parameter so the experiment runner can sweep across them cleanly. A
mode does not directly force a crash. It perturbs the policy in a way
that a real VLA would plausibly mishandle, and the monitoring streams
are then judged on whether they catch it.

  camera_shift   the policy's perceived tool pose is offset, so the
                 attachment can form off-centre or fail to form
  lighting       perception noise rises and grasp confidence degrades,
                 so the grasp can slip after forming
  novel_geometry the tool is treated as an unfamiliar shape, so the
                 grasp is poorly seated and weaker
  approach_traj  the human reaches on an unexpected path, arriving early
                 or off-axis, stressing release timing

Citations implemented here
  Ortenzi et al. 2021   handle-first presentation and fluency framing
  Gu et al. 2025        scripted internal state exposed for monitoring
"""

import enum
import numpy as np
import pybullet as p

from policy.robot_controller import END_EFFECTOR_LINK_INDEX


class HandoverState(enum.Enum):
    """The seven states of the handover sequence plus a terminal state."""

    APPROACHING = "APPROACHING"
    GRASPING = "GRASPING"
    LIFTING = "LIFTING"
    TRANSPORTING = "TRANSPORTING"
    PRESENTING = "PRESENTING"
    RELEASING = "RELEASING"
    DONE = "DONE"


# The four failure scenario groups, plus the nominal control.
# These string keys are what the experiment runner sweeps over.
FAILURE_MODES = (
    "nominal",
    "camera_shift",
    "lighting",
    "novel_geometry",
    "approach_traj",
)

# Gripper orientation pointing straight down, the natural top grasp.
GRIPPER_DOWN_EULER = [np.pi, 0.0, 0.0]

# Heights used during the sequence, relative to the table surface.
APPROACH_CLEARANCE = 0.12   # how far above the tool to hover first
LIFT_HEIGHT = 0.20          # how high to raise the tool after grasping

# A perceived offset larger than this means the attachment misses the
# tool entirely and no grasp forms. Tuned to the few-centimetre offsets
# camera_shift injects.
GRASP_MISS_OFFSET = 0.045


class HandoverPolicy:
    """
    Drives one handover trial from APPROACHING through to DONE.

    The policy owns its own notion of where things are. Under a failure
    mode that notion can diverge from physical truth, which is the point.
    A perfect monitor would notice the divergence. The experiment measures
    whether each condition's monitors actually do.

    Usage
    -----
        policy = HandoverPolicy(controller, robot_id, tool_id,
                                tool_start_pos, handover_target,
                                failure_mode="camera_shift", rng=rng)
        while not policy.is_done():
            policy.step(dt)
    """

    def __init__(
        self,
        controller,
        robot_id,
        tool_id,
        tool_start_pos,
        handover_target,
        failure_mode="nominal",
        rng=None,
        gui=False,
    ):
        """
        Parameters
        ----------
        controller : RobotController
            Low-level Panda controller from robot_controller.py.
        robot_id : int
            PyBullet body ID of the Panda, needed to form the grasp
            constraint against the gripper link.
        tool_id : int
            PyBullet body ID of the tool to be handed over.
        tool_start_pos : list of 3 floats
            World position of the tool at trial start.
        handover_target : list of 3 floats
            World position where the tool should be presented.
        failure_mode : str
            One of FAILURE_MODES. Selects which perturbation, if any,
            is injected this trial.
        rng : numpy.random.Generator, optional
            Seeded generator so trials are reproducible. If omitted a
            fresh default generator is created.
        gui : bool
            Whether a GUI is attached. Only affects optional debug draws.
        """
        if failure_mode not in FAILURE_MODES:
            raise ValueError(
                f"failure_mode must be one of {FAILURE_MODES}, got {failure_mode!r}"
            )

        self.controller = controller
        self.robot_id = robot_id
        self.tool_id = tool_id
        self.tool_start_pos = np.array(tool_start_pos, dtype=float)
        self.handover_target = np.array(handover_target, dtype=float)
        self.failure_mode = failure_mode
        self.rng = rng if rng is not None else np.random.default_rng()
        self.gui = gui

        # State machine bookkeeping
        self.state = HandoverState.APPROACHING
        self.state_entry_time = 0.0
        self.elapsed = 0.0

        # Recorded events for the experiment log. Created before any
        # method that appends to it.
        self.events = []

        # The PyBullet constraint id for the active grasp, or None.
        self.grasp_constraint = None

        # Perception error at grasp time, frozen once the grasp is
        # attempted. Before grasping it is None. This is the failure
        # relevant offset, namely how far the policy's perceived tool pose
        # sat from the true pose when it committed to the grasp. After the
        # tool is carried away its live position is no longer comparable to
        # the perceived start pose, so freezing the value keeps the signal
        # meaningful through presentation.
        self.grasp_time_offset = None

        # Perception of the tool pose. Under nominal this equals the truth.
        # Under camera_shift it is deliberately offset.
        self.perceived_tool_pos = self._inject_perception(self.tool_start_pos)

        # Grasp quality this trial, in [0, 1]. Degraded by some modes.
        # The confidence monitor reads this as the policy's own signal of
        # how sure it is the grasp succeeded.
        self.grasp_quality = 1.0

        # Mismatch between commanded grasp and the tool, set by
        # novel_geometry. Zero means a well-matched grasp.
        self.geometry_mismatch = 0.0

        # Perception noise level, raised by lighting. Feeds the monitor.
        self.perception_noise = 0.0

        # Baseline observation noise present on every trial, including
        # nominal. Real perception is never perfectly clean, so even a
        # healthy handover shows small jitter in its reported signals.
        # This gives the conformal calibration a genuine nominal spread
        # to set its threshold against (Xu et al. 2025).
        self.sensor_noise_sd = 0.004
        self._obs_grasp_jitter = float(self.rng.normal(0.0, 0.02))
        self._obs_offset_jitter = abs(float(self.rng.normal(0.0, self.sensor_noise_sd)))

        # Whether the grasped tool is actually held. A failed or slipped
        # grasp sets this False even while the policy proceeds.
        self.holding_tool = False

        # The human's predicted readiness gate, set externally by fusion.
        # Until release is authorised the policy waits in PRESENTING.
        self.release_authorised = False

        # Apply per-mode setup that does not depend on a later state.
        self._configure_failure_mode()

    # ----- public interface ------------------------------------------------

    def step(self, dt):
        """
        Advance the policy by one timestep and dispatch on current state.

        The robot controller's primitives step the simulation themselves
        for blocking moves, so this method does not call p.stepSimulation
        directly. The caller steps for any idle waiting.
        """
        self.elapsed += dt
        handler = self._state_handlers[self.state]
        handler()

    def is_done(self):
        """True once the machine has reached DONE."""
        return self.state == HandoverState.DONE

    def authorise_release(self):
        """
        Called by the fusion controller when it judges the human ready.
        Until this is set the policy holds in PRESENTING.
        """
        self.release_authorised = True

    def get_internal_state(self):
        """
        Expose the policy's own view of the trial for the confidence
        monitor. Under Path A this is the substitute for a VLA hidden
        layer (Gu et al. 2025). It carries the signals a SAFE-style
        monitor would derive from internal activations.

        Returns
        -------
        dict
            state name, grasp quality, geometry mismatch, perception
            noise, perceived versus true tool offset, holding flag.
        """
        # Use the frozen grasp-time perception offset once it exists. Before
        # the grasp is attempted, fall back to the live offset between the
        # perceived and true tool pose, which is small while the tool still
        # rests at its start position.
        if self.grasp_time_offset is not None:
            perceived_offset = self.grasp_time_offset
        else:
            true_pos, _ = p.getBasePositionAndOrientation(self.tool_id)
            perceived_offset = float(
                np.linalg.norm(self.perceived_tool_pos - np.array(true_pos))
            )
        # Add the per-trial baseline observation jitter so nominal trials
        # are not all identical. Grasp quality is clamped to [0, 1].
        reported_grasp = float(
            np.clip(self.grasp_quality + self._obs_grasp_jitter, 0.0, 1.0)
        )
        reported_offset = float(perceived_offset + self._obs_offset_jitter)
        return {
            "state": self.state.value,
            "grasp_quality": reported_grasp,
            "geometry_mismatch": float(self.geometry_mismatch),
            "perception_noise": float(self.perception_noise),
            "perceived_offset": reported_offset,
            "holding_tool": bool(self.holding_tool),
            "failure_mode": self.failure_mode,
        }

    # ----- failure mode configuration --------------------------------------

    def _configure_failure_mode(self):
        """
        Set the per-mode parameters that bias the trial. Magnitudes are
        chosen so the perturbation is meaningful without being trivially
        catastrophic. Real VLA failures are subtle, so the monitors have
        to earn their detections.
        """
        mode = self.failure_mode

        if mode == "nominal":
            return

        if mode == "lighting":
            # Brighter or darker scenes raise perceptual uncertainty.
            # We sample a noise level and let it degrade grasp quality.
            self.perception_noise = float(self.rng.uniform(0.25, 0.6))
            self.grasp_quality = float(1.0 - self.rng.uniform(0.2, 0.5))

        elif mode == "novel_geometry":
            # An unfamiliar tool shape gives a poorly seated grasp and a
            # weaker hold. The policy still tries its default grasp.
            self.geometry_mismatch = float(self.rng.uniform(0.15, 0.45))
            self.grasp_quality = float(1.0 - self.geometry_mismatch)

        elif mode == "camera_shift":
            # The offset is already baked into perceived_tool_pos via
            # _inject_perception. Grasp quality falls as the offset grows,
            # since the gripper aims at the wrong spot.
            offset = float(
                np.linalg.norm(self.perceived_tool_pos - self.tool_start_pos)
            )
            self.grasp_quality = float(np.clip(1.0 - offset * 6.0, 0.1, 1.0))

        elif mode == "approach_traj":
            # Nothing about the grasp changes. The stress is on release
            # timing, handled when the human arm is driven by the runner.
            # We record the intended perturbation so the log is explicit.
            self.events.append(("approach_traj_armed", self.elapsed))

    def _inject_perception(self, true_pos):
        """
        Return the policy's perceived tool position. Identical to truth
        unless camera_shift is active, in which case a lateral offset is
        added to mimic a viewpoint change the policy has not accounted for.
        """
        perceived = np.array(true_pos, dtype=float)
        if self.failure_mode == "camera_shift":
            # Offset mostly in the table plane, a few centimetres.
            offset = self.rng.uniform(-0.05, 0.05, size=3)
            offset[2] *= 0.3  # less vertical error than lateral
            perceived = perceived + offset
            self.events.append(("camera_shift_offset", offset.tolist()))
        return perceived

    # ----- state handlers --------------------------------------------------

    def _enter(self, new_state):
        """Transition helper that timestamps the new state."""
        self.state = new_state
        self.state_entry_time = self.elapsed
        self.events.append((new_state.value, self.elapsed))

    @property
    def _state_handlers(self):
        return {
            HandoverState.APPROACHING: self._do_approaching,
            HandoverState.GRASPING: self._do_grasping,
            HandoverState.LIFTING: self._do_lifting,
            HandoverState.TRANSPORTING: self._do_transporting,
            HandoverState.PRESENTING: self._do_presenting,
            HandoverState.RELEASING: self._do_releasing,
            HandoverState.DONE: self._do_done,
        }

    def _do_approaching(self):
        """Move to a hover pose above the perceived tool position."""
        hover = self.perceived_tool_pos.copy()
        hover[2] = self.tool_start_pos[2] + APPROACH_CLEARANCE
        orn = p.getQuaternionFromEuler(GRIPPER_DOWN_EULER)
        self.controller.open_gripper(steps=20)
        self.controller.move_gripper_to_pose(hover.tolist(), orn, max_steps=300)
        self._enter(HandoverState.GRASPING)

    def _do_grasping(self):
        """
        Form the grasp attachment and confirm it. Whether the attachment
        forms, and whether it holds, depends on the perceived offset and
        the grasp quality set by the failure mode.
        """
        self.controller.close_gripper(steps=40)
        self.holding_tool = self._attempt_grasp()
        self.events.append(("grasp_confirmed", self.holding_tool))
        self._enter(HandoverState.LIFTING)

    def _do_lifting(self):
        """Raise straight up to clear the tool from the table."""
        pos, orn = self.controller.get_gripper_pose()
        lift = np.array(pos)
        lift[2] = self.tool_start_pos[2] + LIFT_HEIGHT
        self.controller.move_gripper_to_pose(lift.tolist(), orn.tolist(), max_steps=300)
        self._enter(HandoverState.TRANSPORTING)

    def _do_transporting(self):
        """Carry the tool to a point above the handover target."""
        carry = self.handover_target.copy()
        carry[2] = self.tool_start_pos[2] + LIFT_HEIGHT
        orn = p.getQuaternionFromEuler(GRIPPER_DOWN_EULER)
        self.controller.move_gripper_to_pose(carry.tolist(), orn, max_steps=400)
        self._enter(HandoverState.PRESENTING)

    def _do_presenting(self):
        """
        Hold the tool at the handover pose handle-first and wait for the
        release authorisation. The grasped end leads so the handle points
        back toward the human (Ortenzi et al. 2021).
        """
        present = self.handover_target.copy()
        # Tilt so the tool is offered at a natural angle rather than
        # straight down. A roll keeps the handle toward the human side.
        orn = p.getQuaternionFromEuler([np.pi * 0.75, 0.0, 0.0])
        self.controller.move_gripper_to_pose(present.tolist(), orn, max_steps=300)

        if self.release_authorised:
            self._enter(HandoverState.RELEASING)
        # Otherwise stay in PRESENTING. The fusion controller decides.

    def _do_releasing(self):
        """Release the tool to the human by dropping the attachment."""
        self._release_grasp()
        self.controller.open_gripper(steps=60)
        self.holding_tool = False
        self.events.append(("released", self.elapsed))
        self._enter(HandoverState.DONE)

    def _do_done(self):
        """Terminal state. Nothing to do."""
        return

    # ----- grasp mechanics -------------------------------------------------

    def _attempt_grasp(self):
        """
        Try to form the grasp attachment between gripper and tool.

        A large perceived offset means the gripper is over the wrong spot
        and the attachment never forms. Otherwise the attachment forms and
        we then test whether a degraded grasp slips, using grasp_quality.

        Returns True only if a held, non-slipped grasp results.
        """
        # The attachment forms only if the gripper is actually near the
        # tool. Under camera_shift a big offset misses it entirely. We
        # measure the offset against the true tool pose at grasp time and
        # freeze it as the monitor's perception signal.
        true_pos, _ = p.getBasePositionAndOrientation(self.tool_id)
        perceived_error = float(
            np.linalg.norm(self.perceived_tool_pos - np.array(true_pos))
        )
        self.grasp_time_offset = perceived_error
        if perceived_error > GRASP_MISS_OFFSET:
            self.events.append(("grasp_missed", perceived_error))
            return False

        gp, go = self.controller.get_gripper_pose()
        tool_pos, _ = p.getBasePositionAndOrientation(self.tool_id)

        # Form a fixed constraint so the tool tracks the gripper link.
        # The child frame offset keeps the tool just below the grasp point.
        self.grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=END_EFFECTOR_LINK_INDEX,
            childBodyUniqueId=self.tool_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.02],
            childFramePosition=[0, 0, 0],
        )

        # A degraded grasp can slip immediately. The chance scales with
        # how poor the grasp quality is.
        slip_chance = (1.0 - self.grasp_quality) * 0.8
        if self.rng.random() < slip_chance:
            self.events.append(("grasp_slipped", round(slip_chance, 3)))
            self._release_grasp()
            return False

        return True

    def _release_grasp(self):
        """Remove the grasp constraint if one is active."""
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None


if __name__ == "__main__":
    # Manual smoke test. Run each failure mode once in headless mode and
    # report what the machine did. The nominal run should grasp, lift,
    # transport, and release cleanly.
    import time
    from environment.workspace import (
        build_workspace,
        HANDOVER_TARGET,
        SCREWDRIVER_START,
    )
    from policy.robot_controller import RobotController

    def run_one(failure_mode, seed=0):
        ids = build_workspace(gui=False)
        controller = RobotController(ids["robot_id"])
        rng = np.random.default_rng(seed)

        # Let the world settle so the tool rests on the table.
        for _ in range(120):
            p.stepSimulation()

        policy = HandoverPolicy(
            controller,
            robot_id=ids["robot_id"],
            tool_id=ids["screwdriver_id"],
            tool_start_pos=SCREWDRIVER_START,
            handover_target=HANDOVER_TARGET,
            failure_mode=failure_mode,
            rng=rng,
        )

        # Authorise release as soon as the policy reaches PRESENTING. In
        # the real experiment the fusion controller gates this.
        steps = 0
        tool_lift_z = None
        holding_before_release = None
        while not policy.is_done() and steps < 40:
            # Capture the held state and lifted height while presenting,
            # before the release flips holding_tool back to False.
            if policy.state == HandoverState.PRESENTING:
                if holding_before_release is None:
                    holding_before_release = policy.holding_tool
                    tool_lift_z = p.getBasePositionAndOrientation(
                        ids["screwdriver_id"]
                    )[0][2]
                policy.authorise_release()
            policy.step(1.0 / 240.0)
            steps += 1

        out = {
            "final_state": policy.state.value,
            "held_through_present": holding_before_release,
            "grasp_quality": round(policy.grasp_quality, 3),
            "lift_z": round(tool_lift_z, 3) if tool_lift_z else None,
            "events": policy.events,
        }
        p.disconnect()
        return out

    for mode, seed in [
        ("nominal", 1),
        ("camera_shift", 2),
        ("lighting", 3),
        ("novel_geometry", 4),
        ("approach_traj", 5),
    ]:
        r = run_one(mode, seed)
        print(f"=== {mode} ===")
        print(f"  final state         : {r['final_state']}")
        print(f"  held through present : {r['held_through_present']}")
        print(f"  grasp quality       : {r['grasp_quality']}")
        print(f"  tool z while held   : {r['lift_z']} (start ~{round(SCREWDRIVER_START[2], 3)})")
        print()

    print("All five modes ran without error.")
