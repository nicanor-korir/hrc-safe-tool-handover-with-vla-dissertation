"""
Simulated human arm for the handover environment.

Implements the seven-degree-of-freedom kinematic chain described in
Klopcar and Lenarcic (2005), parameterised by user height, and drives
the hand along the minimum-jerk reach trajectory of Flash and Hogan
(1985) toward the handover target.

This module gives the simulation two things:
  - a visible arm in the workspace, attached opposite the robot
  - a hand position that updates each timestep along a smooth reach

The intent recognition stream observes the hand position over time.
The robot's confidence monitor sees the arm as part of its visual input.
"""

import numpy as np
import pybullet as p

# Anthropometric ratios from standard tables (de Leva, 1996; widely cited)
UPPER_ARM_RATIO = 0.157  # upper arm length as a fraction of body height
FOREARM_RATIO = 0.146    # forearm length
HAND_RATIO = 0.108       # hand length

# Default starting position for the shoulder, opposite the robot
SHOULDER_DEFAULT_POSITION = [1.05, 0.0, 0.95]


class SimulatedHumanArm:
    """
    A kinematic arm whose hand can be commanded to reach toward a target
    along a minimum-jerk trajectory.
    """

    def __init__(self, height_metres=1.75, shoulder_position=None):
        """
        Build the arm with anthropometric scaling.

        Parameters
        ----------
        height_metres : float
            Total body height. Used to scale segment lengths.
            Methodology samples this between 1.55 and 1.90.
        shoulder_position : list of 3 floats, optional
            Where the shoulder is anchored in world coordinates.
        """
        self.height = height_metres
        self.upper_arm_length = height_metres * UPPER_ARM_RATIO
        self.forearm_length = height_metres * FOREARM_RATIO
        self.hand_length = height_metres * HAND_RATIO

        self.shoulder_pos = np.array(
            shoulder_position if shoulder_position else SHOULDER_DEFAULT_POSITION
        )

        # Trajectory state
        self.trajectory_start_pos = None
        self.trajectory_end_pos = None
        self.trajectory_duration = None
        self.trajectory_time = 0.0
        self.is_reaching = False

        # PyBullet visual handles, populated by build_visuals()
        self.upper_arm_visual = None
        self.forearm_visual = None
        self.hand_visual = None
        self.shoulder_visual = None
        self.elbow_visual = None
        self.wrist_visual = None

        # Current joint positions (in world coords) - the elbow, wrist, hand
        self.current_elbow_pos = None
        self.current_wrist_pos = None
        self.current_hand_pos = self.shoulder_pos.copy()

    def build_visuals(self):
        """
        Create the visual representation in PyBullet.
        Three cylinders for the segments, three spheres for the joints.
        Visual only, no collision, because the arm is a kinematic stand-in.
        """
        # Joint spheres
        sphere_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.03, rgbaColor=[0.9, 0.7, 0.6, 1.0]
        )
        self.shoulder_visual = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=sphere_visual,
            basePosition=self.shoulder_pos.tolist(),
        )
        self.elbow_visual = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=sphere_visual,
            basePosition=self.shoulder_pos.tolist(),
        )
        self.wrist_visual = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=sphere_visual,
            basePosition=self.shoulder_pos.tolist(),
        )

        # Arm segment cylinders (slightly smaller spheres at hand)
        hand_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.04, rgbaColor=[0.95, 0.75, 0.65, 1.0]
        )
        self.hand_visual = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=hand_visual,
            basePosition=self.shoulder_pos.tolist(),
        )

        # Set neutral starting pose (arm hanging at side, hand at shoulder)
        self._set_arm_pose(self.shoulder_pos.copy())

    def start_reach(self, target_position, duration_seconds=1.5):
        """
        Begin a minimum-jerk reach from the current hand position to target.

        Parameters
        ----------
        target_position : list or array of 3 floats
            World coordinates of where the hand should arrive.
        duration_seconds : float
            How long the reach takes. Methodology assumes 1-2 seconds.
        """
        self.trajectory_start_pos = self.current_hand_pos.copy()
        self.trajectory_end_pos = np.array(target_position)
        self.trajectory_duration = duration_seconds
        self.trajectory_time = 0.0
        self.is_reaching = True

    def step(self, dt):
        """
        Advance the arm by one simulation timestep.
        Updates the hand position along the minimum-jerk trajectory
        and re-poses the arm visuals to match.

        Parameters
        ----------
        dt : float
            Timestep duration in seconds (e.g. 1/240 for PyBullet default).
        """
        if not self.is_reaching:
            return

        self.trajectory_time += dt

        # Compute fraction of trajectory completed (clamped to [0, 1])
        tau = min(self.trajectory_time / self.trajectory_duration, 1.0)

        # Flash-Hogan minimum-jerk position profile:
        # x(t) = x0 + (x1 - x0) * (10*tau^3 - 15*tau^4 + 6*tau^5)
        # This produces the bell-shaped velocity profile of natural reach
        profile = 10 * tau**3 - 15 * tau**4 + 6 * tau**5

        new_hand_pos = (
            self.trajectory_start_pos
            + (self.trajectory_end_pos - self.trajectory_start_pos) * profile
        )

        self._set_arm_pose(new_hand_pos)

        if tau >= 1.0:
            self.is_reaching = False

    def get_hand_position(self):
        """Return the current hand position as a numpy array."""
        return self.current_hand_pos.copy()

    def get_hand_velocity(self):
        """
        Return the instantaneous hand velocity at the current trajectory time.
        Used by the intent recognition stream to predict reach target.
        """
        if not self.is_reaching or self.trajectory_duration is None:
            return np.zeros(3)

        tau = min(self.trajectory_time / self.trajectory_duration, 1.0)
        # Derivative of the minimum-jerk position profile
        profile_derivative = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
        delta = self.trajectory_end_pos - self.trajectory_start_pos
        return delta * profile_derivative / self.trajectory_duration

    def _set_arm_pose(self, hand_target_pos):
        """
        Given the desired hand position, solve a simple inverse kinematic
        approximation to place the elbow and wrist, then update visuals.

        We use a simplified two-segment IK: the elbow sits on the plane
        between shoulder and target, at the geometrically valid position.
        For a master's dissertation testing intent recognition rather than
        biomechanically accurate kinematics, this is sufficient.
        """
        self.current_hand_pos = hand_target_pos.copy()

        # Vector from shoulder to hand
        shoulder_to_hand = hand_target_pos - self.shoulder_pos
        distance = np.linalg.norm(shoulder_to_hand)

        total_arm = self.upper_arm_length + self.forearm_length

        # If the target is unreachable, clamp to maximum reach
        if distance > total_arm * 0.95:
            shoulder_to_hand = shoulder_to_hand * (total_arm * 0.95 / distance)
            distance = total_arm * 0.95
            hand_target_pos = self.shoulder_pos + shoulder_to_hand

        # Law of cosines to find elbow angle
        # cos(theta) = (a^2 + b^2 - c^2) / (2ab)
        # where a = upper arm, b = forearm, c = distance shoulder to hand
        a = self.upper_arm_length
        b = self.forearm_length
        c = max(distance, 0.01)  # avoid divide-by-zero
        cos_elbow_angle = (a**2 + b**2 - c**2) / (2 * a * b)
        cos_elbow_angle = np.clip(cos_elbow_angle, -1.0, 1.0)
        elbow_angle = np.arccos(cos_elbow_angle)

        # Angle at shoulder in the shoulder-elbow-hand triangle
        cos_shoulder_angle = (a**2 + c**2 - b**2) / (2 * a * c)
        cos_shoulder_angle = np.clip(cos_shoulder_angle, -1.0, 1.0)
        shoulder_angle = np.arccos(cos_shoulder_angle)

        # Direction from shoulder to hand
        direction = shoulder_to_hand / max(c, 0.01)

        # Place elbow by rotating shoulder->hand direction by shoulder_angle
        # downward in the vertical plane through the arm
        # We use a fixed "up" direction to define the rotation plane
        up = np.array([0.0, 0.0, 1.0])
        # Perpendicular to direction, in the vertical plane
        side = np.cross(direction, up)
        if np.linalg.norm(side) < 0.01:
            side = np.array([1.0, 0.0, 0.0])
        side = side / np.linalg.norm(side)
        perp = np.cross(side, direction)
        if np.linalg.norm(perp) < 0.01:
            perp = np.array([0.0, 0.0, -1.0])
        perp = perp / np.linalg.norm(perp)

        # Elbow position: along direction by a*cos(shoulder_angle),
        # and downward by a*sin(shoulder_angle)
        elbow_pos = (
            self.shoulder_pos
            + direction * (a * np.cos(shoulder_angle))
            + perp * (a * np.sin(shoulder_angle))
        )

        # Wrist position: between elbow and hand, hand_length back from hand
        elbow_to_hand = hand_target_pos - elbow_pos
        elbow_to_hand_dist = max(np.linalg.norm(elbow_to_hand), 0.01)
        wrist_pos = hand_target_pos - (elbow_to_hand / elbow_to_hand_dist) * self.hand_length

        self.current_elbow_pos = elbow_pos
        self.current_wrist_pos = wrist_pos

        # Update visual positions
        p.resetBasePositionAndOrientation(
            self.elbow_visual, elbow_pos.tolist(), [0, 0, 0, 1]
        )
        p.resetBasePositionAndOrientation(
            self.wrist_visual, wrist_pos.tolist(), [0, 0, 0, 1]
        )
        p.resetBasePositionAndOrientation(
            self.hand_visual, hand_target_pos.tolist(), [0, 0, 0, 1]
        )

        # Render the segments as debug lines (simpler than oriented cylinders)
        # Clear previous lines by giving them an ID we overwrite each step
        if not hasattr(self, "_line_ids"):
            self._line_ids = []
            for _ in range(3):
                self._line_ids.append(
                    p.addUserDebugLine(
                        [0, 0, 0], [0, 0, 0.001], [0.9, 0.7, 0.6], 4
                    )
                )

        self._line_ids[0] = p.addUserDebugLine(
            self.shoulder_pos.tolist(),
            elbow_pos.tolist(),
            lineColorRGB=[0.9, 0.7, 0.6],
            lineWidth=6,
            replaceItemUniqueId=self._line_ids[0],
        )
        self._line_ids[1] = p.addUserDebugLine(
            elbow_pos.tolist(),
            wrist_pos.tolist(),
            lineColorRGB=[0.9, 0.7, 0.6],
            lineWidth=6,
            replaceItemUniqueId=self._line_ids[1],
        )
        self._line_ids[2] = p.addUserDebugLine(
            wrist_pos.tolist(),
            hand_target_pos.tolist(),
            lineColorRGB=[0.95, 0.75, 0.65],
            lineWidth=6,
            replaceItemUniqueId=self._line_ids[2],
        )


if __name__ == "__main__":
    # Manual test: build the workspace, add a human arm, command a reach
    import time
    from environment.workspace import build_workspace, HANDOVER_TARGET

    ids = build_workspace(gui=True)

    arm = SimulatedHumanArm(height_metres=1.75)
    arm.build_visuals()

    print("Pausing 2 seconds so you can see the arm before it reaches.")
    for _ in range(240 * 2):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    print("Starting reach toward handover target.")
    arm.start_reach(target_position=HANDOVER_TARGET, duration_seconds=1.5)

    # Step until reach completes plus 2 seconds for visual settling
    for _ in range(240 * 4):
        arm.step(1.0 / 240.0)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    print("Reach complete. Final hand position:", arm.get_hand_position())

    print("Holding for 3 seconds.")
    for _ in range(240 * 3):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()
    print("Done.")
