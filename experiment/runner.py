"""
Experiment runner for the dual-stream handover study.

This is the orchestration core. It runs one handover trial end to end,
stepping the robot policy, the simulated human arm, the confidence
monitor, the intent recogniser, and the fusion controller together, then
records every measured variable to a row in a CSV. It sweeps the full
experimental matrix the methodology fixes, four conditions by four
failure scenario groups at fifty trials per cell, plus a nominal control
group used to measure false positives.

The metrics recorded follow the handover fluency taxonomy of Ortenzi et
al. (2021). Idle time is how long the human waits with a hand presented
before the tool is released. Functional delay is the gap between the
robot being physically ready to release and the release actually
happening. Handover duration is the total time from the human starting
to reach to the tool changing hands. Alongside these the runner logs
intent prediction accuracy at the halfway operating point, whether a
safety intervention occurred, and whether the trial completed.

Determinism is enforced through a single base seed. Each trial derives
its own seed from the base, the condition, the scenario, and the trial
index, so a rerun reproduces every trial exactly. A pilot mode runs a
small number of trials per cell so the protocol can be checked before the
full eight hundred trial sweep.

Citations implemented here
  Ortenzi et al. 2021   idle time, functional delay, handover duration
  Gu et al. 2025        confidence stream in the loop
  Zhang et al. 2024     intent stream in the loop
  Banerjee et al. 2026  graduated fusion in the loop
  Bartlett et al. 2022  fifty trials per cell for adequate HRI power
"""

import csv
import os
import time

import numpy as np
import pybullet as p

from environment.workspace import (
    build_workspace,
    HANDOVER_TARGET,
    SCREWDRIVER_START,
    MALLET_START,
    PENCIL_START,
)
from environment.human_arm import SimulatedHumanArm, SHOULDER_DEFAULT_POSITION
from policy.robot_controller import RobotController
from policy.handover_policy import HandoverPolicy, HandoverState
from monitors.confidence_monitor import ConfidenceMonitor
from monitors.intent_recognition import IntentRecognizer
from fusion.response_controller import ResponseController, ResponseLevel


# The matrix axes, matching the methodology exactly.
CONDITIONS = ("unmonitored", "confidence_only", "intent_only", "dual_stream")
FAILURE_GROUPS = ("camera_shift", "lighting", "novel_geometry", "approach_traj")
NOMINAL_GROUP = "nominal"

# The three tools, cycled across trials so each cell covers all of them.
TOOLS = (
    ("screwdriver_id", SCREWDRIVER_START),
    ("mallet_id", MALLET_START),
    ("pencil_id", PENCIL_START),
)

# Simulation timestep. PyBullet default is 1/240 s.
DT = 1.0 / 240.0

# Candidate reach targets for the intent recogniser. The handover zone is
# first, the rest are distractors a hand could plausibly head for.
INTENT_CANDIDATES = (
    HANDOVER_TARGET,
    [0.20, 0.35, HANDOVER_TARGET[2]],
    [0.20, -0.35, HANDOVER_TARGET[2]],
)

# How long the human reach takes under a nominal approach, in seconds.
NOMINAL_REACH_DURATION = 1.5

# CSV columns, in order.
CSV_FIELDS = [
    "trial_id",
    "condition",
    "scenario",
    "tool",
    "seed",
    "final_state",
    "trial_completed",
    "safety_intervention",
    "any_caution_signal",
    "response_level",
    "policy_confidence",
    "confidence_alarm",
    "intent_confidence_at_half",
    "intent_correct_at_half",
    "grasp_held",
    "idle_time_s",
    "functional_delay_s",
    "handover_duration_s",
    "unsafe_release",
    "false_positive",
]


def derive_seed(base_seed, condition, scenario, trial_index):
    """
    Produce a stable per-trial seed. Mixing the axis labels into the base
    means a rerun with the same base reproduces every trial, and changing
    one axis does not shift the others' streams.
    """
    key = f"{base_seed}|{condition}|{scenario}|{trial_index}"
    # A simple stable hash. Python's hash is salted per process, so we use
    # a deterministic fold over the bytes instead.
    h = 1469598103934665603
    for byte in key.encode("utf-8"):
        h ^= byte
        h = (h * 1099511628211) % (2**63)
    return h


def run_trial(condition, scenario, tool_key, tool_start, monitor, seed):
    """
    Run a single handover trial and return a dict of measured variables.

    Parameters
    ----------
    condition : str
        One of CONDITIONS.
    scenario : str
        A failure group or NOMINAL_GROUP.
    tool_key : str
        Key into the workspace ids for the tool to hand over.
    tool_start : list of 3 floats
        Tool start position.
    monitor : ConfidenceMonitor
        A calibrated confidence monitor, shared across trials.
    seed : int
        Deterministic seed for this trial.

    Returns
    -------
    dict
        One row of measurements keyed by CSV_FIELDS.
    """
    rng = np.random.default_rng(seed)

    ids = build_workspace(gui=False)
    controller = RobotController(ids["robot_id"])
    arm = SimulatedHumanArm(height_metres=float(rng.uniform(1.55, 1.90)))
    arm.build_visuals()

    # Let the world settle so tools rest on the table.
    for _ in range(120):
        p.stepSimulation()

    policy = HandoverPolicy(
        controller,
        robot_id=ids["robot_id"],
        tool_id=ids[tool_key],
        tool_start_pos=tool_start,
        handover_target=HANDOVER_TARGET,
        failure_mode=scenario,
        rng=rng,
    )

    recog = IntentRecognizer(list(INTENT_CANDIDATES), handover_index=0)
    fusion = ResponseController(condition=condition)

    # The human approach. Under approach_traj the reach is perturbed,
    # arriving early and off the expected axis, which stresses release
    # timing. Otherwise the reach is the nominal minimum-jerk reach.
    reach_target = list(HANDOVER_TARGET)
    reach_duration = NOMINAL_REACH_DURATION
    if scenario == "approach_traj":
        reach_duration = float(rng.uniform(0.7, 1.0))   # arrives early
        reach_target[1] += float(rng.uniform(-0.12, 0.12))  # off-axis
        reach_target[0] += float(rng.uniform(-0.08, 0.08))

    # Trackers for the fluency metrics, all in seconds of human-reach time.
    present_start_time = None     # when the robot first presented
    robot_ready_time = None       # when the robot was physically ready
    release_time = None           # when the tool was released
    intent_at_half = None
    intent_correct_at_half = None
    half_captured = False

    last_decision = {"level": ResponseLevel.PROCEED, "intervention": False}
    intervention_occurred = False
    final_policy_conf = 1.0
    final_alarm = False

    # Phase one. Drive the robot through pick, lift, transport, and into
    # presenting. These policy steps each run blocking moves, so the robot
    # is physically ready by the time it reaches PRESENTING. The human has
    # not started reaching yet, which matches a human waiting for the robot
    # to offer the tool.
    macro_guard = 0
    while policy.state != HandoverState.PRESENTING and not policy.is_done():
        policy.step(DT)
        macro_guard += 1
        if macro_guard > 20:
            break

    internal = policy.get_internal_state()
    assessment = monitor.assess(internal)
    final_policy_conf = assessment["confidence"]
    final_alarm = assessment["alarm"]

    # Whether the grasp physically held at the moment of presentation,
    # captured before any release flips the flag. This is the ground truth
    # the safety logic needs. A trial that presents without actually
    # holding the tool is one a safe controller should refuse to complete.
    grasp_held = bool(policy.holding_tool)

    # Phase two. The robot is presenting and physically ready. The human
    # now reaches over real timesteps while the fusion controller polls
    # each step. Release happens when the controller says PROCEED, or never
    # if it aborts. This is where idle time and functional delay accrue.
    if policy.state == HandoverState.PRESENTING:
        present_start_time = 0.0
        robot_ready_time = 0.0
        arm.start_reach(reach_target, duration_seconds=reach_duration)

        reach_elapsed = 0.0
        # Allow the reach to play out plus a margin for the human to settle.
        max_present_steps = int((reach_duration + 1.0) / DT)
        for _ in range(max_present_steps):
            arm.step(DT)
            recog.update(arm.get_hand_position())
            reach_elapsed += DT

            reach_fraction = min(reach_elapsed / reach_duration, 1.0) \
                if reach_duration > 0 else 1.0
            if not half_captured and reach_fraction >= 0.5:
                pred = recog.predict()
                intent_at_half = pred["confidence"]
                intent_correct_at_half = (
                    pred["predicted_index"] is not None
                    and recog.is_correct(pred["predicted_index"])
                )
                half_captured = True

            pred = recog.predict()
            decision = fusion.decide(
                policy_confidence=assessment["confidence"],
                intent_confidence=pred["confidence"],
                intent_correct=(
                    pred["predicted_index"] is not None
                    and recog.is_correct(pred["predicted_index"])
                ),
            )
            last_decision = decision
            if decision["intervention"]:
                intervention_occurred = True
            if decision["level"] == ResponseLevel.PROCEED:
                policy.authorise_release()
                release_time = reach_elapsed
                break
            if decision["level"] == ResponseLevel.ABORT:
                # A confirmed abort ends the attempt without a release.
                break

        # Step the policy once more so a granted release actually fires the
        # RELEASING and DONE transitions.
        if policy.release_authorised:
            policy.step(DT)   # RELEASING
            policy.step(DT)   # DONE

    # If the half point was never reached (the reach was cut short by an
    # immediate abort), capture the current prediction.
    if not half_captured:
        pred = recog.predict()
        intent_at_half = pred["confidence"] if pred["predicted_index"] is not None else 0.0
        intent_correct_at_half = (
            pred["predicted_index"] is not None and recog.is_correct(pred["predicted_index"])
        )

    # A trial completes when the tool was actually released to the human.
    trial_completed = (
        policy.state == HandoverState.DONE
        and any(e[0] == "released" for e in policy.events)
    )

    # A safety intervention is the controller preventing a completion, by
    # aborting or by holding the handover open until the attempt ends
    # without a release. A transient WAIT that later proceeds is normal
    # operation, not an intervention, so we key this on the trial failing
    # to complete rather than on any non-proceed response ever appearing.
    safety_intervention = bool(not trial_completed)

    # Idle time is how long the human waited from arriving at the handover
    # to the tool being released. Functional delay is the gap between the
    # robot being ready (start of presenting) and release. Handover
    # duration is the whole reach-to-release span. With the human reach
    # starting at the moment the robot presents, present_start_time is zero
    # and the three measures are read off the reach clock.
    if present_start_time is not None and release_time is not None:
        functional_delay = release_time - robot_ready_time
        handover_duration = release_time
        # Idle time is the wait after the hand has effectively arrived,
        # taken as the time past the halfway operating point of the reach.
        idle_time = max(release_time - 0.5 * reach_duration, 0.0)
    else:
        idle_time = ""
        functional_delay = ""
        handover_duration = ""

    # An unsafe release is one that completed while the grasp had actually
    # failed. This is the safety event the framework is meant to prevent.
    # A false positive is the mirror image, a nominal trial that a healthy
    # grasp should have completed but the controller blocked.
    actually_failing = not grasp_held
    unsafe_release = bool(trial_completed and actually_failing)
    false_positive = bool(
        scenario == NOMINAL_GROUP and grasp_held and not trial_completed
    )

    p.disconnect()

    return {
        "trial_id": "",  # filled by caller
        "condition": condition,
        "scenario": scenario,
        "tool": tool_key.replace("_id", ""),
        "seed": seed,
        "final_state": policy.state.value,
        "trial_completed": int(trial_completed),
        "safety_intervention": int(safety_intervention),
        "any_caution_signal": int(intervention_occurred),
        "response_level": last_decision["level"].value,
        "policy_confidence": round(final_policy_conf, 4),
        "confidence_alarm": int(final_alarm),
        "intent_confidence_at_half": round(intent_at_half, 4) if intent_at_half is not None else "",
        "intent_correct_at_half": int(bool(intent_correct_at_half)),
        "grasp_held": int(bool(grasp_held)),
        "idle_time_s": round(idle_time, 4) if idle_time != "" else "",
        "functional_delay_s": round(functional_delay, 4) if functional_delay != "" else "",
        "handover_duration_s": round(handover_duration, 4) if handover_duration != "" else "",
        "unsafe_release": int(unsafe_release),
        "false_positive": int(false_positive),
    }


def build_calibrated_monitor(base_seed, num_nominal=40):
    """
    Run a pool of nominal trials and calibrate the confidence monitor on
    them. The pool is separate from the experimental trials so calibration
    does not peek at test data. The monitor it returns is shared across
    all conditions, which is correct since the monitor is condition
    agnostic.
    """
    from monitors.confidence_monitor import collect_nominal_states

    states = collect_nominal_states(num_trials=num_nominal, seed=base_seed + 99991)
    monitor = ConfidenceMonitor()
    monitor.calibrate(states, target_fpr=0.10)
    return monitor


def run_experiment(
    trials_per_cell=50,
    base_seed=2026,
    output_dir="data",
    include_nominal=True,
    pilot=False,
):
    """
    Sweep the full matrix and write a CSV.

    Parameters
    ----------
    trials_per_cell : int
        Trials per condition by scenario cell. The methodology fixes 50.
    base_seed : int
        Base seed for determinism.
    output_dir : str
        Directory for the CSV, created if missing.
    include_nominal : bool
        Whether to run the nominal control group for each condition.
    pilot : bool
        If True, overrides trials_per_cell to a small number and tags the
        output filename as a pilot.

    Returns
    -------
    str
        Path to the written CSV.
    """
    if pilot:
        trials_per_cell = 5

    os.makedirs(output_dir, exist_ok=True)
    tag = "pilot" if pilot else "full"
    out_path = os.path.join(output_dir, f"handover_results_{tag}.csv")

    print(f"Calibrating confidence monitor on a nominal pool...")
    monitor = build_calibrated_monitor(base_seed)
    print(f"  threshold set, target FPR 10%")

    scenarios = list(FAILURE_GROUPS)
    if include_nominal:
        scenarios = scenarios + [NOMINAL_GROUP]

    total_cells = len(CONDITIONS) * len(scenarios)
    total_trials = total_cells * trials_per_cell
    print(
        f"\nRunning {len(CONDITIONS)} conditions x {len(scenarios)} scenarios "
        f"x {trials_per_cell} trials = {total_trials} trials"
    )

    start = time.time()
    rows = []
    trial_counter = 0
    done = 0

    for condition in CONDITIONS:
        for scenario in scenarios:
            for k in range(trials_per_cell):
                tool_key, tool_start = TOOLS[k % len(TOOLS)]
                seed = derive_seed(base_seed, condition, scenario, k)
                row = run_trial(
                    condition, scenario, tool_key, tool_start, monitor, seed
                )
                row["trial_id"] = trial_counter
                rows.append(row)
                trial_counter += 1
                done += 1

                if done % 25 == 0 or done == total_trials:
                    elapsed = time.time() - start
                    rate = done / elapsed
                    remaining = (total_trials - done) / rate if rate > 0 else 0
                    print(
                        f"  {done}/{total_trials} trials "
                        f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)"
                    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - start
    print(f"\nDone. {total_trials} trials in {elapsed:.0f}s "
          f"({elapsed / total_trials:.2f}s per trial)")
    print(f"Wrote {out_path}")

    # A quick per-condition summary so problems show up early.
    print("\nSafety intervention rate by condition (failure scenarios only):")
    for condition in CONDITIONS:
        cell = [
            r for r in rows
            if r["condition"] == condition and r["scenario"] != NOMINAL_GROUP
        ]
        if cell:
            rate = sum(r["safety_intervention"] for r in cell) / len(cell)
            print(f"  {condition:16s} {rate:.1%}")

    print("\nUnsafe completion rate by condition (failure scenarios only):")
    for condition in CONDITIONS:
        cell = [
            r for r in rows
            if r["condition"] == condition and r["scenario"] != NOMINAL_GROUP
        ]
        if cell:
            rate = sum(r["unsafe_release"] for r in cell) / len(cell)
            print(f"  {condition:16s} {rate:.1%}")

    if include_nominal:
        print("\nNominal completion rate and false positive rate by condition:")
        for condition in CONDITIONS:
            cell = [
                r for r in rows
                if r["condition"] == condition and r["scenario"] == NOMINAL_GROUP
            ]
            if cell:
                comp = sum(r["trial_completed"] for r in cell) / len(cell)
                fp = sum(r["false_positive"] for r in cell) / len(cell)
                print(f"  {condition:16s} completion {comp:.1%}  false positive {fp:.1%}")

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the handover experiment.")
    parser.add_argument(
        "--pilot", action="store_true",
        help="Run a small pilot of 5 trials per cell to check the protocol.",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Trials per cell for the full run (default 50).",
    )
    parser.add_argument(
        "--seed", type=int, default=2026, help="Base seed.",
    )
    args = parser.parse_args()

    run_experiment(
        trials_per_cell=args.trials,
        base_seed=args.seed,
        pilot=args.pilot,
    )
