"""
Confidence monitor for the handover policy.

This is the VLA failure-detection stream of the dual-stream framework. In
the methodology it follows the SAFE design of Gu et al. (2025), which
trains a small monitor on a model's internal activations and emits a
scalar that tracks the probability of task failure. Under Path A there is
no learned VLA, so the monitor reads the scripted policy's internal state
instead of a hidden layer. The signals it consumes are the Path A analogue
of the activation features a real SAFE monitor would see, namely the
policy's own estimate of grasp quality, the perceived versus true tool
offset, perception noise, and geometry mismatch.

Calibration follows the conformal approach of Xu et al. (2025) and
Angelopoulos and Bates (2023). A pool of nominal trials gives a
distribution of raw monitor scores. The decision threshold is the chosen
quantile of that nominal distribution, so the false positive rate on
nominal data is controlled directly and no failure data is needed to set
it. This is the FAIL-Detect property that matters for the dissertation,
since labelled handover failures are scarce.

The monitor produces two things each step. A continuous confidence score
in [0, 1], where high means the monitor believes the handover is going
well. And a binary alarm, raised when confidence falls below the
calibrated threshold.

Citations implemented here
  Gu et al. 2025               SAFE monitor over internal policy signals
  Xu et al. 2025               conformal calibration without failure data
  Angelopoulos and Bates 2023  split-conformal quantile threshold
"""

import numpy as np


# Weights on the raw risk signals. A larger weight means the signal pulls
# confidence down harder. These are fixed design choices, not learned,
# which keeps the monitor rule-based as the methodology requires.
RISK_WEIGHTS = {
    "grasp_deficit": 1.0,     # how far grasp quality sits below perfect
    "perceived_offset": 8.0,  # metres of perception error, scaled up
    "perception_noise": 0.6,  # raised under lighting variation
    "geometry_mismatch": 0.8, # raised under novel tool geometry
}

# Default target false positive rate on nominal data. The threshold is
# set so that this fraction of nominal trials would raise an alarm.
DEFAULT_TARGET_FPR = 0.10


class ConfidenceMonitor:
    """
    Observes policy internal state and reports a calibrated confidence.

    The monitor is calibrated once on a pool of nominal trials, then used
    unchanged across all conditions. Calibration sets a single threshold
    on the raw risk score.

    Usage
    -----
        monitor = ConfidenceMonitor()
        monitor.calibrate(nominal_scores, target_fpr=0.10)
        result = monitor.assess(policy.get_internal_state())
        if result["alarm"]:
            ...
    """

    def __init__(self, weights=None):
        """
        Parameters
        ----------
        weights : dict, optional
            Override the default risk weights. Keys must match
            RISK_WEIGHTS.
        """
        self.weights = dict(weights) if weights else dict(RISK_WEIGHTS)
        self.threshold = None       # raw risk score above which we alarm
        self.calibrated = False
        self.target_fpr = DEFAULT_TARGET_FPR

    def raw_risk(self, internal_state):
        """
        Combine the policy's internal signals into a single raw risk score.

        Higher means more likely to be failing. The score is a weighted
        sum of deficits, so it is monotone in each signal and easy to
        defend in the write-up.

        Parameters
        ----------
        internal_state : dict
            Output of HandoverPolicy.get_internal_state.

        Returns
        -------
        float
            Non-negative raw risk.
        """
        grasp_deficit = 1.0 - internal_state.get("grasp_quality", 1.0)
        offset = internal_state.get("perceived_offset", 0.0)
        noise = internal_state.get("perception_noise", 0.0)
        mismatch = internal_state.get("geometry_mismatch", 0.0)

        risk = (
            self.weights["grasp_deficit"] * grasp_deficit
            + self.weights["perceived_offset"] * offset
            + self.weights["perception_noise"] * noise
            + self.weights["geometry_mismatch"] * mismatch
        )
        return float(max(risk, 0.0))

    def calibrate(self, nominal_internal_states, target_fpr=DEFAULT_TARGET_FPR):
        """
        Set the alarm threshold from a pool of nominal trials.

        We compute the raw risk for each nominal state, then take the
        quantile that leaves target_fpr of nominal trials above it. Any
        future trial scoring above that threshold raises an alarm. On
        nominal data this holds the false positive rate near target_fpr
        by construction (Angelopoulos and Bates 2023).

        Parameters
        ----------
        nominal_internal_states : list of dict
            Internal states sampled from nominal trials. Each is passed
            through raw_risk.
        target_fpr : float
            Desired nominal false positive rate, in (0, 1).
        """
        if not 0.0 < target_fpr < 1.0:
            raise ValueError("target_fpr must be in (0, 1)")
        if len(nominal_internal_states) < 5:
            raise ValueError(
                "need at least five nominal states to calibrate, got "
                f"{len(nominal_internal_states)}"
            )

        scores = np.array(
            [self.raw_risk(s) for s in nominal_internal_states], dtype=float
        )
        # The threshold is the (1 - target_fpr) quantile of nominal risk.
        # Trials above it are flagged, so about target_fpr of nominal
        # trials fall above by construction.
        self.threshold = float(np.quantile(scores, 1.0 - target_fpr))
        self.target_fpr = float(target_fpr)
        self.calibrated = True
        return self.threshold

    def assess(self, internal_state):
        """
        Score a single policy state and decide whether to alarm.

        Parameters
        ----------
        internal_state : dict
            Output of HandoverPolicy.get_internal_state.

        Returns
        -------
        dict
            raw_risk, confidence in [0, 1], and alarm bool.
        """
        if not self.calibrated:
            raise RuntimeError("calibrate the monitor before calling assess")

        risk = self.raw_risk(internal_state)

        # Map raw risk to a confidence in [0, 1] for reporting and fusion.
        # We anchor the midpoint at the threshold, so confidence is 0.5
        # exactly where the alarm fires. A logistic squashing keeps it
        # bounded and smooth.
        confidence = float(1.0 / (1.0 + np.exp(self._logit_scale() * (risk - self.threshold))))

        return {
            "raw_risk": risk,
            "confidence": confidence,
            "alarm": bool(risk > self.threshold),
        }

    def _logit_scale(self):
        """
        Slope of the logistic mapping from risk to confidence. Scaled to
        the threshold so the curve has a sensible spread regardless of the
        absolute risk magnitudes. A small floor avoids a divide-by-zero
        when the nominal threshold is near zero.
        """
        return 4.0 / max(self.threshold, 0.05)


def collect_nominal_states(num_trials=40, seed=0):
    """
    Run a pool of nominal handovers and gather their internal states at
    the grasp confirmation point, where the monitor's signal is richest.

    This helper exists so the monitor can be calibrated on demand. The
    experiment runner builds its own calibration pool, but the smoke test
    and any quick check can use this.

    Parameters
    ----------
    num_trials : int
        How many nominal trials to run.
    seed : int
        Base seed for reproducibility.

    Returns
    -------
    list of dict
        One internal state per trial.
    """
    import pybullet as p
    from environment.workspace import (
        build_workspace,
        HANDOVER_TARGET,
        SCREWDRIVER_START,
    )
    from policy.robot_controller import RobotController
    from policy.handover_policy import HandoverPolicy, HandoverState

    states = []
    for i in range(num_trials):
        ids = build_workspace(gui=False)
        controller = RobotController(ids["robot_id"])
        rng = np.random.default_rng(seed + i)
        for _ in range(120):
            p.stepSimulation()

        policy = HandoverPolicy(
            controller,
            robot_id=ids["robot_id"],
            tool_id=ids["screwdriver_id"],
            tool_start_pos=SCREWDRIVER_START,
            handover_target=HANDOVER_TARGET,
            failure_mode="nominal",
            rng=rng,
        )

        captured = None
        steps = 0
        while not policy.is_done() and steps < 40:
            if policy.state == HandoverState.PRESENTING:
                policy.authorise_release()
            policy.step(1.0 / 240.0)
            # Capture once the grasp has been confirmed, during carry.
            if captured is None and policy.state in (
                HandoverState.LIFTING,
                HandoverState.TRANSPORTING,
            ):
                captured = policy.get_internal_state()
            steps += 1

        states.append(captured if captured else policy.get_internal_state())
        p.disconnect()

    return states


if __name__ == "__main__":
    # Manual smoke test. Calibrate on a small nominal pool, then check the
    # monitor stays quiet on nominal states and alarms on degraded ones.
    print("Collecting nominal calibration pool (this runs the sim)...")
    nominal_states = collect_nominal_states(num_trials=30, seed=100)

    monitor = ConfidenceMonitor()
    threshold = monitor.calibrate(nominal_states, target_fpr=0.10)
    print(f"Calibrated threshold (raw risk): {threshold:.4f}")

    # False positive rate on the calibration pool itself.
    alarms = [monitor.assess(s)["alarm"] for s in nominal_states]
    fpr = sum(alarms) / len(alarms)
    print(f"Alarm rate on nominal pool: {fpr:.2%} (target was 10%)")

    # Synthetic degraded states, the kind failure modes produce.
    print("\nMonitor response to degraded states:")
    degraded_examples = {
        "mild grasp dip": {
            "grasp_quality": 0.8, "perceived_offset": 0.0,
            "perception_noise": 0.0, "geometry_mismatch": 0.0,
        },
        "lighting noise": {
            "grasp_quality": 0.6, "perceived_offset": 0.0,
            "perception_noise": 0.5, "geometry_mismatch": 0.0,
        },
        "camera offset": {
            "grasp_quality": 0.5, "perceived_offset": 0.04,
            "perception_noise": 0.0, "geometry_mismatch": 0.0,
        },
        "novel geometry": {
            "grasp_quality": 0.55, "perceived_offset": 0.0,
            "perception_noise": 0.0, "geometry_mismatch": 0.4,
        },
    }
    for name, state in degraded_examples.items():
        r = monitor.assess(state)
        flag = "ALARM" if r["alarm"] else "quiet"
        print(
            f"  {name:16s} risk={r['raw_risk']:.3f} "
            f"confidence={r['confidence']:.2f} -> {flag}"
        )

    print("\nConfidence monitor smoke test complete.")
