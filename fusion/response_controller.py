"""
Rule-based fusion controller for the dual-stream handover framework.

Two streams feed this controller. The confidence monitor reports whether
the robot policy looks like it is failing. The intent recogniser reports
whether the human looks ready to receive. Banerjee et al. (2026) describe
a human-in-the-loop confidence-aware scheme where these signals combine
into a graduated response rather than a single stop-or-go switch. This
module implements that logic with fixed thresholds, which keeps it
rule-based as the methodology requires and avoids needing failure data to
train a fusion model.

The output is one of four response levels.

  PROCEED   policy confident and human ready, release the tool
  WAIT      policy confident but human not yet ready, hold and keep
            presenting until intent settles
  SLOW      policy confidence marginal, ease off and reassess rather
            than commit to a release
  ABORT     policy confidence low, stop the handover and do not release

The four experimental conditions are realised by which streams the
controller is allowed to listen to. This is the ablation.

  unmonitored     ignore both streams, always PROCEED once presenting
  confidence_only listen to the confidence monitor only
  intent_only     listen to the intent recogniser only
  dual_stream     listen to both, the full framework

A safety intervention is any response that is not PROCEED, since each of
those holds, slows, or aborts the release. The experiment counts these.

Citations implemented here
  Banerjee et al. 2026   graduated confidence-aware human-in-the-loop
  Gu et al. 2025         confidence stream feeding the fusion rule
  Zhang et al. 2024      intent stream feeding the fusion rule
"""

import enum


class ResponseLevel(enum.Enum):
    """The four graduated responses the controller can command."""

    PROCEED = "PROCEED"
    WAIT = "WAIT"
    SLOW = "SLOW"
    ABORT = "ABORT"


# The four experimental conditions, matching the methodology ablation.
CONDITIONS = (
    "unmonitored",
    "confidence_only",
    "intent_only",
    "dual_stream",
)

# Confidence bands on the policy confidence stream. Above the high band
# the policy is trusted. Below the low band the handover is aborted.
# Between them the controller eases off.
CONFIDENCE_HIGH = 0.5    # at or above this, policy is trusted
CONFIDENCE_LOW = 0.2     # below this, abort

# Intent confidence the human must reach before a release is allowed.
# Set near the halfway operating point the recogniser is tuned to.
INTENT_READY = 0.6


class ResponseController:
    """
    Fuses the two streams into a graduated response under a given
    condition. The condition fixes which streams are active.

    Usage
    -----
        controller = ResponseController(condition="dual_stream")
        decision = controller.decide(
            policy_confidence=conf, intent_confidence=intent,
            intent_correct=is_correct,
        )
        if decision["level"] == ResponseLevel.PROCEED:
            policy.authorise_release()
    """

    def __init__(
        self,
        condition="dual_stream",
        confidence_high=CONFIDENCE_HIGH,
        confidence_low=CONFIDENCE_LOW,
        intent_ready=INTENT_READY,
    ):
        """
        Parameters
        ----------
        condition : str
            One of CONDITIONS. Selects which streams are listened to.
        confidence_high, confidence_low : float
            Band edges on the policy confidence stream.
        intent_ready : float
            Intent confidence required before release is allowed.
        """
        if condition not in CONDITIONS:
            raise ValueError(
                f"condition must be one of {CONDITIONS}, got {condition!r}"
            )
        self.condition = condition
        self.confidence_high = confidence_high
        self.confidence_low = confidence_low
        self.intent_ready = intent_ready

    def decide(self, policy_confidence, intent_confidence, intent_correct=True):
        """
        Produce a graduated response from the current stream values.

        Parameters
        ----------
        policy_confidence : float
            Confidence in [0, 1] from the confidence monitor. High means
            the policy looks healthy.
        intent_confidence : float
            Confidence in [0, 1] from the intent recogniser that the human
            is reaching for the handover.
        intent_correct : bool
            Whether the intent prediction points at the true handover
            target. A confident but wrong prediction should not open the
            gate, so the controller treats a wrong call as not ready.

        Returns
        -------
        dict
            level, an intervention bool, and the reason string.
        """
        if self.condition == "unmonitored":
            return self._decide_unmonitored()
        if self.condition == "confidence_only":
            return self._decide_confidence_only(policy_confidence)
        if self.condition == "intent_only":
            return self._decide_intent_only(intent_confidence, intent_correct)
        return self._decide_dual(
            policy_confidence, intent_confidence, intent_correct
        )

    # ----- per-condition rules ---------------------------------------------

    def _decide_unmonitored(self):
        """
        Baseline. No streams are consulted, so the controller always
        releases as soon as the policy presents. This is the unmonitored
        VLA condition the others are measured against.
        """
        return self._result(ResponseLevel.PROCEED, "no monitoring, release")

    def _decide_confidence_only(self, policy_confidence):
        """
        Listen to the confidence monitor only. The human readiness is not
        considered, so a healthy policy releases without waiting on intent.
        """
        if policy_confidence < self.confidence_low:
            return self._result(ResponseLevel.ABORT, "policy confidence low")
        if policy_confidence < self.confidence_high:
            return self._result(ResponseLevel.SLOW, "policy confidence marginal")
        return self._result(ResponseLevel.PROCEED, "policy confident")

    def _decide_intent_only(self, intent_confidence, intent_correct):
        """
        Listen to the intent recogniser only. Policy health is ignored, so
        a failing grasp is not caught, but release waits for the human.
        """
        if intent_confidence < self.intent_ready or not intent_correct:
            return self._result(ResponseLevel.WAIT, "human not yet ready")
        return self._result(ResponseLevel.PROCEED, "human ready")

    def _decide_dual(self, policy_confidence, intent_confidence, intent_correct):
        """
        The full framework. Policy health gates first, then human
        readiness. A low policy confidence aborts regardless of the human.
        A marginal confidence slows. Only a healthy policy plus a ready
        human proceeds (Banerjee et al. 2026).
        """
        if policy_confidence < self.confidence_low:
            return self._result(ResponseLevel.ABORT, "policy confidence low")
        if policy_confidence < self.confidence_high:
            return self._result(ResponseLevel.SLOW, "policy confidence marginal")
        # Policy is healthy. Now gate on the human.
        if intent_confidence < self.intent_ready or not intent_correct:
            return self._result(ResponseLevel.WAIT, "human not yet ready")
        return self._result(ResponseLevel.PROCEED, "policy and human both ready")

    # ----- helpers ---------------------------------------------------------

    def _result(self, level, reason):
        """Bundle a decision with its intervention flag and reason."""
        return {
            "level": level,
            "intervention": level != ResponseLevel.PROCEED,
            "reason": reason,
        }


if __name__ == "__main__":
    # Manual smoke test. Run each condition against a grid of stream
    # values and confirm the responses match the intended ablation logic.
    print("Response grid per condition.")
    print("Columns are (policy_conf, intent_conf, intent_correct).\n")

    cases = [
        (0.9, 0.9, True,  "healthy policy, ready human"),
        (0.9, 0.3, True,  "healthy policy, human not ready"),
        (0.9, 0.9, False, "healthy policy, confident but wrong intent"),
        (0.35, 0.9, True, "marginal policy, ready human"),
        (0.1, 0.9, True,  "failing policy, ready human"),
        (0.1, 0.2, True,  "failing policy, human not ready"),
    ]

    for cond in CONDITIONS:
        ctrl = ResponseController(condition=cond)
        print(f"--- {cond} ---")
        for pc, ic, ok, label in cases:
            d = ctrl.decide(pc, ic, ok)
            flag = "INTERVENE" if d["intervention"] else "proceed  "
            print(
                f"  ({pc:.2f}, {ic:.2f}, {str(ok):5s})  "
                f"{d['level'].value:8s} {flag}  {label}"
            )
        print()

    # Spot checks that the ablation behaves as designed.
    print("Ablation spot checks:")
    unmon = ResponseController("unmonitored")
    assert unmon.decide(0.1, 0.1, False)["level"] == ResponseLevel.PROCEED, \
        "unmonitored must always proceed"
    print("  unmonitored proceeds even when both streams scream: ok")

    conf = ResponseController("confidence_only")
    assert conf.decide(0.1, 0.9, True)["level"] == ResponseLevel.ABORT, \
        "confidence_only must abort on low policy confidence"
    assert conf.decide(0.9, 0.1, False)["level"] == ResponseLevel.PROCEED, \
        "confidence_only ignores intent"
    print("  confidence_only aborts on bad policy, ignores intent: ok")

    intent = ResponseController("intent_only")
    assert intent.decide(0.1, 0.9, True)["level"] == ResponseLevel.PROCEED, \
        "intent_only ignores policy health"
    assert intent.decide(0.9, 0.2, True)["level"] == ResponseLevel.WAIT, \
        "intent_only waits on low intent"
    print("  intent_only waits on human, ignores policy health: ok")

    dual = ResponseController("dual_stream")
    assert dual.decide(0.1, 0.9, True)["level"] == ResponseLevel.ABORT
    assert dual.decide(0.9, 0.2, True)["level"] == ResponseLevel.WAIT
    assert dual.decide(0.9, 0.9, True)["level"] == ResponseLevel.PROCEED
    assert dual.decide(0.9, 0.9, False)["level"] == ResponseLevel.WAIT
    print("  dual_stream gates on both streams in order: ok")

    print("\nResponse controller smoke test complete.")
