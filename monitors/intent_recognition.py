"""
Human intent recognition stream for the handover framework.

This is the second stream of the dual-stream design. It watches the
human hand move and predicts where the reach will end, before the reach
finishes. Zhang et al. (2024) show that a transformer trained on partial
hand trajectories can call the target early, and Kekana et al. (2025)
survey the wider family of early intent methods. The methodology sets the
operating point at fifty percent of trajectory completion, meaning the
prediction should be reliable once half the reach has played out.

Under Path A there is no trained transformer. The predictor here is a
geometric extrapolator that fits the observed hand path and projects it
forward to the nearest candidate target. It reports a predicted target
and a confidence that grows as more of the trajectory is seen. The shape
of that confidence curve is the thing the methodology cares about, since
the fusion controller gates release on intent confidence crossing a
threshold around the halfway mark.

The candidate targets are the plausible places a hand could be reaching,
the handover zone being the one that matters. Distractor targets let the
predictor be wrong early, which is what makes the fifty percent operating
point meaningful rather than trivial.

Citations implemented here
  Zhang et al. 2024     early target prediction from partial trajectory
  Kekana et al. 2025    intent recognition method family
  Flash and Hogan 1985  minimum-jerk prior used to weight extrapolation
"""

import numpy as np


# How confident the predictor must be before the fusion controller treats
# the intent as settled. The methodology anchors reliable prediction at
# the halfway point of the reach.
INTENT_CONFIDENCE_THRESHOLD = 0.7

# Minimum number of observed hand samples before any prediction is made.
# Below this the motion direction is too noisy to extrapolate.
MIN_SAMPLES = 4


class IntentRecognizer:
    """
    Predicts a human reach target from a growing trajectory.

    The recogniser is fed hand positions one timestep at a time. After
    each update it can be queried for its current best guess and the
    confidence in that guess. Confidence rises as the hand commits to a
    direction and closes on a candidate.

    Usage
    -----
        recog = IntentRecognizer(candidate_targets)
        for hand_pos in trajectory:
            recog.update(hand_pos)
        pred = recog.predict()
        if pred["confidence"] >= INTENT_CONFIDENCE_THRESHOLD:
            ...
    """

    def __init__(self, candidate_targets, handover_index=0):
        """
        Parameters
        ----------
        candidate_targets : list of array-like, each length 3
            World positions the hand might be reaching toward. The first
            by convention is the true handover zone unless handover_index
            says otherwise.
        handover_index : int
            Index into candidate_targets of the real handover target.
            Used only for scoring accuracy, never by the predictor itself.
        """
        self.candidates = [np.array(c, dtype=float) for c in candidate_targets]
        if len(self.candidates) < 2:
            raise ValueError("need at least two candidate targets")
        self.handover_index = handover_index

        self.history = []          # observed hand positions
        self.start_pos = None      # first observed position

    def reset(self):
        """Clear the observed trajectory for a fresh reach."""
        self.history = []
        self.start_pos = None

    def update(self, hand_position):
        """
        Add one observed hand position to the trajectory.

        Parameters
        ----------
        hand_position : array-like, length 3
        """
        pos = np.array(hand_position, dtype=float)
        if self.start_pos is None:
            self.start_pos = pos.copy()
        self.history.append(pos)

    def predict(self):
        """
        Return the current best target guess and its confidence.

        The method projects the recent hand motion forward and scores each
        candidate by how well it lines up with both the direction of travel
        and the closing distance. Confidence blends three things. How much
        of the motion has been seen, how sharply the leading candidate
        beats the runner up, and how well the path points at it.

        Returns
        -------
        dict
            predicted_index, predicted_target, confidence in [0, 1],
            and progress, the fraction of the straight-line distance to
            the leading candidate already covered.
        """
        if len(self.history) < MIN_SAMPLES:
            return {
                "predicted_index": None,
                "predicted_target": None,
                "confidence": 0.0,
                "progress": 0.0,
            }

        current = self.history[-1]

        # Direction of travel from the reach origin to the current hand
        # position. Using the whole path rather than the last few samples
        # keeps the heading stable once the reach has committed, and avoids
        # the failure where a near-stationary hand at the end of a
        # minimum-jerk reach produces pure jitter for a heading
        # (Flash and Hogan 1985 give the bell-shaped speed profile that
        # makes late-reach velocity vanish).
        travel = current - self.start_pos
        travel_norm = np.linalg.norm(travel)
        if travel_norm < 1e-6:
            heading = np.zeros(3)
        else:
            heading = travel / travel_norm

        # As the hand slows near a target the heading carries less
        # information, so we lean on closeness instead. This weight shifts
        # from heading-led early to closeness-led late, which stops a
        # stationary endpoint from being misread.
        recent_step = (
            np.linalg.norm(self.history[-1] - self.history[-2])
            if len(self.history) >= 2 else 0.0
        )
        motion_trust = float(np.clip(recent_step / 0.01, 0.0, 1.0))

        # Score each candidate. A candidate scores well when the hand is
        # heading toward it and is closing on it. Proximity is the stronger
        # cue, so a candidate the hand has nearly reached dominates even if
        # a late jitter sample skews the heading.
        scores = []
        for cand in self.candidates:
            to_cand = cand - current
            dist = np.linalg.norm(to_cand)
            if dist < 1e-6:
                alignment = 1.0
            else:
                alignment = float(np.dot(heading, to_cand / dist))
            # Map alignment from [-1, 1] to [0, 1].
            align01 = (alignment + 1.0) / 2.0
            # Closeness rises sharply inside about 10 cm so that being
            # at a target is decisive. Beyond that it falls off fast.
            closeness = float(np.exp(-dist / 0.06))
            # Heading matters most mid-reach. Near a target, or when the
            # last step was tiny, lean on closeness instead.
            proximity_override = float(np.clip(1.0 - dist / 0.08, 0.0, 1.0))
            align_weight = (0.6 * motion_trust + 0.1) * (1.0 - proximity_override)
            scores.append(align01 * align_weight + closeness * (1.0 - align_weight))

        scores = np.array(scores)
        best = int(np.argmax(scores))
        best_score = scores[best]

        # Margin over the second best candidate. A clear winner means a
        # confident call. Ties keep confidence low.
        order = np.argsort(scores)[::-1]
        if len(order) > 1:
            margin = float(scores[order[0]] - scores[order[1]])
        else:
            margin = best_score

        # Progress along the straight line from start to the leading
        # candidate. This stands in for trajectory completion fraction.
        total_dist = np.linalg.norm(self.candidates[best] - self.start_pos)
        covered = np.linalg.norm(current - self.start_pos)
        progress = float(np.clip(covered / max(total_dist, 1e-6), 0.0, 1.0))

        # Confidence blends progress, margin, and alignment. The progress
        # term is why confidence is low early and reliable near halfway.
        align_term = float(np.clip(best_score, 0.0, 1.0))
        margin_term = float(np.clip(margin * 4.0, 0.0, 1.0))
        confidence = float(
            np.clip(
                0.45 * progress + 0.30 * margin_term + 0.25 * align_term,
                0.0,
                1.0,
            )
        )

        return {
            "predicted_index": best,
            "predicted_target": self.candidates[best].tolist(),
            "confidence": confidence,
            "progress": progress,
        }

    def is_correct(self, predicted_index):
        """True if the predicted index is the real handover target."""
        return predicted_index == self.handover_index


def simulate_reach_samples(start, target, num_samples, jitter=0.01, rng=None):
    """
    Produce a minimum-jerk reach from start to target as a list of hand
    positions, with small Gaussian jitter so the path is not perfectly
    clean. This mirrors what the SimulatedHumanArm produces and lets the
    recogniser be tested without standing up PyBullet.

    Parameters
    ----------
    start, target : array-like length 3
    num_samples : int
        How many positions to emit along the reach.
    jitter : float
        Standard deviation of per-sample positional noise in metres.
    rng : numpy.random.Generator, optional

    Returns
    -------
    list of numpy arrays
    """
    rng = rng if rng is not None else np.random.default_rng()
    start = np.array(start, dtype=float)
    target = np.array(target, dtype=float)
    samples = []
    for i in range(num_samples):
        tau = i / max(num_samples - 1, 1)
        profile = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        pos = start + (target - start) * profile
        pos = pos + rng.normal(0.0, jitter, size=3)
        samples.append(pos)
    return samples


if __name__ == "__main__":
    # Manual smoke test. Drive a simulated reach toward the true target,
    # feed it to the recogniser sample by sample, and print the confidence
    # curve. The key check is that confidence is low early, crosses the
    # threshold near the halfway mark, and the call is correct by then.
    rng = np.random.default_rng(7)

    handover = [0.55, 0.0, 0.90]
    distractors = [
        [0.20, 0.30, 0.90],   # off to one side
        [0.20, -0.30, 0.90],  # off to the other side
    ]
    candidates = [handover] + distractors

    start = [1.05, 0.0, 0.95]
    samples = simulate_reach_samples(start, handover, num_samples=40, rng=rng)

    recog = IntentRecognizer(candidates, handover_index=0)

    print("Confidence curve over trajectory completion:")
    print("  pct   progress  pred  correct  confidence")
    crossing = None
    for i, pos in enumerate(samples):
        recog.update(pos)
        pred = recog.predict()
        pct = (i + 1) / len(samples)
        if pred["predicted_index"] is not None:
            correct = recog.is_correct(pred["predicted_index"])
            if crossing is None and pred["confidence"] >= INTENT_CONFIDENCE_THRESHOLD:
                crossing = pct
            # Print a sampled subset to keep the output short.
            if (i + 1) % 5 == 0 or i == len(samples) - 1:
                print(
                    f"  {pct:4.0%}  {pred['progress']:7.2f}  "
                    f"{pred['predicted_index']}     "
                    f"{str(correct):5s}    {pred['confidence']:.2f}"
                )

    final = recog.predict()
    print(f"\nFinal prediction index: {final['predicted_index']} "
          f"(true handover is {recog.handover_index})")
    print(f"Final confidence: {final['confidence']:.2f}")
    if crossing is not None:
        print(f"Confidence crossed {INTENT_CONFIDENCE_THRESHOLD} at "
              f"{crossing:.0%} of the trajectory")
    else:
        print("Confidence never crossed the threshold")

    # A second run reaching for a distractor, to confirm the recogniser
    # does not blindly favour the handover zone.
    print("\nControl run, hand reaches for a distractor instead:")
    recog2 = IntentRecognizer(candidates, handover_index=0)
    samples2 = simulate_reach_samples(start, distractors[0], num_samples=40, rng=rng)
    for pos in samples2:
        recog2.update(pos)
    p2 = recog2.predict()
    print(f"  predicted index {p2['predicted_index']} "
          f"(distractor is index 1), confidence {p2['confidence']:.2f}")

    # The methodology metric is accuracy at the halfway operating point.
    # Run many reaches toward the handover zone and report how often the
    # call is right once half the trajectory is seen.
    print("\nAccuracy at the 50% operating point over 200 reaches:")
    correct_at_half = 0
    n = 200
    for k in range(n):
        r = IntentRecognizer(candidates, handover_index=0)
        rk = np.random.default_rng(1000 + k)
        ss = simulate_reach_samples(start, handover, num_samples=40, rng=rk)
        half = len(ss) // 2
        for pos in ss[:half]:
            r.update(pos)
        pr = r.predict()
        if pr["predicted_index"] is not None and r.is_correct(pr["predicted_index"]):
            correct_at_half += 1
    print(f"  {correct_at_half}/{n} correct = {correct_at_half / n:.1%}")

    print("\nIntent recognition smoke test complete.")
