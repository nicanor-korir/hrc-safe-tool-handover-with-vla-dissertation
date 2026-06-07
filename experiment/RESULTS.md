# Experimental Results

Input data for Chapter Four. Generated from the full simulation run on
the dual-stream handover framework. This file states what was run and
what the data shows. It does not draft the chapter.

## Provenance

- Results CSV: `data/handover_results_full.csv` (1000 trials)
- Figures: `data/figures/`
- Base seed: 2026, deterministic per-trial seeding, so the run reproduces
- Run time: about 290 seconds, roughly 0.29 seconds per trial
- Approach: Path A, scripted handover policy standing in for OpenVLA, with
  a constraint-based grasp model and the four-condition ablation preserved

## Experimental matrix

Four conditions crossed with four failure scenario groups, fifty trials
per cell, giving 800 failure-scenario trials. A nominal control group of
fifty trials per condition adds 200 more, for 1000 trials total. Every
cell ran the full fifty trials with no dropouts.

| condition       | camera_shift | lighting | novel_geometry | approach_traj | nominal |
| --------------- | ------------ | -------- | -------------- | ------------- | ------- |
| unmonitored     | 50           | 50       | 50             | 50            | 50      |
| confidence_only | 50           | 50       | 50             | 50            | 50      |
| intent_only     | 50           | 50       | 50             | 50            | 50      |
| dual_stream     | 50           | 50       | 50             | 50            | 50      |

The three tools (screwdriver, rubber mallet, pencil) were cycled across
trials within every cell, so each cell covers all three. Human body
height was sampled uniformly between 1.55 and 1.90 metres per trial.

## Headline numbers per condition

### Failure scenarios (n = 200 per condition)

| condition       | safety intervention | unsafe completion | trial completed |
| --------------- | ------------------- | ----------------- | --------------- |
| unmonitored     | 0.0%                | 24.5%             | 100.0%          |
| confidence_only | 76.5%               | 0.0%              | 23.5%           |
| intent_only     | 4.5%                | 28.0%             | 95.5%           |
| dual_stream     | 81.5%               | 0.0%              | 18.5%           |

Safety intervention means the controller prevented a completion that
would otherwise have gone ahead. Unsafe completion means a trial finished
with a handover while the grasp had actually failed, the event the
framework exists to stop.

### Nominal control (n = 50 per condition)

| condition       | completion | false positive |
| --------------- | ---------- | -------------- |
| unmonitored     | 100.0%     | 0.0%           |
| confidence_only | 96.0%      | 4.0%           |
| intent_only     | 100.0%     | 0.0%           |
| dual_stream     | 96.0%      | 4.0%           |

A false positive is a nominal trial that a healthy grasp should have
completed but the controller blocked.

### Fluency metrics, released trials only (Ortenzi et al. 2021)

| condition       | idle time (s) | functional delay (s) | handover duration (s) | n released |
| --------------- | ------------- | -------------------- | --------------------- | ---------- |
| unmonitored     | 0.000         | 0.004                | 0.004                 | 250        |
| confidence_only | 0.000         | 0.004                | 0.004                 | 95         |
| intent_only     | 0.221         | 0.916                | 0.916                 | 241        |
| dual_stream     | 0.208         | 0.821                | 0.821                 | 85         |

The two conditions that do not consult the intent stream release as soon
as the robot presents, so their idle time and delay are near zero. The
two conditions that do consult intent wait for the human reach to reach
its readiness point, which is where their delay comes from. The numbers
across these two groups are therefore not like for like, and the chapter
should frame the delay as the cost intent gating pays for waiting on the
human rather than as a defect.

## Pre-registered hypothesis verdict

The dissertation registered that the dual-stream framework wins only if
all three conditions hold. On the full data, all three hold.

**Condition 1, safety intervention higher in failure groups. HOLDS.**
Dual-stream intervened on 81.5% of failure trials against 0.0% for the
unmonitored baseline. Chi-square on the two conditions gives p < 0.0001
with Cramer's V of 0.78, a large effect.

**Condition 2, nominal false positives within ten points of baseline.
HOLDS.** Dual-stream false positive rate was 4.0% against 0.0% for
unmonitored, a gap of 4.0 percentage points, inside the ten point
tolerance.

**Condition 3, nominal completion within ten points of baseline. HOLDS.**
Dual-stream completed 96.0% of nominal trials against 100.0% for
unmonitored, a gap of 4.0 percentage points, inside the ten point
tolerance.

**Overall, the dual-stream framework wins on the pre-registered
criteria.**

## What the data shows

The unmonitored baseline completed every handover, including 24.5% of
failure trials where the grasp had actually failed. Those are the
dangerous completions the framework targets. Both confidence-using
conditions drove unsafe completions to zero. The intent-only condition
did not, sitting at 28.0%, close to the unmonitored baseline, because the
intent stream watches the human and cannot see a failing grasp.

The most informative breakdown is safety intervention split by scenario.

| condition       | camera_shift | lighting | novel_geometry | approach_traj |
| --------------- | ------------ | -------- | -------------- | ------------- |
| unmonitored     | 0%           | 0%       | 0%             | 0%            |
| confidence_only | 98%          | 100%     | 100%           | 8%            |
| intent_only     | 0%           | 0%       | 0%             | 18%           |
| dual_stream     | 98%          | 100%     | 100%           | 28%           |

The confidence stream is strong on the three failures that show up in the
robot's own state, namely camera shift, lighting, and novel geometry. It
is weak on the unexpected human approach, catching only 8% of those,
because that failure lives on the human side. The intent stream is the
mirror image, weak on the robot-side failures and the better of the two
on the human approach. The dual-stream condition inherits the confidence
stream's strength on robot-side failures and adds the intent stream's
contribution on the human approach, lifting approach_traj interventions
from 8% to 28%. This is the concrete evidence that the second stream adds
something the first cannot supply.

The unsafe completion breakdown by scenario reinforces the same point.
Intent-only and unmonitored both let through camera shift, lighting, and
novel geometry failures at rates of 26% to 54%, while both
confidence-using conditions caught all of them.

## Statistical summary

- Two-way ANOVA, all four continuous metrics, condition main effect
  p < 0.0001 with eta squared between 0.70 and 0.94. Scenario main effect
  significant but small. Interaction significant for the timing metrics.
- Chi-square across conditions, safety intervention p < 0.0001, Cramer's
  V 0.78. Unsafe release p < 0.0001, Cramer's V 0.39. Both large to
  moderate effects.
- Pairwise versus unmonitored on safety intervention with Bonferroni
  correction, confidence_only and dual_stream both p < 0.0001,
  intent_only p = 0.007.
- Tukey HSD on handover duration, the intent-using conditions differ from
  the others at p < 0.0001, and dual_stream differs from intent_only at
  p < 0.0001.

## Caveats and honest notes for the write-up

1. [ ] **Path A grasp model.** The grasp is a fixed kinematic attachment
    rather than a friction pinch. The Franka mounted level with the table
    could not reliably pinch the thin primitive tools at the floor of its
    reach, and the study measures monitoring and fusion rather than grasp
    mechanics, so the attachment keeps the nominal pick deterministic. Any
    failure in the data comes from the injected scenario, not from grasp
    noise. Full VLA integration and a learned grasp remain future work.
2. [ ] **Confidence signal is scripted, not a VLA hidden layer.** The
    confidence monitor reads the policy's own internal state, the Path A
    analogue of the activations a SAFE monitor would use. The conformal
    calibration that sets the alarm threshold is real and is calibrated on
    a held-out nominal pool, holding the nominal false positive rate near
    its ten percent target by construction.
3. [ ] **Intent accuracy at the half point is logged per trial but reads low
    for conditions that release early.** Conditions that proceed before the
    human reach reaches its midpoint record the early, low-confidence
    prediction. This is a logging timing artifact, not a property of the
    recogniser. The recogniser's own accuracy at the fifty percent
    operating point is 99.5% over 200 independent reaches, shown in
    `intent_accuracy_vs_completion.png`. Cite the figure for the
    recogniser's accuracy, not the per-condition column.
4. [ ] **Fluency timings are not like for like across condition groups.** See
    the fluency table note above. Compare within the intent-using pair and
    within the non-intent pair, not across the two groups directly.
5. [ ] **approach_traj is the hardest scenario for every condition.** Even
    dual-stream catches only 28% of unexpected human approaches. The
    intent stream as built is a geometric extrapolator, and a trained
    transformer of the kind Zhang et al. 2024 describe would likely lift
    this. This is a clear and honest limitation to name.

## Figures

- `data/figures/safety_intervention_by_condition.png`
  Safety intervention rate across the four conditions, failure scenarios.
- `data/figures/intent_accuracy_vs_completion.png`
  Intent accuracy and mean confidence over trajectory completion, with
  the fifty percent operating point marked.
- `data/figures/fluency_metrics_by_condition.png`
  The three Ortenzi fluency metrics across conditions.
- `data/figures/outcome_confusion_by_condition.png`
  Outcome breakdown per condition, safe completion, correct catch, missed
  catch, and false alarm.
