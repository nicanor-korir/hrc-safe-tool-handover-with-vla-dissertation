# Safe Tool Handover in Human-Robot Collaboration

A dual-stream framework combining VLA failure detection and human intent
recognition, with a PyBullet simulation study and a four-condition
ablation.

This repository holds the simulation experiments for an MSc dissertation,
*Safe Tool Handover in Human-Robot Collaboration: A Dual-Stream Framework
Combining VLA Failure Detection and Human Intent Recognition*, by Nicanor
Korir (BSBI / UCA), supervised by Professor Vincent English.

---

## Overview

A robot that hands tools to a person has to decide when it is safe to let
go. Two kinds of thing can go wrong. The robot's own grasp or perception
can fail, and the human can reach in a way the robot did not expect. A
single safety signal tends to cover one of these well and the other
poorly. This study tests whether fusing two complementary streams, one
watching the robot and one watching the human, gives safer handovers than
either stream alone or no monitoring at all.

The framework has two streams.

- A **confidence monitor** in the style of the SAFE failure detector,
  which reads the policy's internal state and raises an alarm when the
  handover looks like it is failing. Its alarm threshold is set by
  conformal calibration on nominal trials, so it needs no labelled
  failure data.
- An **intent recogniser** that watches the human hand and predicts the
  reach target early, with reliable prediction by the halfway point of
  the reach.

A rule-based **fusion controller** combines the two into a four-level
graduated response, proceed, wait, slow, or abort. The four experimental
conditions are realised by which streams the controller is allowed to
consult.

The headline finding from the full run of 1000 trials is that the
dual-stream framework satisfies all three pre-registered success
criteria. It intervenes on 81.5% of failure trials against 0% for an
unmonitored baseline, drives unsafe completions to zero, and stays within
four percentage points of the baseline on both nominal completion and
nominal false positives. Full numbers are in
[`experiment/RESULTS.md`](experiment/RESULTS.md).

---

## A note on scope, Path A

The methodology commits to OpenVLA as the policy backbone. Full
OpenVLA-7B integration was not feasible in the time available, so this
implementation follows what the project calls **Path A**. A scripted
handover policy stands in for the learned VLA, and the grasp is modelled
as a kinematic attachment rather than a friction pinch. The four-condition
ablation logic and the pre-registered hypothesis are preserved exactly as
the methodology specifies. The substitution is sound for this study
because the monitoring and fusion streams operate on policy state and
trajectory data, not on raw model activations or visual fidelity.

The confidence monitor therefore reads the scripted policy's internal
state, which is the Path A analogue of the hidden-layer features a real
SAFE monitor would consume. Every figure and result is honest about this.
Full VLA integration and a learned intent transformer are named as
immediate future work. The detailed caveats are listed in
[`experiment/RESULTS.md`](experiment/RESULTS.md).

---

## Repository layout

```
.
├── environment/
│   ├── workspace.py            Table, Franka Panda, three tools, target marker
│   └── human_arm.py            7-DOF kinematic arm, minimum-jerk reach
├── policy/
│   ├── robot_controller.py     IK pose control and gripper primitives
│   └── handover_policy.py      Seven-state machine, four failure modes
├── monitors/
│   ├── confidence_monitor.py   SAFE-style monitor, conformal calibration
│   └── intent_recognition.py   Early intent predictor from hand trajectory
├── fusion/
│   └── response_controller.py  Rule-based four-level graduated response
├── experiment/
│   ├── runner.py               Sweeps the matrix, logs per-trial CSV
│   └── RESULTS.md              Findings input for Chapter Four
├── analysis/
│   ├── statistics.py           ANOVA, chi-square, Tukey, hypothesis verdict
│   └── figures.py              The four Chapter Four figures
└── data/
    ├── handover_results_full.csv    1000-trial results
    ├── handover_results_pilot.csv   Pilot results
    └── figures/                     Generated PNGs
```

---

## Method

### Experimental design

A two-factor design crosses four monitoring conditions with four failure
scenario groups, at fifty trials per cell. That gives 800 failure-scenario
trials. A nominal control group of fifty trials per condition adds 200
more, for 1000 trials in total. Fifty trials per cell follows the power
guidance for HRI studies in Bartlett et al. (2022).

**Conditions (the ablation)**

| condition         | confidence stream | intent stream |
| ----------------- | ----------------- | ------------- |
| `unmonitored`     | off               | off           |
| `confidence_only` | on                | off           |
| `intent_only`     | off               | on            |
| `dual_stream`     | on                | on            |

**Failure scenario groups**

| group            | what is perturbed                                         |
| ---------------- | --------------------------------------------------------- |
| `camera_shift`   | perceived tool pose offset, grasp aimed at the wrong spot |
| `lighting`       | perceptual noise up, grasp confidence degraded            |
| `novel_geometry` | unfamiliar tool shape, poorly seated grasp                |
| `approach_traj`  | human reaches early and off-axis, release timing stressed |

A `nominal` control group runs the same handover with no perturbation,
which is what the false positive rate is measured against.

**Tools**, all presented handle-first per Ortenzi et al. (2021)

- Screwdriver, 15 cm, sharp, an orientation control test
- Rubber mallet, 1.2 kg, a grip and release timing test
- Pencil, 17 cm by 7 mm, a precision test

### Metrics

Continuous metrics use the handover fluency taxonomy of Ortenzi et al.
(2021).

- Idle time, how long the human waits with a hand presented
- Functional delay, the gap from the robot being ready to the release
- Handover duration, the full reach-to-release span
- Intent prediction accuracy at the halfway operating point

Binary outcomes.

- Safety intervention, the controller prevented a completion
- Trial completed, the tool was released to the human
- Unsafe completion, a handover finished while the grasp had failed
- False positive, a nominal trial wrongly blocked

### Statistics

- Two-way ANOVA on the continuous metrics, with eta squared effect sizes
- Chi-square on binary outcomes, with Fisher's exact for sparse two by two
  tables and Cramer's V for effect size
- Tukey HSD on continuous metrics and Bonferroni-corrected pairwise
  proportion tests on binary outcomes

### Pre-registered hypothesis

The dual-stream framework is registered to win only if all three hold.

1. Safety intervention rate in failure groups is statistically higher than
   the unmonitored baseline.
2. Nominal false positive rate is no more than ten percentage points above
   the unmonitored baseline.
3. Nominal completion rate is within ten percentage points of the
   unmonitored baseline.

The analysis reports each condition plainly and the experiment is
falsifiable. On the full run all three hold.

---

## Installation

The project uses PyBullet for physics and the standard scientific Python
stack for analysis.

### With conda

```bash
conda create -n handover python=3.11
conda activate handover
conda install -c conda-forge pybullet numpy scipy pandas matplotlib
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiment runs headless, so no display is required. The smoke tests
in `environment/` open a GUI window if run directly, which is optional.

---

## Reproducing the study

All commands run from the repository root. The package imports are
relative, so set `PYTHONPATH` to the root or run the modules with `-m`.

### 1. Check the build with smoke tests

Each module has a self-contained test in its `__main__` block.

```bash
PYTHONPATH=. python -m policy.handover_policy
PYTHONPATH=. python -m monitors.confidence_monitor
PYTHONPATH=. python -m monitors.intent_recognition
PYTHONPATH=. python -m fusion.response_controller
```

### 2. Run a pilot first

A small pilot checks the protocol before committing to the full run.

```bash
PYTHONPATH=. python -m experiment.runner --pilot
```

This writes `data/handover_results_pilot.csv`, five trials per cell.

### 3. Run the full experiment

```bash
PYTHONPATH=. python -m experiment.runner --trials 50 --seed 2026
```

This writes `data/handover_results_full.csv`, 1000 trials, in about five
minutes on a laptop. The base seed makes the run reproducible.

### 4. Analyse

```bash
PYTHONPATH=. python -m analysis.statistics data/handover_results_full.csv
PYTHONPATH=. python -m analysis.figures data/handover_results_full.csv
```

The statistics command prints the full ANOVA, the binary tests, the
post-hoc comparisons, and the hypothesis verdict. The figures command
writes four PNGs to `data/figures/`.

---

## Results at a glance

From the full 1000-trial run, seed 2026.

| condition         | safety intervention (failures) | unsafe completion (failures) | nominal completion | nominal false positive |
| ----------------- | ------------------------------ | ---------------------------- | ------------------ | ---------------------- |
| `unmonitored`     | 0.0%                           | 24.5%                        | 100.0%             | 0.0%                   |
| `confidence_only` | 76.5%                          | 0.0%                         | 96.0%              | 4.0%                   |
| `intent_only`     | 4.5%                           | 28.0%                        | 100.0%             | 0.0%                   |
| `dual_stream`     | 81.5%                          | 0.0%                         | 96.0%              | 4.0%                   |

The clearest evidence for the second stream is the per-scenario
breakdown. The confidence stream catches robot-side failures at 98 to
100% but only 8% of unexpected human approaches. Adding the intent stream
lifts approach interventions to 28%, a failure the confidence stream
cannot see. Full tables and the statistical detail are in
[`experiment/RESULTS.md`](experiment/RESULTS.md).

### Figures

| file                                       | shows                                                     |
| ------------------------------------------ | --------------------------------------------------------- |
| `safety_intervention_by_condition.png`     | safety intervention rate across conditions                |
| `intent_accuracy_vs_completion.png`        | intent accuracy over reach completion, 50% point marked   |
| `fluency_metrics_by_condition.png`         | the three Ortenzi fluency metrics across conditions       |
| `outcome_confusion_by_condition.png`       | safe, correct catch, missed catch, and false alarm shares |

---

## Reference list

The citations implemented in the code, named in module docstrings.

- Klopcar and Lenarcic 2005, human arm kinematic model
- Flash and Hogan 1985, minimum-jerk reach trajectory
- Gu et al. 2025, SAFE framework for VLA failure detection
- Xu et al. 2025, FAIL-Detect, conformal calibration without failure data
- Banerjee et al. 2026, human-in-the-loop confidence-aware framework
- Kim et al. 2025, OpenVLA pre-trained model
- Angelopoulos and Bates 2023, conformal prediction calibration
- Bartlett et al. 2022, statistical power in HRI
- Ortenzi et al. 2021, handover taxonomy and fluency metrics
- Zhang et al. 2024, transformer-based early intent prediction
- Kekana et al. 2025, review of intent recognition methods

---

## Author

Nicanor Korir, MSc, BSBI / UCA. Supervisor Professor Vincent English.
