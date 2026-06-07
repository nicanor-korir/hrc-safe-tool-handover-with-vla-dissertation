"""
Figure generation for Chapter Four of the dissertation.

This module reads the trial CSV and the intent recogniser to produce the
charts the findings chapter needs. Every figure is saved as a PNG in
data/figures with a descriptive name. Nothing here recomputes statistics,
the numbers come straight from the logged trials and from a fresh sweep of
the recogniser for the accuracy curve.

Figures produced

  safety_intervention_by_condition.png
      bar chart of safety intervention rate across the four conditions in
      failure scenarios, the headline safety result

  safety_intervention_by_scenario.png
      the same rate split by failure scenario, one bar per condition. This
      carries the complementarity result, the confidence-using conditions
      tall on robot-side failures and only the dual stream lifting the
      human-side approach column appreciably

  intent_accuracy_vs_completion.png
      intent prediction accuracy as a function of how much of the reach
      has been seen, marking the fifty percent operating point

  fluency_metrics_by_condition.png
      grouped bars for the three Ortenzi fluency metrics across conditions

  outcome_confusion_by_condition.png
      a confusion-style breakdown of safe and unsafe outcomes per
      condition, the false positive and false negative picture

Citations implemented here
  Ortenzi et al. 2021   the three fluency metrics charted
  Zhang et al. 2024     the intent accuracy curve and its operating point
  Gu et al. 2025        the safety intervention framing
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt


CONDITIONS = ["unmonitored", "confidence_only", "intent_only", "dual_stream"]
CONDITION_LABELS = {
    "unmonitored": "Unmonitored",
    "confidence_only": "Confidence\nonly",
    "intent_only": "Intent\nonly",
    "dual_stream": "Dual\nstream",
}
NOMINAL = "nominal"

# A colourblind-friendly palette, one colour per condition.
PALETTE = {
    "unmonitored": "#999999",
    "confidence_only": "#0072B2",
    "intent_only": "#E69F00",
    "dual_stream": "#009E73",
}


def _ensure_dir(figures_dir):
    os.makedirs(figures_dir, exist_ok=True)


def figure_safety_intervention(df, figures_dir):
    """Bar chart of safety intervention rate by condition, failures only."""
    fail = df[df["scenario"] != NOMINAL]
    rates = [
        fail[fail["condition"] == c]["safety_intervention"].mean()
        for c in CONDITIONS
    ]
    counts = [len(fail[fail["condition"] == c]) for c in CONDITIONS]
    # Wald standard error on each proportion for the error bars.
    errs = [
        np.sqrt(max(r * (1 - r), 0) / n) if n else 0
        for r, n in zip(rates, counts)
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(CONDITIONS))
    bars = ax.bar(
        x, rates, yerr=errs, capsize=4,
        color=[PALETTE[c] for c in CONDITIONS], edgecolor="black", linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Safety intervention rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Safety intervention rate by condition, failure scenarios")
    for bar, r in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{r:.0%}", ha="center", va="bottom", fontsize=10,
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "safety_intervention_by_condition.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# Failure scenarios in the order that reads best, the three robot-side
# groups first and the human-side approach last so the eye lands on the
# column where the streams diverge.
SCENARIO_ORDER = ["camera_shift", "lighting", "novel_geometry", "approach_traj"]
SCENARIO_LABELS = {
    "camera_shift": "Camera\nshift",
    "lighting": "Lighting",
    "novel_geometry": "Novel\ngeometry",
    "approach_traj": "Approach\ntrajectory",
}


def figure_safety_intervention_by_scenario(df, figures_dir):
    """
    Grouped bar chart of safety intervention rate split by failure
    scenario, one bar per condition within each scenario. This is the
    figure that carries the complementarity argument. The confidence-using
    conditions stand tall on the three robot-side scenarios and drop on the
    human-side approach, while the dual-stream bar is the only one that
    lifts the approach column appreciably above the single streams.
    """
    fail = df[df["scenario"] != NOMINAL]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.2

    for i, cond in enumerate(CONDITIONS):
        rates = []
        errs = []
        for scen in SCENARIO_ORDER:
            cell = fail[(fail["condition"] == cond) & (fail["scenario"] == scen)]
            r = cell["safety_intervention"].mean() if len(cell) else 0.0
            n = len(cell)
            rates.append(r)
            errs.append(np.sqrt(max(r * (1 - r), 0) / n) if n else 0.0)
        offset = (i - 1.5) * width
        bars = ax.bar(
            x + offset, rates, width, yerr=errs, capsize=3,
            color=PALETTE[cond], edgecolor="black", linewidth=0.5,
            label=CONDITION_LABELS[cond].replace("\n", " "),
        )
        # Label only the dual-stream and confidence bars on the approach
        # column, where the divergence the figure exists to show lives.
        for bar, r, scen in zip(bars, rates, SCENARIO_ORDER):
            if scen == "approach_traj" and cond in ("confidence_only", "dual_stream"):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{r:.0%}", ha="center", va="bottom", fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.set_ylabel("Safety intervention rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Safety intervention rate by failure scenario and condition")
    # Legend below the axes so it never overlaps the bars or annotations.
    ax.legend(
        title="Condition", ncol=4, fontsize=9,
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
    )
    # A light divider separating the robot-side scenarios from the human
    # side one, so the structural split is visible at a glance.
    ax.axvline(2.5, color="grey", linestyle=":", alpha=0.6)
    ax.text(1.0, 1.09, "robot-side failures", ha="center", fontsize=9, color="grey")
    ax.text(3.0, 1.09, "human-side", ha="center", fontsize=9, color="grey")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(
        figures_dir, "safety_intervention_by_scenario.png"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_intent_accuracy_curve(figures_dir, num_reaches=300, seed=42):
    """
    Intent accuracy as a function of trajectory completion. Runs the
    recogniser over many simulated reaches and records, at each completion
    fraction, whether the leading prediction is the true target.
    """
    from monitors.intent_recognition import (
        IntentRecognizer, simulate_reach_samples,
    )

    handover = [0.55, 0.0, 0.90]
    distractors = [[0.20, 0.35, 0.90], [0.20, -0.35, 0.90]]
    candidates = [handover] + distractors
    start = [1.05, 0.0, 0.95]

    fractions = np.linspace(0.1, 1.0, 19)
    correct_counts = np.zeros_like(fractions)
    confidence_means = np.zeros_like(fractions)

    rng = np.random.default_rng(seed)
    n_samples = 40
    for k in range(num_reaches):
        rk = np.random.default_rng(int(rng.integers(0, 1_000_000)))
        samples = simulate_reach_samples(start, handover, num_samples=n_samples, rng=rk)
        recog = IntentRecognizer(candidates, handover_index=0)
        for fi, frac in enumerate(fractions):
            recog.reset()
            cut = max(int(frac * n_samples), 1)
            for pos in samples[:cut]:
                recog.update(pos)
            pred = recog.predict()
            if pred["predicted_index"] is not None:
                if recog.is_correct(pred["predicted_index"]):
                    correct_counts[fi] += 1
                confidence_means[fi] += pred["confidence"]

    accuracy = correct_counts / num_reaches
    confidence_means = confidence_means / num_reaches

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        fractions * 100, accuracy, marker="o", color="#0072B2",
        label="Prediction accuracy",
    )
    ax.plot(
        fractions * 100, confidence_means, marker="s", color="#E69F00",
        linestyle="--", label="Mean confidence",
    )
    ax.axvline(50, color="black", linestyle=":", alpha=0.7)
    ax.text(51, 0.05, "50% operating point", rotation=90, va="bottom", fontsize=9)
    ax.set_xlabel("Trajectory completion (%)")
    ax.set_ylabel("Accuracy / confidence")
    ax.set_ylim(0, 1.05)
    ax.set_title("Intent prediction accuracy over reach completion")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "intent_accuracy_vs_completion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_fluency_metrics(df, figures_dir):
    """Grouped bars of the three Ortenzi fluency metrics across conditions."""
    metrics = ["idle_time_s", "functional_delay_s", "handover_duration_s"]
    metric_labels = ["Idle time", "Functional delay", "Handover duration"]

    means = {m: [] for m in metrics}
    errs = {m: [] for m in metrics}
    for c in CONDITIONS:
        cell = df[df["condition"] == c]
        for m in metrics:
            vals = pd.to_numeric(cell[m], errors="coerce").dropna()
            means[m].append(vals.mean() if len(vals) else 0.0)
            errs[m].append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CONDITIONS))
    width = 0.25
    colours = ["#56B4E9", "#D55E00", "#009E73"]
    for i, (m, lbl) in enumerate(zip(metrics, metric_labels)):
        ax.bar(
            x + (i - 1) * width, means[m], width, yerr=errs[m], capsize=3,
            label=lbl, color=colours[i], edgecolor="black", linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Seconds")
    ax.set_title("Handover fluency metrics by condition (Ortenzi et al. 2021)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, "fluency_metrics_by_condition.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_outcome_confusion(df, figures_dir):
    """
    A confusion-style outcome breakdown per condition. For each condition
    we count four mutually exclusive outcomes, treating an unsafe
    completion as a missed catch and a nominal block as a false alarm.

      safe completion    nominal trial that completed, the good case
      correct catch      failure trial that did not complete unsafely
      missed catch       failure trial that completed while grasp failed
      false alarm        nominal trial blocked despite a healthy grasp
    """
    categories = ["safe completion", "correct catch", "missed catch", "false alarm"]
    matrix = np.zeros((len(CONDITIONS), len(categories)))

    for ci, c in enumerate(CONDITIONS):
        cell = df[df["condition"] == c]
        nominal = cell[cell["scenario"] == NOMINAL]
        fail = cell[cell["scenario"] != NOMINAL]

        safe_completion = (
            (nominal["trial_completed"] == 1).sum()
        )
        false_alarm = int(nominal["false_positive"].sum())
        missed_catch = int(fail["unsafe_release"].sum())
        correct_catch = int(len(fail) - missed_catch)

        total = max(len(cell), 1)
        matrix[ci] = [
            safe_completion / total,
            correct_catch / total,
            missed_catch / total,
            false_alarm / total,
        ]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=matrix.max() or 1)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(CONDITIONS)))
    ax.set_yticklabels([c.replace("_", " ") for c in CONDITIONS])
    for i in range(len(CONDITIONS)):
        for j in range(len(categories)):
            ax.text(
                j, i, f"{matrix[i, j]:.0%}", ha="center", va="center",
                color="black" if matrix[i, j] < (matrix.max() or 1) * 0.6 else "white",
                fontsize=10,
            )
    ax.set_title("Outcome breakdown by condition (share of all trials)")
    fig.colorbar(im, ax=ax, label="Share of trials")
    fig.tight_layout()
    path = os.path.join(figures_dir, "outcome_confusion_by_condition.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all(csv_path, figures_dir="data/figures"):
    """Produce every figure and return the list of written paths."""
    _ensure_dir(figures_dir)
    df = pd.read_csv(csv_path)
    paths = []
    paths.append(figure_safety_intervention(df, figures_dir))
    paths.append(figure_safety_intervention_by_scenario(df, figures_dir))
    paths.append(figure_intent_accuracy_curve(figures_dir))
    paths.append(figure_fluency_metrics(df, figures_dir))
    paths.append(figure_outcome_confusion(df, figures_dir))
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Chapter Four figures.")
    parser.add_argument(
        "csv", nargs="?", default="data/handover_results_pilot.csv",
        help="Path to the results CSV.",
    )
    parser.add_argument(
        "--figures-dir", default="data/figures",
        help="Where to write the PNG files.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(
            f"No results file at {args.csv}. Run experiment.runner first."
        )

    written = generate_all(args.csv, args.figures_dir)
    print("Wrote figures:")
    for path in written:
        print(f"  {path}")
