"""
Statistical analysis for the dual-stream handover experiment.

This module reads the trial CSV the runner produces and applies the tests
the methodology fixes. Continuous fluency metrics get a two-way ANOVA
across condition and scenario, with eta squared as the effect size.
Binary outcomes get a chi-square test, falling back to Fisher's exact
when an expected cell count is small (Bartlett et al. 2022 warn that HRI
cells are often sparse). Pairwise condition differences get Tukey HSD
with a Bonferroni adjustment on the family of comparisons.

The module then evaluates the pre-registered hypothesis directly. The
dissertation registered that the dual-stream framework wins only if three
conditions all hold.

  1. its safety intervention rate in failure groups is statistically
     higher than the unmonitored baseline
  2. its nominal false positive rate is no more than ten percentage
     points above the unmonitored baseline
  3. its nominal completion rate is within ten percentage points of the
     unmonitored baseline

The verdict is reported plainly, stating which conditions hold and which
do not from the data as it stands. The framework is meant to be
falsifiable, so a failed condition is reported, not smoothed over.

Citations implemented here
  Bartlett et al. 2022          power and sparse cells in HRI
  Ortenzi et al. 2021           the fluency metrics under test
  Angelopoulos and Bates 2023   the calibrated false positive target
"""

import numpy as np
import pandas as pd
from scipy import stats


# Continuous metrics analysed with ANOVA.
CONTINUOUS_METRICS = [
    "idle_time_s",
    "functional_delay_s",
    "handover_duration_s",
    "intent_confidence_at_half",
]

# Binary outcomes analysed with chi-square or Fisher's exact.
BINARY_OUTCOMES = [
    "safety_intervention",
    "trial_completed",
    "unsafe_release",
    "intent_correct_at_half",
]

CONDITIONS = ["unmonitored", "confidence_only", "intent_only", "dual_stream"]
NOMINAL = "nominal"

# The ten percentage point tolerance the hypothesis registers.
PP_TOLERANCE = 0.10


def load_results(csv_path):
    """Read the trial CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def two_way_anova(df, metric):
    """
    Two-way ANOVA of a continuous metric across condition and scenario.

    Uses a sum-of-squares decomposition computed directly rather than
    through a formula API, so the dependency surface stays small. Returns
    the F statistic, p value, and eta squared for each main effect and the
    interaction. Rows where the metric is missing are dropped, which is
    correct since an aborted trial has no release-based timing.

    Parameters
    ----------
    df : DataFrame
    metric : str
        Column name of the continuous metric.

    Returns
    -------
    dict
        Per-effect F, p, and eta squared, plus the sample size used.
    """
    sub = df[["condition", "scenario", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=[metric])

    if len(sub) < 8 or sub[metric].nunique() < 2:
        return {"note": "insufficient data for ANOVA", "n": int(len(sub))}

    grand_mean = sub[metric].mean()
    ss_total = ((sub[metric] - grand_mean) ** 2).sum()

    def factor_ss(factor):
        ss = 0.0
        for _, group in sub.groupby(factor):
            ss += len(group) * (group[metric].mean() - grand_mean) ** 2
        return ss

    ss_cond = factor_ss("condition")
    ss_scen = factor_ss("scenario")

    # Interaction sum of squares from the cell means. With an unbalanced
    # design, dropped aborted trials leave unequal cell counts, and the
    # factors stop being orthogonal. The sequential decomposition can then
    # push the interaction term slightly negative, which is not meaningful.
    # We flag that case and clamp the term to zero rather than report a
    # negative sum of squares.
    ss_cells = 0.0
    cell_sizes = []
    for _, cell in sub.groupby(["condition", "scenario"]):
        ss_cells += len(cell) * (cell[metric].mean() - grand_mean) ** 2
        cell_sizes.append(len(cell))
    balanced = len(set(cell_sizes)) == 1
    ss_inter = ss_cells - ss_cond - ss_scen
    interaction_unreliable = ss_inter < 0
    ss_inter = max(ss_inter, 0.0)
    ss_error = ss_total - ss_cells

    n_cond = sub["condition"].nunique()
    n_scen = sub["scenario"].nunique()
    n = len(sub)

    df_cond = n_cond - 1
    df_scen = n_scen - 1
    df_inter = df_cond * df_scen
    df_error = n - n_cond * n_scen

    if df_error <= 0 or ss_error <= 0:
        return {"note": "degenerate ANOVA design", "n": int(n)}

    ms_error = ss_error / df_error

    def effect(ss, df_effect):
        ms = ss / df_effect if df_effect > 0 else 0.0
        f = ms / ms_error if ms_error > 0 else 0.0
        p = float(stats.f.sf(f, df_effect, df_error)) if f > 0 else 1.0
        eta_sq = ss / ss_total if ss_total > 0 else 0.0
        return {"F": float(f), "p": p, "eta_squared": float(eta_sq), "df": df_effect}

    return {
        "n": int(n),
        "balanced": bool(balanced),
        "interaction_unreliable": bool(interaction_unreliable),
        "condition": effect(ss_cond, df_cond),
        "scenario": effect(ss_scen, df_scen),
        "interaction": effect(ss_inter, df_inter),
        "df_error": int(df_error),
    }


def binary_test(df, outcome, group_col="condition", subset=None):
    """
    Test whether a binary outcome differs across groups.

    Builds a contingency table of the outcome by the grouping column, then
    runs a chi-square test. If any expected cell count falls below five and
    the table is two by two, it switches to Fisher's exact, as the
    methodology directs. Reports Cramer's V as the effect size.

    Parameters
    ----------
    df : DataFrame
    outcome : str
        Binary column, values 0 or 1.
    group_col : str
        Column defining the groups.
    subset : str, optional
        If "failure", restrict to failure scenarios. If "nominal",
        restrict to nominal trials.

    Returns
    -------
    dict
        test used, statistic, p value, effect size, and the table.
    """
    sub = df.copy()
    if subset == "failure":
        sub = sub[sub["scenario"] != NOMINAL]
    elif subset == "nominal":
        sub = sub[sub["scenario"] == NOMINAL]

    table = pd.crosstab(sub[group_col], sub[outcome])
    # Ensure both outcome columns exist so the table is well formed.
    for val in (0, 1):
        if val not in table.columns:
            table[val] = 0
    table = table[[0, 1]]

    if table.values.sum() == 0 or (table.sum(axis=1) == 0).any():
        return {"note": "empty contingency table", "outcome": outcome}

    chi2, p, dof, expected = stats.chi2_contingency(table.values)
    test_used = "chi-square"

    # Fisher's exact for a sparse two by two table.
    if table.shape == (2, 2) and (expected < 5).any():
        _, p = stats.fisher_exact(table.values)
        test_used = "fisher exact"
        chi2 = float("nan")

    n = table.values.sum()
    min_dim = min(table.shape) - 1
    cramers_v = (
        float(np.sqrt(chi2 / (n * min_dim)))
        if test_used == "chi-square" and min_dim > 0 and not np.isnan(chi2)
        else float("nan")
    )

    return {
        "outcome": outcome,
        "test": test_used,
        "statistic": float(chi2),
        "p": float(p),
        "cramers_v": cramers_v,
        "table": table,
    }


def pairwise_proportions(df, outcome, subset="failure", reference="unmonitored"):
    """
    Compare each condition against a reference on a binary outcome, using a
    two-proportion z test with Bonferroni correction across the family.

    This stands in for Tukey HSD on the binary outcomes, since Tukey
    assumes continuous normal data. For the continuous metrics use
    tukey_hsd. Both share the Bonferroni spirit the methodology asks for.

    Returns
    -------
    list of dict
        Per-comparison rate difference, raw p, and Bonferroni p.
    """
    sub = df.copy()
    if subset == "failure":
        sub = sub[sub["scenario"] != NOMINAL]
    elif subset == "nominal":
        sub = sub[sub["scenario"] == NOMINAL]

    others = [c for c in CONDITIONS if c != reference]
    m = len(others)
    ref = sub[sub["condition"] == reference][outcome]
    ref_rate = ref.mean() if len(ref) else float("nan")

    results = []
    for cond in others:
        grp = sub[sub["condition"] == cond][outcome]
        if len(grp) == 0 or len(ref) == 0:
            continue
        rate = grp.mean()
        # Pooled two-proportion z test.
        n1, n2 = len(grp), len(ref)
        p_pool = (grp.sum() + ref.sum()) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        if se > 0:
            z = (rate - ref_rate) / se
            p_raw = float(2 * stats.norm.sf(abs(z)))
        else:
            z, p_raw = 0.0, 1.0
        results.append({
            "condition": cond,
            "reference": reference,
            "rate": float(rate),
            "reference_rate": float(ref_rate),
            "difference": float(rate - ref_rate),
            "z": float(z),
            "p_raw": p_raw,
            "p_bonferroni": float(min(p_raw * m, 1.0)),
        })
    return results


def tukey_hsd(df, metric):
    """
    Tukey HSD across conditions on a continuous metric, with the family
    wise control Tukey provides built in. Returns each pair's mean
    difference and p value. Missing metric rows are dropped.
    """
    sub = df[["condition", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna(subset=[metric])

    groups = [
        sub[sub["condition"] == c][metric].values
        for c in CONDITIONS
        if len(sub[sub["condition"] == c]) > 1
    ]
    labels = [c for c in CONDITIONS if len(sub[sub["condition"] == c]) > 1]

    if len(groups) < 2:
        return {"note": "insufficient groups for Tukey", "metric": metric}

    try:
        res = stats.tukey_hsd(*groups)
    except Exception as exc:  # pragma: no cover - defensive
        return {"note": f"tukey failed {exc}", "metric": metric}

    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append({
                "pair": f"{labels[i]} vs {labels[j]}",
                "mean_difference": float(res.statistic[i, j]),
                "p": float(res.pvalue[i, j]),
            })
    return {"metric": metric, "pairs": pairs}


def evaluate_hypothesis(df, alpha=0.05):
    """
    Test the three pre-registered conditions and return a plain verdict.

    Returns
    -------
    dict
        Per-condition booleans, the numbers behind them, and an overall
        pass flag that is True only if all three hold.
    """
    fail = df[df["scenario"] != NOMINAL]
    nominal = df[df["scenario"] == NOMINAL]

    def rate(frame, cond, col):
        cell = frame[frame["condition"] == cond][col]
        return float(cell.mean()) if len(cell) else float("nan")

    # Condition 1. Dual-stream safety intervention rate in failure groups
    # is statistically higher than unmonitored.
    dual_si = rate(fail, "dual_stream", "safety_intervention")
    unmon_si = rate(fail, "unmonitored", "safety_intervention")
    si_test = binary_test(
        fail[fail["condition"].isin(["unmonitored", "dual_stream"])],
        "safety_intervention",
    )
    cond1 = bool(dual_si > unmon_si and si_test.get("p", 1.0) < alpha)

    # Condition 2. Dual-stream nominal false positive rate is at most ten
    # points above unmonitored.
    dual_fp = rate(nominal, "dual_stream", "false_positive")
    unmon_fp = rate(nominal, "unmonitored", "false_positive")
    cond2 = bool((dual_fp - unmon_fp) <= PP_TOLERANCE)

    # Condition 3. Dual-stream nominal completion rate is within ten points
    # of unmonitored.
    dual_comp = rate(nominal, "dual_stream", "trial_completed")
    unmon_comp = rate(nominal, "unmonitored", "trial_completed")
    cond3 = bool(abs(dual_comp - unmon_comp) <= PP_TOLERANCE)

    return {
        "condition_1_safety_higher": {
            "holds": cond1,
            "dual_rate": dual_si,
            "unmonitored_rate": unmon_si,
            "test": si_test.get("test"),
            "p": si_test.get("p"),
        },
        "condition_2_false_positive_bounded": {
            "holds": cond2,
            "dual_rate": dual_fp,
            "unmonitored_rate": unmon_fp,
            "gap_pp": (dual_fp - unmon_fp) * 100,
            "tolerance_pp": PP_TOLERANCE * 100,
        },
        "condition_3_completion_preserved": {
            "holds": cond3,
            "dual_rate": dual_comp,
            "unmonitored_rate": unmon_comp,
            "gap_pp": abs(dual_comp - unmon_comp) * 100,
            "tolerance_pp": PP_TOLERANCE * 100,
        },
        "all_three_hold": bool(cond1 and cond2 and cond3),
    }


def full_report(csv_path):
    """
    Run every test and print a readable report. Returns the assembled
    results dict so a caller can use the numbers directly.
    """
    df = load_results(csv_path)
    print(f"Loaded {len(df)} trials from {csv_path}\n")

    print("=" * 64)
    print("CONTINUOUS METRICS, two-way ANOVA (condition x scenario)")
    print("=" * 64)
    anova_results = {}
    for metric in CONTINUOUS_METRICS:
        res = two_way_anova(df, metric)
        anova_results[metric] = res
        print(f"\n{metric}  (n={res.get('n', 0)})")
        if "note" in res:
            print(f"  {res['note']}")
            continue
        for effect in ("condition", "scenario", "interaction"):
            e = res[effect]
            tag = ""
            if effect == "interaction" and res.get("interaction_unreliable"):
                tag = "  (unbalanced design, interaction unreliable)"
            print(
                f"  {effect:12s} F={e['F']:7.3f}  p={e['p']:.4f}  "
                f"eta^2={e['eta_squared']:.3f}{tag}"
            )

    print("\n" + "=" * 64)
    print("BINARY OUTCOMES, chi-square or Fisher's exact across conditions")
    print("=" * 64)
    binary_results = {}
    for outcome in BINARY_OUTCOMES:
        # Safety and unsafe outcomes are about failure scenarios. Completion
        # is reported on nominal trials for the false positive story, and
        # also overall. We report failure-scenario tables here.
        res = binary_test(df, outcome, subset="failure")
        binary_results[outcome] = res
        if "note" in res:
            print(f"\n{outcome}: {res['note']}")
            continue
        v = res["cramers_v"]
        v_str = f"{v:.3f}" if not np.isnan(v) else "n/a"
        print(
            f"\n{outcome} (failure scenarios)\n"
            f"  test={res['test']}  stat={res['statistic']:.3f}  "
            f"p={res['p']:.4f}  Cramer's V={v_str}"
        )

    print("\n" + "=" * 64)
    print("PAIRWISE vs unmonitored, safety intervention (Bonferroni)")
    print("=" * 64)
    pw = pairwise_proportions(df, "safety_intervention", subset="failure")
    for r in pw:
        print(
            f"  {r['condition']:16s} rate={r['rate']:.2%} "
            f"(ref {r['reference_rate']:.2%})  diff={r['difference']:+.2%}  "
            f"p_bonf={r['p_bonferroni']:.4f}"
        )

    print("\n" + "=" * 64)
    print("TUKEY HSD on handover duration across conditions")
    print("=" * 64)
    tk = tukey_hsd(df, "handover_duration_s")
    if "pairs" in tk:
        for pair in tk["pairs"]:
            print(
                f"  {pair['pair']:36s} mean diff={pair['mean_difference']:+.3f}  "
                f"p={pair['p']:.4f}"
            )
    else:
        print(f"  {tk.get('note')}")

    print("\n" + "=" * 64)
    print("PRE-REGISTERED HYPOTHESIS VERDICT")
    print("=" * 64)
    verdict = evaluate_hypothesis(df)
    c1 = verdict["condition_1_safety_higher"]
    c2 = verdict["condition_2_false_positive_bounded"]
    c3 = verdict["condition_3_completion_preserved"]

    print(
        f"\n  1. Safety intervention higher in failures?  "
        f"{'HOLDS' if c1['holds'] else 'FAILS'}"
    )
    print(
        f"     dual {c1['dual_rate']:.1%} vs unmonitored "
        f"{c1['unmonitored_rate']:.1%}, {c1['test']} p={c1['p']:.4f}"
    )
    print(
        f"\n  2. Nominal false positives within 10pp?     "
        f"{'HOLDS' if c2['holds'] else 'FAILS'}"
    )
    print(
        f"     dual {c2['dual_rate']:.1%} vs unmonitored "
        f"{c2['unmonitored_rate']:.1%}, gap {c2['gap_pp']:+.1f}pp "
        f"(tolerance {c2['tolerance_pp']:.0f}pp)"
    )
    print(
        f"\n  3. Nominal completion within 10pp?          "
        f"{'HOLDS' if c3['holds'] else 'FAILS'}"
    )
    print(
        f"     dual {c3['dual_rate']:.1%} vs unmonitored "
        f"{c3['unmonitored_rate']:.1%}, gap {c3['gap_pp']:.1f}pp "
        f"(tolerance {c3['tolerance_pp']:.0f}pp)"
    )

    print(
        f"\n  OVERALL: dual-stream "
        f"{'WINS, all three conditions hold' if verdict['all_three_hold'] else 'does NOT win, at least one condition fails'}"
    )

    return {
        "anova": anova_results,
        "binary": binary_results,
        "pairwise_safety": pw,
        "tukey_duration": tk,
        "hypothesis": verdict,
    }


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Analyse handover results.")
    parser.add_argument(
        "csv", nargs="?", default="data/handover_results_pilot.csv",
        help="Path to the results CSV.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(
            f"No results file at {args.csv}. Run experiment.runner first."
        )

    full_report(args.csv)
