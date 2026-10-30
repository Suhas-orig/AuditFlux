import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from config import OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = ["vgg16", "resnet50", "senet50"]
METRICS = ["FNR", "FPR"]

# Corrected outputs are kept separate from the first run because the
# original threshold Kendall-tau calculation compared group-name strings
# in ranking order instead of comparing each group's rank position.
THRESHOLD_OUTPUT_ALL = OUTPUT_DIR / "threshold_instability_metrics_all_models_corrected.csv"
SAMPLE_OUTPUT_ALL = OUTPUT_DIR / "sample_instability_metrics_all_models_corrected.csv"


# ============================================================
# HELPERS
# ============================================================

def load_required(path, required_columns, description):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\nRun the corresponding experiment first."
        )
    data = pd.read_csv(path)
    missing = set(required_columns).difference(data.columns)
    if missing:
        raise ValueError(
            f"{description} is missing required columns: {sorted(missing)}"
        )
    return data


def group_metric_values(data, metric):
    """One metric value per demographic group."""
    return data.groupby("group")[metric].mean().dropna()


def get_ranking(data, metric):
    """Groups ordered from highest (worst) to lowest metric value."""
    values = group_metric_values(data, metric)
    return values.sort_values(ascending=False).index.tolist()


def calculate_gap(data, metric):
    """Difference between the highest and lowest group-level metric."""
    values = group_metric_values(data, metric)
    if values.empty:
        return np.nan
    return float(values.max() - values.min())


def compare_rankings(previous_ranking, current_ranking):
    """
    Correct Kendall tau: compare rank positions for the same group labels.
    Do not pass ordered lists of group-name strings directly to kendalltau.
    """
    common_groups = sorted(set(previous_ranking).intersection(current_ranking))
    if len(common_groups) < 2:
        return np.nan, np.nan, np.nan

    previous_positions = {
        group: rank for rank, group in enumerate(previous_ranking, start=1)
    }
    current_positions = {
        group: rank for rank, group in enumerate(current_ranking, start=1)
    }

    previous_ranks = [previous_positions[group] for group in common_groups]
    current_ranks = [current_positions[group] for group in common_groups]

    tau, p_value = kendalltau(previous_ranks, current_ranks)
    changed_count = sum(
        previous_positions[group] != current_positions[group]
        for group in common_groups
    )
    return float(tau), float(p_value), int(changed_count)


# ============================================================
# THRESHOLD INSTABILITY
# ============================================================

threshold_frames = []

for model in MODELS:
    input_path = OUTPUT_DIR / f"threshold_sweep_{model}.csv"
    threshold_df = load_required(
        input_path,
        {"threshold", "group", *METRICS},
        f"{model} threshold-sweep results"
    )

    model_results = []

    for metric in METRICS:
        thresholds = sorted(threshold_df["threshold"].dropna().unique())
        previous_worst = None
        previous_ranking = None
        previous_gap = None
        previous_threshold = None

        for threshold in thresholds:
            current = threshold_df[threshold_df["threshold"] == threshold].copy()
            ranking = get_ranking(current, metric)
            if not ranking:
                continue

            worst_group = ranking[0]
            gap = calculate_gap(current, metric)
            worst_reversal = (
                previous_worst is not None and worst_group != previous_worst
            )

            if previous_ranking is not None:
                tau, p_value, changed_count = compare_rankings(
                    previous_ranking, ranking
                )
                ranking_instability = 1 - tau if pd.notna(tau) else np.nan
            else:
                tau = p_value = ranking_instability = np.nan
                changed_count = np.nan

            magnitude_change = (
                abs(gap - previous_gap)
                if previous_gap is not None and pd.notna(gap) and pd.notna(previous_gap)
                else np.nan
            )

            model_results.append({
                "model": model,
                "metric": metric,
                "threshold": threshold,
                "previous_threshold": previous_threshold,
                "worst_group": worst_group,
                "group_metric_gap": gap,
                "worst_group_reversal": int(worst_reversal),
                "kendall_tau_previous": tau,
                "kendall_p_value_descriptive": p_value,
                "changed_group_rank_count": changed_count,
                "ranking_instability": ranking_instability,
                "absolute_gap_change": magnitude_change
            })

            previous_worst = worst_group
            previous_ranking = ranking
            previous_gap = gap
            previous_threshold = threshold

    threshold_frames.append(pd.DataFrame(model_results))

threshold_results_df = pd.concat(threshold_frames, ignore_index=True)
threshold_results_df.to_csv(THRESHOLD_OUTPUT_ALL, index=False)

for model in MODELS:
    threshold_results_df[threshold_results_df["model"] == model].to_csv(
        OUTPUT_DIR / f"threshold_instability_metrics_{model}_corrected.csv",
        index=False
    )


# ============================================================
# SAMPLE-SIZE INSTABILITY
# ============================================================

sample_frames = []

for model in MODELS:
    input_path = OUTPUT_DIR / f"sample_size_sweep_{model}.csv"
    sample_df = load_required(
        input_path,
        {"sample_size", "group", "repetition", *METRICS},
        f"{model} sample-size results"
    )

    if "model" in sample_df.columns:
        sample_df = sample_df[
            sample_df["model"].astype(str).str.lower() == model.lower()
        ].copy()

    model_results = []

    for metric in METRICS:
        for sample_size in sorted(sample_df["sample_size"].dropna().unique(), reverse=True):
            size_df = sample_df[sample_df["sample_size"] == sample_size]

            for repetition, rep_data in size_df.groupby("repetition", sort=True):
                ranking = get_ranking(rep_data, metric)
                if not ranking:
                    continue

                gap = calculate_gap(rep_data, metric)
                model_results.append({
                    "model": model,
                    "metric": metric,
                    "sample_size": int(sample_size),
                    "repetition": int(repetition),
                    "worst_group": ranking[0],
                    "group_metric_gap": gap
                })

    sample_frames.append(pd.DataFrame(model_results))

sample_results_df = pd.concat(sample_frames, ignore_index=True)
sample_results_df.to_csv(SAMPLE_OUTPUT_ALL, index=False)

for model in MODELS:
    sample_results_df[sample_results_df["model"] == model].to_csv(
        OUTPUT_DIR / f"sample_instability_metrics_{model}_corrected.csv",
        index=False
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("CORRECTED THRESHOLD INSTABILITY — ALL MODELS")
print("=" * 72)
print(threshold_results_df.to_string(index=False))

print("\n" + "=" * 72)
print("WORST-GROUP COUNTS BY MODEL, METRIC, AND SAMPLE SIZE")
print("=" * 72)
worst_group_counts = (
    sample_results_df
    .groupby(["model", "metric", "sample_size", "worst_group"])
    .size()
    .rename("repetitions")
    .reset_index()
)
print(worst_group_counts.to_string(index=False))

print("\n" + "=" * 72)
print("AVERAGE GAP BY MODEL, METRIC, AND SAMPLE SIZE")
print("=" * 72)
gap_summary = (
    sample_results_df
    .groupby(["model", "metric", "sample_size"])["group_metric_gap"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)
print(gap_summary.to_string(index=False))

print("\nDONE")
print(f"Corrected combined threshold results: {THRESHOLD_OUTPUT_ALL}")
print(f"Sample-size results: {SAMPLE_OUTPUT_ALL}")
print("Corrected per-model threshold and sample-size files were also saved.")
