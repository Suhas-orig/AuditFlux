import pandas as pd
import numpy as np
from scipy.stats import kendalltau, rankdata
from config import OUTPUT_DIR

# ============================================================
# CONFIGURATION
# ============================================================
MODELS = ["vgg16", "resnet50", "senet50"]
METRICS = ["FNR", "FPR"]
TIE_ATOL = 1e-12

THRESHOLD_OUTPUT_ALL = OUTPUT_DIR / "threshold_instability_metrics_all_models_revised.csv"
SAMPLE_OUTPUT_ALL = OUTPUT_DIR / "sample_instability_metrics_all_models_revised.csv"
SAMPLE_SUMMARY_ALL = OUTPUT_DIR / "sample_instability_summary_all_models_revised.csv"
SAMPLE_RANKING_ALL = OUTPUT_DIR / "sample_size_ranking_comparison_all_models_revised.csv"


def load_required(path, required_columns, description):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\nRun the corresponding experiment first."
        )
    data = pd.read_csv(path)
    missing = set(required_columns).difference(data.columns)
    if missing:
        raise ValueError(f"{description} is missing required columns: {sorted(missing)}")
    return data


def group_metric_values(data, metric):
    """Return one mean metric value per demographic group."""
    return data.groupby("group")[metric].mean().dropna().sort_index()


def rank_values(values):
    """Average ranks; rank 1 is worst (largest error metric)."""
    return pd.Series(rankdata(-values.to_numpy(), method="average"), index=values.index)


def worst_groups(values):
    """Return all groups tied for the maximum, using a strict numerical tolerance."""
    if values.empty:
        return []
    maximum = values.max()
    return sorted(values.index[np.isclose(values.to_numpy(), maximum, rtol=0, atol=TIE_ATOL)].tolist())


def gap_from_values(values):
    return float(values.max() - values.min()) if not values.empty else np.nan


def compare_rank_series(previous_values, current_values):
    """Kendall tau-b on aligned average ranks, plus changed rank positions."""
    common = sorted(set(previous_values.index).intersection(current_values.index))
    if len(common) < 2:
        return np.nan, np.nan, np.nan
    prev_ranks = rank_values(previous_values.loc[common])
    curr_ranks = rank_values(current_values.loc[common])
    tau, p_value = kendalltau(prev_ranks.to_numpy(), curr_ranks.to_numpy(), variant="b")
    changed = int((~np.isclose(prev_ranks.to_numpy(), curr_ranks.to_numpy(), rtol=0, atol=TIE_ATOL)).sum())
    return float(tau), float(p_value), changed


# ============================================================
# 1. THRESHOLD INSTABILITY
# ============================================================
threshold_frames = []

for model in MODELS:
    threshold_df = load_required(
        OUTPUT_DIR / f"threshold_sweep_{model}.csv",
        {"threshold", "group", *METRICS},
        f"{model} threshold-sweep results",
    )
    model_rows = []

    for metric in METRICS:
        previous_values = None
        previous_worst = None
        previous_gap = None
        previous_threshold = None

        for threshold in sorted(threshold_df["threshold"].dropna().unique()):
            current = threshold_df[threshold_df["threshold"] == threshold]
            values = group_metric_values(current, metric)
            if values.empty:
                continue

            tied_worst = worst_groups(values)
            gap = gap_from_values(values)
            if previous_values is None:
                tau = p_value = ranking_instability = np.nan
                changed_count = np.nan
                signed_gap_change = absolute_gap_change = np.nan
                reversal = np.nan
            else:
                tau, p_value, changed_count = compare_rank_series(previous_values, values)
                ranking_instability = 1 - tau if pd.notna(tau) else np.nan
                signed_gap_change = gap - previous_gap
                absolute_gap_change = abs(signed_gap_change)
                reversal = int(set(tied_worst).isdisjoint(previous_worst))

            model_rows.append({
                "model": model,
                "metric": metric,
                "threshold": threshold,
                "previous_threshold": previous_threshold,
                "worst_group": "|".join(tied_worst),
                "number_of_tied_worst_groups": len(tied_worst),
                "group_metric_gap": gap,
                "worst_group_reversal": reversal,
                "kendall_tau_b_previous": tau,
                "kendall_p_value_descriptive": p_value,
                "changed_group_rank_count": changed_count,
                "ranking_instability": ranking_instability,
                "signed_gap_change": signed_gap_change,
                "absolute_gap_change": absolute_gap_change,
            })
            previous_values = values
            previous_worst = tied_worst
            previous_gap = gap
            previous_threshold = threshold

    threshold_frames.append(pd.DataFrame(model_rows))

threshold_results = pd.concat(threshold_frames, ignore_index=True)
threshold_results.to_csv(THRESHOLD_OUTPUT_ALL, index=False)
for model in MODELS:
    threshold_results[threshold_results.model == model].to_csv(
        OUTPUT_DIR / f"threshold_instability_metrics_{model}_revised.csv", index=False
    )


# ============================================================
# 2. SAMPLE-SIZE INSTABILITY — REPETITION-LEVEL RESULTS
# ============================================================
sample_frames = []
for model in MODELS:
    sample_df = load_required(
        OUTPUT_DIR / f"sample_size_sweep_{model}.csv",
        {"sample_size", "group", "repetition", *METRICS},
        f"{model} sample-size results",
    )
    if "model" in sample_df.columns:
        sample_df = sample_df[sample_df.model.astype(str).str.lower() == model].copy()

    rows = []
    for metric in METRICS:
        for sample_size in sorted(sample_df.sample_size.dropna().unique(), reverse=True):
            size_df = sample_df[sample_df.sample_size == sample_size]
            for repetition, rep_data in size_df.groupby("repetition", sort=True):
                values = group_metric_values(rep_data, metric)
                if values.empty:
                    continue
                tied_worst = worst_groups(values)
                rows.append({
                    "model": model,
                    "metric": metric,
                    "sample_size": int(sample_size),
                    "repetition": int(repetition),
                    "worst_group": "|".join(tied_worst),
                    "number_of_tied_worst_groups": len(tied_worst),
                    "group_metric_gap": gap_from_values(values),
                })
    sample_frames.append(pd.DataFrame(rows))

sample_results = pd.concat(sample_frames, ignore_index=True)
sample_results.to_csv(SAMPLE_OUTPUT_ALL, index=False)
for model in MODELS:
    sample_results[sample_results.model == model].to_csv(
        OUTPUT_DIR / f"sample_instability_metrics_{model}_revised.csv", index=False
    )


# ============================================================
# 3. SAMPLE-SIZE SUMMARY + RANKING CONSISTENCY
# ============================================================
summary_rows = []
ranking_rows = []

for (model, metric, sample_size), subset in sample_results.groupby(
    ["model", "metric", "sample_size"], sort=True
):
    counts = subset.worst_group.str.split("|").explode().value_counts()
    n_repetitions = subset.repetition.nunique()

    summary_rows.append({
        "model": model,
        "metric": metric,
        "sample_size": int(sample_size),
        "repetitions": int(n_repetitions),
        "mean_gap": subset.group_metric_gap.mean(),
        "sd_gap": subset.group_metric_gap.std(ddof=1),
        "min_gap": subset.group_metric_gap.min(),
        "max_gap": subset.group_metric_gap.max(),
        "most_frequent_worst_group": "|".join(counts[counts == counts.max()].index.tolist()),
        "most_frequent_worst_group_count": int(counts.max()),
        "most_frequent_worst_group_frequency": float(counts.max() / n_repetitions),
    })

    # Compare each repetition's group ranking with the consensus ranking
    # formed from the mean group metrics across all repetitions at this size.
    raw_model = None
    # Input files are reloaded below only once per model/metric/size for consensus values.
    input_path = OUTPUT_DIR / f"sample_size_sweep_{model}.csv"
    raw_model = pd.read_csv(input_path)
    if "model" in raw_model.columns:
        raw_model = raw_model[raw_model.model.astype(str).str.lower() == model]
    consensus_data = raw_model[raw_model.sample_size == sample_size]
    consensus_values = group_metric_values(consensus_data, metric)

    taus = []
    for repetition, rep_data in consensus_data.groupby("repetition", sort=True):
        rep_values = group_metric_values(rep_data, metric)
        tau, _, changed = compare_rank_series(consensus_values, rep_values)
        if pd.notna(tau):
            taus.append((tau, changed))

    ranking_rows.append({
        "model": model,
        "metric": metric,
        "sample_size": int(sample_size),
        "repetitions_compared": len(taus),
        "mean_kendall_tau_b_to_consensus": float(np.mean([x[0] for x in taus])) if taus else np.nan,
        "sd_kendall_tau_b_to_consensus": float(np.std([x[0] for x in taus], ddof=1)) if len(taus) > 1 else np.nan,
        "mean_changed_rank_positions": float(np.mean([x[1] for x in taus])) if taus else np.nan,
    })

# Compare consensus rankings between sample sizes. These are aggregate,
# descriptive comparisons; repetition indices are NOT treated as paired samples.
for model in MODELS:
    raw = pd.read_csv(OUTPUT_DIR / f"sample_size_sweep_{model}.csv")
    if "model" in raw.columns:
        raw = raw[raw.model.astype(str).str.lower() == model]
    for metric in METRICS:
        consensus = {}
        for size in sorted(raw.sample_size.dropna().unique(), reverse=True):
            consensus[size] = group_metric_values(raw[raw.sample_size == size], metric)
        sizes = list(consensus)
        for i in range(1, len(sizes)):
            larger, smaller = sizes[i - 1], sizes[i]
            tau, p_value, changed = compare_rank_series(consensus[larger], consensus[smaller])
            large_worst = worst_groups(consensus[larger])
            small_worst = worst_groups(consensus[smaller])
            ranking_rows.append({
                "model": model,
                "metric": metric,
                "comparison": "consensus_ranking_between_sample_sizes",
                "larger_sample_size": int(larger),
                "smaller_sample_size": int(smaller),
                "kendall_tau_b": tau,
                "kendall_p_value_descriptive": p_value,
                "changed_rank_positions": changed,
                "larger_size_worst_group": "|".join(large_worst),
                "smaller_size_worst_group": "|".join(small_worst),
                "worst_group_changed": int(set(large_worst).isdisjoint(small_worst)),
            })

sample_summary = pd.DataFrame(summary_rows)
sample_ranking = pd.DataFrame(ranking_rows)
sample_summary.to_csv(SAMPLE_SUMMARY_ALL, index=False)
sample_ranking.to_csv(SAMPLE_RANKING_ALL, index=False)


# ============================================================
# 4. CONCISE CONSOLE REPORT
# ============================================================
print("\nTHRESHOLD INSTABILITY — REVISED")
print(threshold_results.to_string(index=False))
print("\nSAMPLE-SIZE SUMMARY — REVISED")
print(sample_summary.to_string(index=False))
print("\nSAMPLE-SIZE RANKING COMPARISONS — REVISED")
print(sample_ranking.to_string(index=False))
print("\nDONE")
print(f"Threshold results: {THRESHOLD_OUTPUT_ALL}")
print(f"Sample repetition results: {SAMPLE_OUTPUT_ALL}")
print(f"Sample summary: {SAMPLE_SUMMARY_ALL}")
print(f"Sample ranking comparisons: {SAMPLE_RANKING_ALL}")
