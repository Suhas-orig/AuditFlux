import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import BFW_DATA_PATH, OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = ["vgg16", "resnet50", "senet50"]

THRESHOLDS = [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70
]

BOOTSTRAP_REPETITIONS = 1000
RANDOM_SEED = 42
METRICS = ["FNR", "FPR"]
CHECKPOINT_EVERY = 100
TIE_TOLERANCE = 1e-12


# ============================================================
# HELPERS
# ============================================================

def calculate_metrics_fast(labels, scores, group_codes, group_names, threshold):
    """Calculate group-level TPR, FNR and FPR using integer counts."""
    predictions = (scores >= threshold).astype(np.int8)

    # 0 = TN, 1 = FP, 2 = FN, 3 = TP.
    outcome_codes = labels * 2 + predictions
    combined_codes = group_codes * 4 + outcome_codes

    counts = np.bincount(
        combined_codes,
        minlength=len(group_names) * 4
    ).reshape(len(group_names), 4)

    tn = counts[:, 0]
    fp = counts[:, 1]
    fn = counts[:, 2]
    tp = counts[:, 3]

    positives = tp + fn
    negatives = fp + tn

    tpr = np.divide(
        tp, positives,
        out=np.full(len(group_names), np.nan, dtype=float),
        where=positives > 0
    )
    fnr = np.divide(
        fn, positives,
        out=np.full(len(group_names), np.nan, dtype=float),
        where=positives > 0
    )
    fpr = np.divide(
        fp, negatives,
        out=np.full(len(group_names), np.nan, dtype=float),
        where=negatives > 0
    )

    return pd.DataFrame({
        "group": group_names,
        "TPR": tpr,
        "FNR": fnr,
        "FPR": fpr
    })


def get_ranking_and_rank_positions(metrics_df, metric):
    """
    Return a deterministic display order and average rank positions.
    Tied metric values receive equal average ranks.
    """
    values = metrics_df[metric].to_numpy(dtype=float)

    if np.isnan(values).any():
        raise ValueError(
            f"NaN encountered in {metric}; at least one group "
            "has no valid positive/negative examples in this sample."
        )

    ranking_indices = np.argsort(-values, kind="stable")
    ranking = metrics_df.iloc[ranking_indices]["group"].tolist()
    rank_positions = rankdata(-values, method="average")

    max_value = np.max(values)
    worst_groups = metrics_df.loc[
        np.isclose(values, max_value, rtol=0.0, atol=TIE_TOLERANCE),
        "group"
    ].tolist()

    return ranking, rank_positions, worst_groups


def calculate_gap(metrics_df, metric):
    """Maximum minus minimum group error rate."""
    values = metrics_df[metric].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError(f"NaN encountered while calculating {metric} gap.")
    return float(values.max() - values.min())


def save_checkpoints(raw_results, transition_results, raw_path, transition_path):
    """Persist accumulated results without changing the experiment."""
    pd.DataFrame(raw_results).to_csv(raw_path, index=False)
    pd.DataFrame(transition_results).to_csv(transition_path, index=False)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("PAIRED THRESHOLD BOOTSTRAP — ALL THREE MODELS")
print("=" * 72)

required_columns = ["label", "a1"] + MODELS
df = pd.read_csv(BFW_DATA_PATH, usecols=required_columns)

labels = df["label"].to_numpy(dtype=np.int8)
group_names, group_codes = np.unique(
    df["a1"].to_numpy(),
    return_inverse=True
)
group_codes = group_codes.astype(np.int8, copy=False)

scores_by_model = {
    model: df[model].to_numpy(dtype=np.float64)
    for model in MODELS
}

n = len(df)
del df
gc.collect()

print(f"Dataset size: {n:,}")
print(f"Models: {MODELS}")
print(f"Groups: {list(group_names)}")
print(f"Thresholds: {THRESHOLDS}")
print(f"Bootstrap repetitions: {BOOTSTRAP_REPETITIONS}")
print(f"Random seed: {RANDOM_SEED}")
print(
    "Pairing design: each bootstrap sample is reused across "
    "all thresholds and all three models."
)


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw_path = OUTPUT_DIR / "paired_threshold_bootstrap_all_models_raw.csv"
transition_path = (
    OUTPUT_DIR / "paired_threshold_bootstrap_all_models_transitions.csv"
)
summary_path = (
    OUTPUT_DIR / "paired_threshold_bootstrap_all_models_summary.csv"
)
reference_path = (
    OUTPUT_DIR / "paired_threshold_bootstrap_all_models_reference.csv"
)


# ============================================================
# REFERENCE AUDITS
# ============================================================

print("\nCalculating full-dataset reference audits...")

reference_rows = []

for model in MODELS:
    for threshold in THRESHOLDS:
        metrics = calculate_metrics_fast(
            labels,
            scores_by_model[model],
            group_codes,
            group_names,
            threshold
        )

        for _, row in metrics.iterrows():
            reference_rows.append({
                "model": model,
                "threshold": threshold,
                "group": row["group"],
                "TPR": row["TPR"],
                "FNR": row["FNR"],
                "FPR": row["FPR"]
            })

pd.DataFrame(reference_rows).to_csv(reference_path, index=False)


# ============================================================
# BOOTSTRAP
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)
raw_results = []
transition_results = []
completed_repetitions = 0

# Resume from the latest complete checkpoint when both checkpoint files
# exist. The random generator is advanced through the already-completed
# samples so subsequent bootstrap samples match an uninterrupted run.
if raw_path.exists() and transition_path.exists():
    print("\\nCheckpoint files found. Checking whether they can be resumed...")
    saved_raw = pd.read_csv(raw_path)
    saved_transitions = pd.read_csv(transition_path)

    if not saved_raw.empty and not saved_transitions.empty:
        candidate = int(saved_raw["bootstrap_id"].max())
        expected_raw_per_rep = (
            len(MODELS) * len(THRESHOLDS) * len(group_names)
        )
        expected_transitions_per_rep = (
            len(MODELS) * len(METRICS) * (len(THRESHOLDS) - 1)
        )

        raw_counts = saved_raw.groupby("bootstrap_id").size()
        transition_counts = saved_transitions.groupby("bootstrap_id").size()

        complete_ids = [
            rep for rep in range(1, candidate + 1)
            if raw_counts.get(rep, 0) == expected_raw_per_rep
            and transition_counts.get(rep, 0) == expected_transitions_per_rep
        ]

        if complete_ids and complete_ids == list(range(1, max(complete_ids) + 1)):
            completed_repetitions = max(complete_ids)
            raw_results = saved_raw[
                saved_raw["bootstrap_id"] <= completed_repetitions
            ].to_dict("records")
            transition_results = saved_transitions[
                saved_transitions["bootstrap_id"] <= completed_repetitions
            ].to_dict("records")

            print(
                f"Resuming after {completed_repetitions} complete "
                "bootstrap repetitions."
            )
            print("Replaying the random-sampling sequence to restore RNG state...")
            for _ in range(completed_repetitions):
                rng.integers(0, n, size=n)
        else:
            print(
                "Checkpoint files are incomplete or inconsistent. "
                "Starting a fresh run; existing files will be replaced "
                "at the next checkpoint."
            )
    else:
        print("Checkpoint files are empty. Starting a fresh run.")

for bootstrap_id in range(
    completed_repetitions + 1,
    BOOTSTRAP_REPETITIONS + 1
):

    if bootstrap_id % 100 == 0:
        print(f"Bootstrap {bootstrap_id}/{BOOTSTRAP_REPETITIONS}")

    # One sample is reused across all models and thresholds.
    sample_indices = rng.integers(0, n, size=n)
    boot_labels = labels[sample_indices]
    boot_group_codes = group_codes[sample_indices]

    for model in MODELS:
        boot_scores = scores_by_model[model][sample_indices]
        threshold_results = {}

        for threshold in THRESHOLDS:
            metrics = calculate_metrics_fast(
                boot_labels,
                boot_scores,
                boot_group_codes,
                group_names,
                threshold
            )
            threshold_results[threshold] = metrics

            for _, row in metrics.iterrows():
                raw_results.append({
                    "bootstrap_id": bootstrap_id,
                    "model": model,
                    "threshold": threshold,
                    "group": row["group"],
                    "TPR": row["TPR"],
                    "FNR": row["FNR"],
                    "FPR": row["FPR"]
                })

        for metric in METRICS:
            rank_positions = {}
            gaps = {}
            worst_groups = {}

            for threshold in THRESHOLDS:
                metrics = threshold_results[threshold]

                _, ranks, worst = get_ranking_and_rank_positions(
                    metrics, metric
                )
                rank_positions[threshold] = ranks
                gaps[threshold] = calculate_gap(metrics, metric)
                worst_groups[threshold] = set(worst)

            for i in range(len(THRESHOLDS) - 1):
                t1 = THRESHOLDS[i]
                t2 = THRESHOLDS[i + 1]

                tau, _ = kendalltau(
                    rank_positions[t1],
                    rank_positions[t2]
                )

                # Kendall's tau is undefined if either ranking is
                # completely tied (constant rank vector). Preserve
                # this as NaN rather than aborting the experiment.
                ranking_status = (
                    "defined" if not np.isnan(tau)
                    else "undefined_constant_ranking"
                )
                ranking_instability = (
                    float(1 - tau) if not np.isnan(tau) else np.nan
                )

                # A worst-group reversal is only well-defined when each
                # threshold has exactly one worst group. If either side is
                # tied, preserve the result as undefined rather than forcing
                # an arbitrary winner.
                worst_group_comparison_defined = (
                    len(worst_groups[t1]) == 1
                    and len(worst_groups[t2]) == 1
                )
                reversal = (
                    int(worst_groups[t1] != worst_groups[t2])
                    if worst_group_comparison_defined else np.nan
                )

                magnitude_change = abs(gaps[t2] - gaps[t1])

                transition_results.append({
                    "bootstrap_id": bootstrap_id,
                    "model": model,
                    "metric": metric,
                    "threshold_from": t1,
                    "threshold_to": t2,
                    "worst_group_from": "|".join(sorted(worst_groups[t1])),
                    "worst_group_to": "|".join(sorted(worst_groups[t2])),
                    "worst_group_reversal": reversal,
                    "worst_group_comparison_defined":
                        worst_group_comparison_defined,
                    "kendall_tau": (
                        float(tau) if not np.isnan(tau) else np.nan
                    ),
                    "ranking_instability": ranking_instability,
                    "ranking_comparison_status": ranking_status,
                    "gap_from": gaps[t1],
                    "gap_to": gaps[t2],
                    "magnitude_change": magnitude_change
                })

        del boot_scores, threshold_results

    del sample_indices, boot_labels, boot_group_codes

    if bootstrap_id % CHECKPOINT_EVERY == 0:
        print(f"  Saving checkpoint at {bootstrap_id}...")
        save_checkpoints(
            raw_results,
            transition_results,
            raw_path,
            transition_path
        )
        gc.collect()


# ============================================================
# SAVE FINAL RAW RESULTS
# ============================================================

save_checkpoints(
    raw_results,
    transition_results,
    raw_path,
    transition_path
)


# ============================================================
# SUMMARY
# ============================================================

transition_df = pd.DataFrame(transition_results)
summary_rows = []

for model in MODELS:
    for metric in METRICS:
        metric_df = transition_df[
            (transition_df["model"] == model)
            & (transition_df["metric"] == metric)
        ]

        for t1, t2 in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
            transition = metric_df[
                (metric_df["threshold_from"] == t1)
                & (metric_df["threshold_to"] == t2)
            ]

            valid_tau = transition["kendall_tau"].dropna()
            valid_instability = transition["ranking_instability"].dropna()

            summary_rows.append({
                "model": model,
                "metric": metric,
                "threshold_from": t1,
                "threshold_to": t2,
                "bootstrap_repetitions": len(transition),
                "defined_ranking_comparisons": len(valid_tau),
                "undefined_ranking_comparisons":
                    transition["kendall_tau"].isna().sum(),
                "defined_worst_group_comparisons":
                    transition["worst_group_reversal"].notna().sum(),
                "undefined_worst_group_comparisons":
                    transition["worst_group_reversal"].isna().sum(),
                "worst_group_reversal_rate":
                    transition["worst_group_reversal"].mean(),
                "kendall_tau_mean": valid_tau.mean(),
                "kendall_tau_sd": valid_tau.std(),
                "ranking_instability_mean": valid_instability.mean(),
                "ranking_instability_sd": valid_instability.std(),
                "magnitude_change_mean":
                    transition["magnitude_change"].mean(),
                "magnitude_change_sd":
                    transition["magnitude_change"].std(),
                "magnitude_change_q025":
                    transition["magnitude_change"].quantile(0.025),
                "magnitude_change_median":
                    transition["magnitude_change"].quantile(0.50),
                "magnitude_change_q975":
                    transition["magnitude_change"].quantile(0.975)
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(summary_path, index=False)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 72)
print("PAIRED THRESHOLD BOOTSTRAP SUMMARY")
print("=" * 72)

for model in MODELS:
    print(f"\n{'#' * 24} {model.upper()} {'#' * 24}")

    for metric in METRICS:
        print(f"\n{metric}")

        metric_summary = summary_df[
            (summary_df["model"] == model)
            & (summary_df["metric"] == metric)
        ]

        for _, row in metric_summary.iterrows():
            tau_text = (
                f"{row['kendall_tau_mean']:.3f} "
                f"(SD={row['kendall_tau_sd']:.3f})"
                if pd.notna(row["kendall_tau_mean"])
                else "undefined"
            )
            instability_text = (
                f"{row['ranking_instability_mean']:.3f}"
                if pd.notna(row["ranking_instability_mean"])
                else "undefined"
            )

            print(
                f"  {row['threshold_from']:.2f} -> "
                f"{row['threshold_to']:.2f}: "
                f"reversal={row['worst_group_reversal_rate']:.3f}, "
                f"tau={tau_text}, "
                f"instability={instability_text}, "
                f"undefined={int(row['undefined_ranking_comparisons'])}, "
                f"gap-change={row['magnitude_change_mean']:.6f}"
            )


# ============================================================
# FILES
# ============================================================

print("\n")
print("=" * 72)
print("FILES SAVED")
print("=" * 72)
print(raw_path)
print(transition_path)
print(summary_path)
print(reference_path)
print("\nExperiment complete.")
