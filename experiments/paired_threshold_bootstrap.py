import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from config import BFW_DATA_PATH, OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "resnet50"

THRESHOLDS = [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70
]

BOOTSTRAP_REPETITIONS = 1000

RANDOM_SEED = 42

METRICS = ["FNR", "FPR"]


# ============================================================
# HELPERS
# ============================================================

def calculate_metrics(labels, scores, groups, threshold):
    """
    Calculate FNR, FPR and TPR for every demographic group.
    """

    predictions = scores >= threshold

    rows = []

    for group in sorted(np.unique(groups)):

        mask = groups == group

        y = labels[mask]
        pred = predictions[mask]

        tp = np.sum((y == 1) & (pred == 1))
        fn = np.sum((y == 1) & (pred == 0))

        fp = np.sum((y == 0) & (pred == 1))
        tn = np.sum((y == 0) & (pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan

        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        rows.append({
            "threshold": threshold,
            "group": group,
            "TPR": tpr,
            "FNR": fnr,
            "FPR": fpr
        })

    return pd.DataFrame(rows)


def get_ranking(metrics_df, metric):
    """
    Return groups ordered from worst to best.

    For FNR/FPR:
        higher = worse
    """

    ranking_df = (
        metrics_df
        .groupby("group")[metric]
        .mean()
        .sort_values(ascending=False)
    )

    return list(ranking_df.index)


def compare_rankings(ranking_a, ranking_b):
    """
    Compare two rankings correctly.

    We align the rankings by group and compare each group's
    rank position.
    """

    rank_a = {
        group: position
        for position, group in enumerate(ranking_a)
    }

    rank_b = {
        group: position
        for position, group in enumerate(ranking_b)
    }

    groups = sorted(rank_a.keys())

    vector_a = [rank_a[g] for g in groups]
    vector_b = [rank_b[g] for g in groups]

    tau, _ = kendalltau(vector_a, vector_b)

    return tau


def calculate_gap(metrics_df, metric):
    """
    Maximum group disparity for a metric.
    """

    values = metrics_df[metric]

    return values.max() - values.min()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PAIRED THRESHOLD BOOTSTRAP EXPERIMENT")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    BFW_DATA_PATH,
    usecols=["label", MODEL, "a1"]
)

labels = df["label"].to_numpy()
scores = df[MODEL].to_numpy()
groups = df["a1"].to_numpy()

n = len(df)

print(f"Dataset size: {n:,}")
print(f"Model: {MODEL}")
print(f"Thresholds: {THRESHOLDS}")
print(f"Bootstrap repetitions: {BOOTSTRAP_REPETITIONS}")
print(f"Random seed: {RANDOM_SEED}")


# ============================================================
# BOOTSTRAP
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

raw_results = []
transition_results = []


for bootstrap_id in range(BOOTSTRAP_REPETITIONS):

    if (bootstrap_id + 1) % 100 == 0:
        print(
            f"Bootstrap "
            f"{bootstrap_id + 1}/{BOOTSTRAP_REPETITIONS}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # One bootstrap sample is generated and reused for EVERY
    # threshold.
    #
    # This is what makes this a PAIRED threshold experiment.
    # --------------------------------------------------------

    sample_indices = rng.integers(
        0,
        n,
        size=n
    )

    boot_labels = labels[sample_indices]
    boot_scores = scores[sample_indices]
    boot_groups = groups[sample_indices]

    threshold_results = {}

    # --------------------------------------------------------
    # Evaluate every threshold on the SAME bootstrap sample
    # --------------------------------------------------------

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            boot_labels,
            boot_scores,
            boot_groups,
            threshold
        )

        threshold_results[threshold] = metrics

        # ----------------------------------------------------
        # Save raw group-level results
        # ----------------------------------------------------

        for _, row in metrics.iterrows():

            raw_results.append({
                "bootstrap_id": bootstrap_id,
                "threshold": threshold,
                "group": row["group"],
                "TPR": row["TPR"],
                "FNR": row["FNR"],
                "FPR": row["FPR"]
            })

    # ========================================================
    # TRANSITION ANALYSIS
    # ========================================================

    for metric in METRICS:

        rankings = {}

        gaps = {}

        worst_groups = {}

        for threshold in THRESHOLDS:

            metrics = threshold_results[threshold]

            rankings[threshold] = get_ranking(
                metrics,
                metric
            )

            gaps[threshold] = calculate_gap(
                metrics,
                metric
            )

            worst_groups[threshold] = rankings[threshold][0]

        # ----------------------------------------------------
        # Compare consecutive thresholds
        # ----------------------------------------------------

        for i in range(len(THRESHOLDS) - 1):

            t1 = THRESHOLDS[i]
            t2 = THRESHOLDS[i + 1]

            ranking_1 = rankings[t1]
            ranking_2 = rankings[t2]

            tau = compare_rankings(
                ranking_1,
                ranking_2
            )

            ranking_instability = 1 - tau

            worst_group_reversal = int(
                worst_groups[t1] != worst_groups[t2]
            )

            magnitude_change = abs(
                gaps[t2] - gaps[t1]
            )

            transition_results.append({
                "bootstrap_id": bootstrap_id,
                "metric": metric,
                "threshold_from": t1,
                "threshold_to": t2,
                "worst_group_from": worst_groups[t1],
                "worst_group_to": worst_groups[t2],
                "worst_group_reversal": worst_group_reversal,
                "kendall_tau": tau,
                "ranking_instability": ranking_instability,
                "gap_from": gaps[t1],
                "gap_to": gaps[t2],
                "magnitude_change": magnitude_change
            })


# ============================================================
# SAVE RAW RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

raw_path = OUTPUT_DIR / (
    "paired_threshold_bootstrap_raw.csv"
)

transition_path = OUTPUT_DIR / (
    "paired_threshold_bootstrap_transitions.csv"
)

summary_path = OUTPUT_DIR / (
    "paired_threshold_bootstrap_summary.csv"
)

pd.DataFrame(raw_results).to_csv(
    raw_path,
    index=False
)

pd.DataFrame(transition_results).to_csv(
    transition_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

transition_df = pd.DataFrame(
    transition_results
)

summary_rows = []


for metric in METRICS:

    metric_df = transition_df[
        transition_df["metric"] == metric
    ]

    for t1, t2 in zip(
        THRESHOLDS[:-1],
        THRESHOLDS[1:]
    ):

        transition = metric_df[
            (metric_df["threshold_from"] == t1)
            &
            (metric_df["threshold_to"] == t2)
        ]

        reversal_rate = (
            transition["worst_group_reversal"].mean()
        )

        summary_rows.append({
            "metric": metric,
            "threshold_from": t1,
            "threshold_to": t2,

            "worst_group_reversal_rate":
                reversal_rate,

            "kendall_tau_mean":
                transition["kendall_tau"].mean(),

            "kendall_tau_sd":
                transition["kendall_tau"].std(),

            "ranking_instability_mean":
                transition[
                    "ranking_instability"
                ].mean(),

            "ranking_instability_sd":
                transition[
                    "ranking_instability"
                ].std(),

            "magnitude_change_mean":
                transition[
                    "magnitude_change"
                ].mean(),

            "magnitude_change_sd":
                transition[
                    "magnitude_change"
                ].std(),

            "magnitude_change_q025":
                transition[
                    "magnitude_change"
                ].quantile(0.025),

            "magnitude_change_median":
                transition[
                    "magnitude_change"
                ].quantile(0.50),

            "magnitude_change_q975":
                transition[
                    "magnitude_change"
                ].quantile(0.975)
        })


summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BOOTSTRAP TRANSITION SUMMARY")
print("=" * 70)

for metric in METRICS:

    print(f"\n{'=' * 20} {metric} {'=' * 20}")

    metric_summary = summary_df[
        summary_df["metric"] == metric
    ]

    for _, row in metric_summary.iterrows():

        print(
            f"\n{row['threshold_from']:.2f}"
            f" -> "
            f"{row['threshold_to']:.2f}"
        )

        print(
            f"  Worst-group reversal: "
            f"{row['worst_group_reversal_rate']:.3f}"
        )

        print(
            f"  Kendall tau: "
            f"{row['kendall_tau_mean']:.3f}"
            f" ± "
            f"{row['kendall_tau_sd']:.3f}"
        )

        print(
            f"  Ranking instability: "
            f"{row['ranking_instability_mean']:.3f}"
        )

        print(
            f"  Magnitude change: "
            f"{row['magnitude_change_mean']:.6f}"
        )


print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(raw_path)
print(transition_path)
print(summary_path)

print("\nExperiment complete.")