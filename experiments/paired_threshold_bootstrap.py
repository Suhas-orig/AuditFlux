import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from pathlib import Path
import sys
import gc

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

# Save intermediate results so a long run does not lose
# everything if the process is interrupted.
CHECKPOINT_EVERY = 100


# ============================================================
# HELPERS
# ============================================================

def calculate_metrics_fast(labels, scores, group_codes, group_names, threshold):
    """
    Calculate FNR, FPR and TPR for every demographic group.

    This is mathematically equivalent to the original
    calculate_metrics(), but avoids repeatedly creating an
    object/string array and calling np.unique() on ~924k rows.

    Group/outcome combinations are encoded into 32 integer
    categories (8 groups x 4 outcomes), then counted with
    np.bincount().
    """

    predictions = (scores >= threshold).astype(np.int8)

    # Outcome encoding is identical to:
    # 0 = TN
    # 1 = FP
    # 2 = FN
    # 3 = TP
    #
    # label*2 + prediction gives exactly this mapping.
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
        tp,
        positives,
        out=np.full(len(group_names), np.nan, dtype=float),
        where=positives > 0
    )

    fnr = np.divide(
        fn,
        positives,
        out=np.full(len(group_names), np.nan, dtype=float),
        where=positives > 0
    )

    fpr = np.divide(
        fp,
        negatives,
        out=np.full(len(group_names), np.nan, dtype=float),
        where=negatives > 0
    )

    return pd.DataFrame({
        "threshold": threshold,
        "group": group_names,
        "TPR": tpr,
        "FNR": fnr,
        "FPR": fpr
    })


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

    The rankings are aligned by group and Kendall's tau is
    calculated on the corresponding rank positions.
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


def save_checkpoints(raw_results, transition_results, raw_path, transition_path):
    """
    Save the current accumulated results.

    This does not alter the experiment or RNG sequence; it only
    writes the results accumulated so far to disk.
    """

    pd.DataFrame(raw_results).to_csv(
        raw_path,
        index=False
    )

    pd.DataFrame(transition_results).to_csv(
        transition_path,
        index=False
    )


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

# Convert once to compact NumPy arrays.
labels = df["label"].to_numpy(dtype=np.int8)
scores = df[MODEL].to_numpy(dtype=np.float64)

# np.unique() is performed ONLY ONCE on the original 923k rows.
# It returns sorted group names, matching sorted(np.unique(groups))
# from the original implementation.
group_names, group_codes = np.unique(
    df["a1"].to_numpy(),
    return_inverse=True
)

group_codes = group_codes.astype(np.int8, copy=False)

n = len(df)

# The DataFrame is no longer needed.
del df
gc.collect()

print(f"Dataset size: {n:,}")
print(f"Model: {MODEL}")
print(f"Groups: {list(group_names)}")
print(f"Thresholds: {THRESHOLDS}")
print(f"Bootstrap repetitions: {BOOTSTRAP_REPETITIONS}")
print(f"Random seed: {RANDOM_SEED}")


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

raw_path = OUTPUT_DIR / "paired_threshold_bootstrap_raw.csv"

transition_path = (
    OUTPUT_DIR / "paired_threshold_bootstrap_transitions.csv"
)

summary_path = OUTPUT_DIR / "paired_threshold_bootstrap_summary.csv"


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
    #
    # The RNG call is intentionally identical to the original
    # script so the bootstrap samples remain the same for the
    # same seed.
    # --------------------------------------------------------

    sample_indices = rng.integers(
        0,
        n,
        size=n
    )

    boot_labels = labels[sample_indices]
    boot_scores = scores[sample_indices]
    boot_group_codes = group_codes[sample_indices]

    threshold_results = {}

    # --------------------------------------------------------
    # Evaluate every threshold on the SAME bootstrap sample
    # --------------------------------------------------------

    for threshold in THRESHOLDS:

        metrics = calculate_metrics_fast(
            boot_labels,
            boot_scores,
            boot_group_codes,
            group_names,
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

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    completed = bootstrap_id + 1

    if completed % CHECKPOINT_EVERY == 0:
        print(f"  Saving checkpoint at {completed}/{BOOTSTRAP_REPETITIONS}...")

        save_checkpoints(
            raw_results,
            transition_results,
            raw_path,
            transition_path
        )

    # Explicitly release large per-bootstrap arrays.
    del sample_indices
    del boot_labels
    del boot_scores
    del boot_group_codes
    del threshold_results

    # Do not run gc.collect() every iteration; that would slow
    # the experiment substantially. Checkpoints are enough.
    if completed % CHECKPOINT_EVERY == 0:
        gc.collect()


# ============================================================
# SAVE FINAL RAW RESULTS
# ============================================================

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
