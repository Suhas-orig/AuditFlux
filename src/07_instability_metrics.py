import pandas as pd
import numpy as np
from scipy.stats import kendalltau

THRESHOLD_FILE = r"K:\Capstone Project\AuditFlux\outputs\threshold_sweep_resnet50.csv"
SAMPLE_FILE = r"K:\Capstone Project\AuditFlux\outputs\sample_size_sweep_resnet50.csv"

THRESHOLD_OUTPUT = r"K:\Capstone Project\AuditFlux\outputs\threshold_instability_metrics.csv"
SAMPLE_OUTPUT = r"K:\Capstone Project\AuditFlux\outputs\sample_instability_metrics.csv"

METRICS = ["FNR", "FPR"]


# =========================================================
# Helper: get ranking
# =========================================================

def get_ranking(data, metric):
    """
    Higher FNR/FPR = worse.
    Returns groups from worst -> best.
    """
    return (
        data.groupby("group")[metric]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


# =========================================================
# THRESHOLD INSTABILITY
# =========================================================

threshold_df = pd.read_csv(THRESHOLD_FILE)

threshold_results = []

for metric in METRICS:

    thresholds = sorted(threshold_df["threshold"].unique())

    previous_worst = None
    previous_ranking = None
    previous_gap = None

    for threshold in thresholds:

        current = threshold_df[
            threshold_df["threshold"] == threshold
        ]

        ranking = get_ranking(current, metric)

        worst_group = ranking[0]

        # Magnitude of disparity
        values = current[metric]
        gap = values.max() - values.min()

        # Compare with previous threshold
        worst_reversal = (
            previous_worst is not None
            and worst_group != previous_worst
        )

        if previous_ranking is not None:
            tau, _ = kendalltau(
                previous_ranking,
                ranking
            )

            ranking_instability = 1 - tau
        else:
            tau = np.nan
            ranking_instability = np.nan

        if previous_gap is not None:
            magnitude_change = abs(gap - previous_gap)
        else:
            magnitude_change = np.nan

        threshold_results.append({
            "metric": metric,
            "threshold": threshold,
            "worst_group": worst_group,
            "gap": gap,
            "worst_group_reversal": int(worst_reversal),
            "kendall_tau_previous": tau,
            "ranking_instability": ranking_instability,
            "magnitude_change": magnitude_change
        })

        previous_worst = worst_group
        previous_ranking = ranking
        previous_gap = gap


threshold_results_df = pd.DataFrame(threshold_results)

threshold_results_df.to_csv(
    THRESHOLD_OUTPUT,
    index=False
)


# =========================================================
# SAMPLE-SIZE INSTABILITY
# =========================================================

sample_df = pd.read_csv(SAMPLE_FILE)

sample_results = []

for metric in METRICS:

    for sample_size in sorted(
        sample_df["sample_size"].unique(),
        reverse=True
    ):

        current = sample_df[
            sample_df["sample_size"] == sample_size
        ]

        # Each repetition is one independently sampled audit.
        for repetition, rep_data in current.groupby(
            current.groupby("group").cumcount()
        ):

            ranking = get_ranking(rep_data, metric)

            worst_group = ranking[0]

            values = rep_data[metric]
            gap = values.max() - values.min()

            sample_results.append({
                "metric": metric,
                "sample_size": sample_size,
                "repetition": repetition,
                "worst_group": worst_group,
                "gap": gap
            })


sample_results_df = pd.DataFrame(sample_results)

sample_results_df.to_csv(
    SAMPLE_OUTPUT,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("THRESHOLD INSTABILITY")
print("=" * 60)

print(threshold_results_df.to_string(index=False))


print("\n" + "=" * 60)
print("WORST-GROUP COUNTS BY SAMPLE SIZE")
print("=" * 60)

for metric in METRICS:

    print(f"\n{metric}")

    summary = pd.crosstab(
        sample_results_df[
            sample_results_df["metric"] == metric
        ]["sample_size"],
        sample_results_df[
            sample_results_df["metric"] == metric
        ]["worst_group"]
    )

    print(summary)


print("\n" + "=" * 60)
print("AVERAGE GAP BY SAMPLE SIZE")
print("=" * 60)

gap_summary = (
    sample_results_df
    .groupby(["metric", "sample_size"])["gap"]
    .agg(["mean", "std"])
    .reset_index()
)

print(gap_summary.to_string(index=False))


print("\nDONE")