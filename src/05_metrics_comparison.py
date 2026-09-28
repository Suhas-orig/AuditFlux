import pandas as pd
from scipy.stats import kendalltau
from config import OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = OUTPUT_DIR / "threshold_sweep_resnet50.csv"

METRICS = ["FNR", "FPR", "TPR"]


# ============================================================
# LOAD THRESHOLD-SWEEP RESULTS
# ============================================================

df = pd.read_csv(INPUT_PATH)


# ============================================================
# STORE RANKINGS
# ============================================================

rankings = {}

for threshold in sorted(df["threshold"].unique()):

    threshold_df = df[df["threshold"] == threshold].copy()

    rankings[threshold] = {}

    for metric in METRICS:

        # FNR/FPR: higher = worse
        # TPR: lower = worse

        if metric == "TPR":

            sorted_df = threshold_df.sort_values(
                by=metric,
                ascending=True
            )

        else:

            sorted_df = threshold_df.sort_values(
                by=metric,
                ascending=False
            )

        rankings[threshold][metric] = (
            sorted_df["group"].tolist()
        )


# ============================================================
# PRINT WORST GROUP FOR EACH METRIC
# ============================================================

print("\n=== WORST GROUP BY METRIC ===")

for threshold in sorted(rankings.keys()):

    print(f"\nThreshold {threshold:.2f}")

    for metric in METRICS:

        worst_group = rankings[threshold][metric][0]

        print(
            f"{metric}: {worst_group}"
        )


# ============================================================
# PRINT FULL RANKINGS
# ============================================================

print("\n\n=== FULL GROUP RANKINGS ===")

for threshold in sorted(rankings.keys()):

    print(f"\nThreshold {threshold:.2f}")

    for metric in METRICS:

        ranking = rankings[threshold][metric]

        ranking_text = " > ".join(ranking)

        print(
            f"{metric}: {ranking_text}"
        )


# ============================================================
# COMPARE METRIC RANKINGS
# ============================================================

print("\n\n=== METRIC RANKING SIMILARITY ===")

for threshold in sorted(rankings.keys()):

    print(f"\nThreshold {threshold:.2f}")

    for i in range(len(METRICS)):

        for j in range(i + 1, len(METRICS)):

            metric_a = METRICS[i]
            metric_b = METRICS[j]

            ranking_a = rankings[threshold][metric_a]
            ranking_b = rankings[threshold][metric_b]

            # Convert rankings into numerical positions

            positions_a = {
                group: rank
                for rank, group in enumerate(ranking_a)
            }

            positions_b = {
                group: rank
                for rank, group in enumerate(ranking_b)
            }

            groups = ranking_a

            values_a = [
                positions_a[group]
                for group in groups
            ]

            values_b = [
                positions_b[group]
                for group in groups
            ]

            tau, p_value = kendalltau(
                values_a,
                values_b
            )

            print(
                f"{metric_a} vs {metric_b}"
                f" | Kendall tau = {tau:.4f}"
                f" | p = {p_value:.4g}"
            )


# ============================================================
# SAVE WORST-GROUP RESULTS
# ============================================================

worst_rows = []

for threshold in sorted(rankings.keys()):

    for metric in METRICS:

        worst_group = rankings[threshold][metric][0]

        worst_rows.append({
            "threshold": threshold,
            "metric": metric,
            "worst_group": worst_group
        })


worst_df = pd.DataFrame(worst_rows)

OUTPUT_PATH = OUTPUT_DIR / "metric_comparison.csv"

worst_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n\n=== SAVED ===")
print(OUTPUT_PATH)