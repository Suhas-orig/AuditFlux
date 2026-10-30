import pandas as pd
from scipy.stats import kendalltau
from config import OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = ["vgg16", "resnet50", "senet50"]
METRICS = ["FNR", "FPR", "TPR"]

# Existing ResNet50 output is preserved. This script writes new
# all-model output files rather than overwriting metric_comparison.csv.
WORST_GROUP_OUTPUT = OUTPUT_DIR / "metric_comparison_all_models.csv"
SIMILARITY_OUTPUT = OUTPUT_DIR / "metric_ranking_similarity_all_models.csv"
FULL_RANKINGS_OUTPUT = OUTPUT_DIR / "metric_full_rankings_all_models.csv"


# ============================================================
# LOAD AND VALIDATE EACH MODEL'S THRESHOLD-SWEEP RESULTS
# ============================================================

worst_rows = []
similarity_rows = []
full_ranking_rows = []

for model in MODELS:
    input_path = OUTPUT_DIR / f"threshold_sweep_{model}.csv"
    print(f"\\n\\n{'=' * 70}")
    print(f"MODEL: {model.upper()}")
    print(f"Input: {input_path}")
    print("=" * 70)

    df = pd.read_csv(input_path)

    required_columns = {"threshold", "group", *METRICS}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{input_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    rankings = {}

    for threshold in sorted(df["threshold"].unique()):
        threshold_df = df[df["threshold"] == threshold].copy()
        rankings[threshold] = {}

        for metric in METRICS:
            # FNR/FPR: higher = worse; TPR: lower = worse.
            ascending = (metric == "TPR")
            sorted_df = threshold_df.sort_values(
                by=metric,
                ascending=ascending,
                kind="mergesort"
            )
            ranking = sorted_df["group"].tolist()
            rankings[threshold][metric] = ranking

            for rank, (_, row) in enumerate(sorted_df.iterrows(), start=1):
                full_ranking_rows.append({
                    "model": model,
                    "threshold": threshold,
                    "metric": metric,
                    "group": row["group"],
                    "rank": rank,
                    "metric_value": row[metric],
                })

            worst_group = ranking[0]
            worst_value = sorted_df.iloc[0][metric]
            worst_rows.append({
                "model": model,
                "threshold": threshold,
                "metric": metric,
                "worst_group": worst_group,
                "worst_metric_value": worst_value,
            })

    # Print worst groups and full rankings for each threshold.
    print("\\n=== WORST GROUP BY METRIC ===")
    for threshold in sorted(rankings):
        print(f"\\nThreshold {threshold:.2f}")
        for metric in METRICS:
            print(f"{metric}: {rankings[threshold][metric][0]}")

    print("\\n=== FULL GROUP RANKINGS ===")
    for threshold in sorted(rankings):
        print(f"\\nThreshold {threshold:.2f}")
        for metric in METRICS:
            print(f"{metric}: {' > '.join(rankings[threshold][metric])}")

    # Compare rankings between metrics at the same threshold.
    print("\\n=== METRIC RANKING SIMILARITY ===")
    for threshold in sorted(rankings):
        print(f"\\nThreshold {threshold:.2f}")

        for i in range(len(METRICS)):
            for j in range(i + 1, len(METRICS)):
                metric_a = METRICS[i]
                metric_b = METRICS[j]
                ranking_a = rankings[threshold][metric_a]
                ranking_b = rankings[threshold][metric_b]

                positions_a = {
                    group: rank for rank, group in enumerate(ranking_a)
                }
                positions_b = {
                    group: rank for rank, group in enumerate(ranking_b)
                }
                groups = ranking_a
                values_a = [positions_a[group] for group in groups]
                values_b = [positions_b[group] for group in groups]

                tau, p_value = kendalltau(values_a, values_b)

                similarity_rows.append({
                    "model": model,
                    "threshold": threshold,
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "kendall_tau": tau,
                    "p_value": p_value,
                })

                print(
                    f"{metric_a} vs {metric_b}"
                    f" | Kendall tau = {tau:.4f}"
                    f" | p = {p_value:.4g}"
                )


# ============================================================
# SAVE NEW ALL-MODEL OUTPUTS
# ============================================================

worst_df = pd.DataFrame(worst_rows)
similarity_df = pd.DataFrame(similarity_rows)
full_rankings_df = pd.DataFrame(full_ranking_rows)

worst_df.to_csv(WORST_GROUP_OUTPUT, index=False)
similarity_df.to_csv(SIMILARITY_OUTPUT, index=False)
full_rankings_df.to_csv(FULL_RANKINGS_OUTPUT, index=False)

print("\\n\\n=== SAVED ===")
print(WORST_GROUP_OUTPUT)
print(SIMILARITY_OUTPUT)
print(FULL_RANKINGS_OUTPUT)

print("\\nNote: FNR and TPR are complements on the same positive pairs, so their")
print("rankings should be exact reversals when there are no ties. Interpret")
print("their comparison accordingly; it is not independent evidence.")
