import pandas as pd
from scipy.stats import kendalltau
from config import OUTPUT_DIR

# ============================================================
# CONFIGURATION
# ============================================================

MODELS = ["vgg16", "resnet50", "senet50"]
METRIC = "FNR"  # Higher FNR = worse; rank 1 is the worst-performing group.

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RESULTS, RANK GROUPS, AND COMPARE THRESHOLDS
# ============================================================

all_rankings = []
all_transitions = []

for model in MODELS:
    input_path = OUTPUT_DIR / f"threshold_sweep_{model}.csv"
    df = pd.read_csv(input_path)

    required_columns = {"threshold", "group", METRIC}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"{input_path.name} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    rankings = {}

    # Higher FNR is worse, so sort descending: rank 1 is the worst group.
    for threshold in sorted(df["threshold"].unique()):
        threshold_df = df[df["threshold"] == threshold].copy()
        threshold_df = threshold_df.sort_values(
            by=METRIC,
            ascending=False,
            kind="mergesort",
        )
        rankings[threshold] = threshold_df["group"].tolist()

        for rank, row in enumerate(
            threshold_df[["group", METRIC]].itertuples(index=False, name=None),
            start=1,
        ):
            group, metric_value = row
            all_rankings.append({
                "model": model,
                "metric": METRIC,
                "threshold": threshold,
                "group": group,
                "rank": rank,
                "metric_value": metric_value,
            })

    print(f"\n=== GROUP RANKINGS: {model.upper()} ===")
    print(f"Metric: {METRIC} (rank 1 = highest/worst {METRIC})")
    for threshold, ranking in rankings.items():
        print(f"\nThreshold {threshold:.2f}")
        print(" > ".join(ranking))

    print(f"\n=== RANKING CHANGES: {model.upper()} ===")
    thresholds = sorted(rankings.keys())

    for i in range(1, len(thresholds)):
        previous_threshold = thresholds[i - 1]
        current_threshold = thresholds[i]

        previous_ranking = rankings[previous_threshold]
        current_ranking = rankings[current_threshold]

        previous_positions = {
            group: rank + 1 for rank, group in enumerate(previous_ranking)
        }
        current_positions = {
            group: rank + 1 for rank, group in enumerate(current_ranking)
        }

        changed_groups = []
        for group in previous_positions:
            old_rank = previous_positions[group]
            new_rank = current_positions[group]
            if old_rank != new_rank:
                changed_groups.append((group, old_rank, new_rank))

        print(f"\n{previous_threshold:.2f} -> {current_threshold:.2f}")
        if not changed_groups:
            print("No ranking changes.")
        else:
            for group, old_rank, new_rank in changed_groups:
                print(f"{group}: rank {old_rank} -> {new_rank}")

        # Kendall tau measures similarity between the two full rankings.
        previous_position_values = {
            group: rank for rank, group in enumerate(previous_ranking)
        }
        current_position_values = {
            group: rank for rank, group in enumerate(current_ranking)
        }
        groups = previous_ranking
        previous_values = [previous_position_values[group] for group in groups]
        current_values = [current_position_values[group] for group in groups]
        tau, p_value = kendalltau(previous_values, current_values)

        print(f"Kendall tau: {tau:.4f} | p = {p_value:.4g}")

        all_transitions.append({
            "model": model,
            "metric": METRIC,
            "from_threshold": previous_threshold,
            "to_threshold": current_threshold,
            "kendall_tau": tau,
            "p_value": p_value,
            "changed_group_count": len(changed_groups),
            "worst_group_before": previous_ranking[0],
            "worst_group_after": current_ranking[0],
            "worst_group_changed": previous_ranking[0] != current_ranking[0],
        })


# ============================================================
# SAVE RESULTS
# ============================================================

ranking_df = pd.DataFrame(all_rankings)
ranking_output_path = OUTPUT_DIR / "ranking_stability_fnr_all_models.csv"
ranking_df.to_csv(ranking_output_path, index=False)

transitions_df = pd.DataFrame(all_transitions)
transitions_output_path = (
    OUTPUT_DIR / "ranking_stability_fnr_transitions_all_models.csv"
)
transitions_df.to_csv(transitions_output_path, index=False)

print("\n=== SAVED ===")
print(ranking_output_path)
print(transitions_output_path)