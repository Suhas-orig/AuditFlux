import pandas as pd
from scipy.stats import kendalltau
from config import OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = OUTPUT_DIR / "threshold_sweep_resnet50.csv"
METRIC = "FNR"


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(INPUT_PATH)


# ============================================================
# CREATE RANKINGS
# Higher FNR = worse = rank 1
# ============================================================

rankings = {}

for threshold in sorted(df["threshold"].unique()):

    threshold_df = df[df["threshold"] == threshold].copy()

    threshold_df = threshold_df.sort_values(
        by=METRIC,
        ascending=False
    )

    rankings[threshold] = threshold_df["group"].tolist()


# ============================================================
# PRINT RANKING AT EACH THRESHOLD
# ============================================================

print("\n=== GROUP RANKINGS ===")
print(f"Metric: {METRIC}")

for threshold, ranking in rankings.items():

    ranking_text = " > ".join(ranking)

    print(f"\nThreshold {threshold:.2f}")
    print(ranking_text)


# ============================================================
# COMPARE CONSECUTIVE THRESHOLDS
# ============================================================

print("\n\n=== RANKING CHANGES ===")

thresholds = sorted(rankings.keys())

for i in range(1, len(thresholds)):

    previous_threshold = thresholds[i - 1]
    current_threshold = thresholds[i]

    previous_ranking = rankings[previous_threshold]
    current_ranking = rankings[current_threshold]

    previous_positions = {
        group: rank + 1
        for rank, group in enumerate(previous_ranking)
    }

    current_positions = {
        group: rank + 1
        for rank, group in enumerate(current_ranking)
    }

    changed_groups = []

    for group in previous_positions:

        old_rank = previous_positions[group]
        new_rank = current_positions[group]

        if old_rank != new_rank:
            changed_groups.append(
                (group, old_rank, new_rank)
            )

    print(
        f"\n{previous_threshold:.2f} → {current_threshold:.2f}"
    )

    if not changed_groups:
        print("No ranking changes.")

    else:
        for group, old_rank, new_rank in changed_groups:
            print(
                f"{group}: rank {old_rank} → {new_rank}"
            )


# ============================================================
# KENDALL'S TAU BETWEEN CONSECUTIVE THRESHOLDS
# ============================================================

print("\n\n=== KENDALL RANKING SIMILARITY ===")

for i in range(1, len(thresholds)):

    previous_threshold = thresholds[i - 1]
    current_threshold = thresholds[i]

    previous_ranking = rankings[previous_threshold]
    current_ranking = rankings[current_threshold]

    previous_positions = {
        group: rank
        for rank, group in enumerate(previous_ranking)
    }

    current_positions = {
        group: rank
        for rank, group in enumerate(current_ranking)
    }

    groups = previous_ranking

    previous_values = [
        previous_positions[group]
        for group in groups
    ]

    current_values = [
        current_positions[group]
        for group in groups
    ]

    tau, p_value = kendalltau(
        previous_values,
        current_values
    )

    print(
        f"{previous_threshold:.2f} → {current_threshold:.2f} "
        f"| Kendall tau = {tau:.4f} "
        f"| p = {p_value:.4g}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

ranking_rows = []

for threshold, ranking in rankings.items():

    for rank, group in enumerate(ranking, start=1):

        ranking_rows.append({
            "threshold": threshold,
            "group": group,
            "rank": rank
        })


ranking_df = pd.DataFrame(ranking_rows)

OUTPUT_PATH = OUTPUT_DIR / "ranking_stability_fnr.csv"

ranking_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n\n=== SAVED ===")
print(OUTPUT_PATH)