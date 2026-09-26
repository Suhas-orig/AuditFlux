import pandas as pd

# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = r"K:\Capstone Project\bfw-datatable.csv"

MODEL = "resnet50"

# Thresholds to test
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(DATA_PATH)


# -----------------------------
# Evaluate each threshold
# -----------------------------

results = []

for threshold in THRESHOLDS:

    df["prediction"] = (df[MODEL] >= threshold).astype(int)

    for group in sorted(df["a1"].unique()):

        group_df = df[df["a1"] == group]

        true_positive = (
            (group_df["label"] == 1) &
            (group_df["prediction"] == 1)
        ).sum()

        false_negative = (
            (group_df["label"] == 1) &
            (group_df["prediction"] == 0)
        ).sum()

        false_positive = (
            (group_df["label"] == 0) &
            (group_df["prediction"] == 1)
        ).sum()

        true_negative = (
            (group_df["label"] == 0) &
            (group_df["prediction"] == 0)
        ).sum()

        tpr = true_positive / (true_positive + false_negative)
        fpr = false_positive / (false_positive + true_negative)
        fnr = false_negative / (true_positive + false_negative)

        results.append({
            "threshold": threshold,
            "group": group,
            "TPR": tpr,
            "FPR": fpr,
            "FNR": fnr
        })


# -----------------------------
# Create results table
# -----------------------------

results_df = pd.DataFrame(results)


# -----------------------------
# Display results
# -----------------------------

print("\n=== THRESHOLD SWEEP ===")
print(f"Model: {MODEL}")

print("\n=== FNR BY GROUP ===")

fnr_table = results_df.pivot(
    index="threshold",
    columns="group",
    values="FNR"
)

print(fnr_table.round(4).to_string())

print("\n=== FPR BY GROUP ===")

fpr_table = results_df.pivot(
    index="threshold",
    columns="group",
    values="FPR"
)

print(fpr_table.round(4).to_string())

print("\n=== TPR BY GROUP ===")

tpr_table = results_df.pivot(
    index="threshold",
    columns="group",
    values="TPR"
)

print(tpr_table.round(4).to_string())

# -----------------------------
# Save results
# -----------------------------

results_df.to_csv(
    "outputs/threshold_sweep_resnet50.csv",
    index=False
)

print("\nResults saved to:")
print("outputs/threshold_sweep_resnet50.csv")