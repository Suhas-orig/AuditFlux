import pandas as pd

# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = r"K:\Capstone Project\bfw-datatable.csv"

MODEL = "resnet50"

# Initial threshold.
# We will make this data-driven in the next version.
THRESHOLD = 0.5


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(DATA_PATH)


# -----------------------------
# Make predictions
# -----------------------------

df["prediction"] = (df[MODEL] >= THRESHOLD).astype(int)


# -----------------------------
# Calculate metrics
# -----------------------------

results = []

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
        "group": group,
        "TPR": tpr,
        "FPR": fpr,
        "FNR": fnr,
        "samples": len(group_df)
    })


# -----------------------------
# Display results
# -----------------------------

results_df = pd.DataFrame(results)

print("\n=== BASELINE FAIRNESS AUDIT ===")
print(f"Model: {MODEL}")
print(f"Threshold: {THRESHOLD}")

print("\n")
print(results_df.to_string(index=False))

print("\n=== RANGE ACROSS GROUPS ===")

print(
    f"TPR range: "
    f"{results_df['TPR'].min():.4f} - "
    f"{results_df['TPR'].max():.4f}"
)

print(
    f"FPR range: "
    f"{results_df['FPR'].min():.4f} - "
    f"{results_df['FPR'].max():.4f}"
)

print(
    f"FNR range: "
    f"{results_df['FNR'].min():.4f} - "
    f"{results_df['FNR'].max():.4f}"
)