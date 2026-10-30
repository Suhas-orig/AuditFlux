import pandas as pd
from config import BFW_DATA_PATH, OUTPUT_DIR

# -----------------------------
# Configuration
# -----------------------------

MODELS = ["vgg16", "resnet50", "senet50"]
THRESHOLD = 0.5

# Save each model's baseline results separately so existing ResNet50 outputs
# are not accidentally overwritten.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(BFW_DATA_PATH)

required_columns = {"label", "a1", *MODELS}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(
        "The BFW dataset is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

if df["label"].isna().any() or df["a1"].isna().any():
    raise ValueError("The 'label' and 'a1' columns must not contain missing values.")

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Expected 'label' to contain only binary values 0 and 1.")


# -----------------------------
# Calculate baseline fairness metrics for each model
# -----------------------------

all_results = []

for model in MODELS:
    scores = pd.to_numeric(df[model], errors="coerce")

    if scores.isna().any():
        raise ValueError(
            f"Model column '{model}' contains missing or non-numeric scores."
        )

    # Preserve the original experiment's threshold rule:
    # score >= threshold means predicted same identity (label 1).
    predictions = (scores >= THRESHOLD).astype(int)

    model_results = []

    for group in sorted(df["a1"].unique()):
        group_mask = df["a1"] == group
        group_labels = df.loc[group_mask, "label"]
        group_predictions = predictions.loc[group_mask]

        true_positive = ((group_labels == 1) & (group_predictions == 1)).sum()
        false_negative = ((group_labels == 1) & (group_predictions == 0)).sum()
        false_positive = ((group_labels == 0) & (group_predictions == 1)).sum()
        true_negative = ((group_labels == 0) & (group_predictions == 0)).sum()

        positive_total = true_positive + false_negative
        negative_total = false_positive + true_negative

        # Guard against undefined rates if a group has no examples of a class.
        tpr = true_positive / positive_total if positive_total else float("nan")
        fpr = false_positive / negative_total if negative_total else float("nan")
        fnr = false_negative / positive_total if positive_total else float("nan")

        model_results.append({
            "model": model,
            "group": group,
            "TPR": tpr,
            "FPR": fpr,
            "FNR": fnr,
            "samples": int(group_mask.sum()),
            "positive_pairs": int(positive_total),
            "negative_pairs": int(negative_total),
            "TP": int(true_positive),
            "FN": int(false_negative),
            "FP": int(false_positive),
            "TN": int(true_negative),
            "threshold": THRESHOLD,
        })

    model_results_df = pd.DataFrame(model_results)
    all_results.extend(model_results)

    print(f"\n=== BASELINE FAIRNESS AUDIT: {model.upper()} ===")
    print(f"Threshold: {THRESHOLD}")
    print(model_results_df[
        ["group", "TPR", "FPR", "FNR", "samples"]
    ].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print("\n=== RANGE ACROSS GROUPS ===")
    for metric in ["TPR", "FPR", "FNR"]:
        print(
            f"{metric} range: "
            f"{model_results_df[metric].min():.4f} - "
            f"{model_results_df[metric].max():.4f}"
        )

    output_path = OUTPUT_DIR / f"baseline_audit_{model}.csv"
    model_results_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

# Combined file is convenient for cross-model comparisons.
combined_df = pd.DataFrame(all_results)
combined_path = OUTPUT_DIR / "baseline_audit_all_models.csv"
combined_df.to_csv(combined_path, index=False)
print(f"\nSaved combined results: {combined_path}")
