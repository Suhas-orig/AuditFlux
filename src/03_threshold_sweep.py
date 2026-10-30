import pandas as pd
from config import BFW_DATA_PATH, OUTPUT_DIR

# -----------------------------
# Configuration
# -----------------------------

MODELS = ["vgg16", "resnet50", "senet50"]
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load and validate data
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
# Evaluate each model and threshold
# -----------------------------

all_results = []

for model in MODELS:
    scores = pd.to_numeric(df[model], errors="coerce")
    if scores.isna().any():
        raise ValueError(
            f"Model column '{model}' contains missing or non-numeric scores."
        )

    model_results = []

    for threshold in THRESHOLDS:
        predictions = (scores >= threshold).astype(int)

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

            tpr = true_positive / positive_total if positive_total else float("nan")
            fpr = false_positive / negative_total if negative_total else float("nan")
            fnr = false_negative / positive_total if positive_total else float("nan")

            model_results.append({
                "model": model,
                "threshold": threshold,
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
            })

    model_results_df = pd.DataFrame(model_results)
    all_results.extend(model_results)

    # Print metric tables for this model, preserving the original output style.
    print(f"\\n=== THRESHOLD SWEEP: {model.upper()} ===")
    print("Thresholds:", THRESHOLDS)

    for metric in ["FNR", "FPR", "TPR"]:
        print(f"\\n=== {metric} BY GROUP ===")
        metric_table = model_results_df.pivot(
            index="threshold",
            columns="group",
            values=metric,
        )
        print(metric_table.round(4).to_string())

    model_output_path = OUTPUT_DIR / f"threshold_sweep_{model}.csv"
    model_results_df.to_csv(model_output_path, index=False)
    print(f"\\nResults saved to: {model_output_path}")

# Combined long-form output makes cross-model comparisons easier.
combined_df = pd.DataFrame(all_results)
combined_path = OUTPUT_DIR / "threshold_sweep_all_models.csv"
combined_df.to_csv(combined_path, index=False)
print(f"\\nCombined results saved to: {combined_path}")