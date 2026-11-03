import pandas as pd
import numpy as np

from config import BFW_DATA_PATH, OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = ["vgg16", "resnet50", "senet50"]
THRESHOLD = 0.50
SAMPLE_SIZES = [100000, 50000, 20000, 10000, 5000]
REPETITIONS = 500
GROUP_COLUMN = "a1"
RANDOM_SEED = 42

# Confidence intervals are descriptive percentile intervals across
# repeated comparison-level samples. They are not identity-clustered
# confidence intervals.
CI_LOWER = 2.5
CI_UPPER = 97.5


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

df = pd.read_csv(BFW_DATA_PATH)

required_columns = {GROUP_COLUMN, "label", *MODELS}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns in BFW data: {sorted(missing_columns)}"
    )

if df[GROUP_COLUMN].isna().any():
    raise ValueError(f"Missing values found in subgroup column '{GROUP_COLUMN}'.")

if df["label"].isna().any():
    raise ValueError("Missing values found in 'label'.")

for model in MODELS:
    if df[model].isna().any():
        raise ValueError(f"Missing values found in model score column '{model}'.")
    if not pd.api.types.is_numeric_dtype(df[model]):
        raise TypeError(f"Model score column '{model}' must be numeric.")

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("'label' must contain only binary values 0 and 1.")

groups = sorted(df[GROUP_COLUMN].unique())

# Cache each subgroup once; avoids repeatedly filtering the full dataset.
group_frames = {
    group: df.loc[df[GROUP_COLUMN] == group]
    for group in groups
}

group_proportions = (
    df[GROUP_COLUMN]
    .value_counts(normalize=True)
    .reindex(groups)
)

rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(sample, model, sample_size, repetition):
    """Calculate subgroup-level TPR, FNR and FPR for one model."""
    predictions = (sample[model] >= THRESHOLD).astype(np.int8)

    rows = []

    for group in groups:
        group_mask = sample[GROUP_COLUMN].to_numpy() == group
        group_labels = sample.loc[group_mask, "label"].to_numpy()
        group_predictions = predictions.loc[group_mask].to_numpy()

        tp = int(((group_labels == 1) & (group_predictions == 1)).sum())
        fn = int(((group_labels == 1) & (group_predictions == 0)).sum())
        fp = int(((group_labels == 0) & (group_predictions == 1)).sum())
        tn = int(((group_labels == 0) & (group_predictions == 0)).sum())

        positive_count = tp + fn
        negative_count = fp + tn

        tpr = tp / positive_count if positive_count else np.nan
        fnr = fn / positive_count if positive_count else np.nan
        fpr = fp / negative_count if negative_count else np.nan

        rows.append({
            "model": model,
            "threshold": THRESHOLD,
            "sample_size": sample_size,
            "repetition": repetition,
            "group": group,
            "group_sample_count": int(group_mask.sum()),
            "positive_pairs": positive_count,
            "negative_pairs": negative_count,
            "TPR": tpr,
            "FNR": fnr,
            "FPR": fpr,
        })

    return rows


# ============================================================
# REPEATED SUBGROUP-PROPORTIONAL SAMPLING
# ============================================================

results = []

for sample_size in SAMPLE_SIZES:
    # Preserve the original design: allocate the requested total
    # proportionally across a1 groups, without replacement.
    counts = (group_proportions * sample_size).round().astype(int)

    # Correct rounding while keeping the requested total exact.
    difference = sample_size - int(counts.sum())
    if difference != 0:
        largest_group = counts.idxmax()
        counts.loc[largest_group] += difference

    insufficient = {
        group: int(counts[group])
        for group in groups
        if counts[group] > len(group_frames[group])
    }
    if insufficient:
        raise ValueError(
            f"Cannot draw sample_size={sample_size} without replacement; "
            f"requested counts exceed available rows: {insufficient}"
        )

    print(f"\nSample size: {sample_size:,}")

    for repetition in range(1, REPETITIONS + 1):
        sampled_parts = []

        # Draw ONE sample for this repetition. All three models are
        # evaluated on exactly these same comparison rows.
        for group in groups:
            group_df = group_frames[group]
            sampled_group = group_df.sample(
                n=int(counts[group]),
                replace=False,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
            sampled_parts.append(sampled_group)

        sample = pd.concat(sampled_parts, ignore_index=True)

        # Paired model evaluation: identical sampled comparisons,
        # labels and subgroup composition for every model.
        for model in MODELS:
            results.extend(
                calculate_metrics(sample, model, sample_size, repetition)
            )

        if repetition % 25 == 0 or repetition == REPETITIONS:
            print(f"  repetition {repetition}/{REPETITIONS}")


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results_df = pd.DataFrame(results)

combined_path = OUTPUT_DIR / "sample_size_sweep_all_models.csv"
results_df.to_csv(combined_path, index=False)

# Preserve the existing per-model output filenames.
for model in MODELS:
    model_path = OUTPUT_DIR / f"sample_size_sweep_{model}.csv"
    results_df.loc[results_df["model"] == model].to_csv(
        model_path, index=False
    )


# ============================================================
# SUMMARY: MEAN, SD AND PERCENTILE INTERVALS
# ============================================================

metric_columns = ["TPR", "FNR", "FPR"]

summary = (
    results_df
    .groupby(["model", "sample_size", "group"], as_index=False)[metric_columns]
    .agg(["mean", "std", lambda x: x.quantile(CI_LOWER / 100),
          lambda x: x.quantile(CI_UPPER / 100)])
)

# Flatten the multi-level column names into readable output columns.
summary.columns = [
    "_".join(str(part) for part in column if str(part))
    if isinstance(column, tuple) else str(column)
    for column in summary.columns
]
summary = summary.rename(columns={
    "TPR_<lambda_0>": "TPR_ci_lower_2_5",
    "TPR_<lambda_1>": "TPR_ci_upper_97_5",
    "FNR_<lambda_0>": "FNR_ci_lower_2_5",
    "FNR_<lambda_1>": "FNR_ci_upper_97_5",
    "FPR_<lambda_0>": "FPR_ci_lower_2_5",
    "FPR_<lambda_1>": "FPR_ci_upper_97_5",
})

summary_path = OUTPUT_DIR / "sample_size_sweep_summary.csv"
summary.to_csv(summary_path, index=False)


# ============================================================
# WORST-GROUP FREQUENCY
# ============================================================

worst_group_rows = []

for (model, sample_size, repetition), repetition_df in results_df.groupby(
    ["model", "sample_size", "repetition"]
):
    for metric in ["FNR", "FPR"]:
        valid = repetition_df.dropna(subset=[metric])
        if valid.empty:
            continue

        worst_value = valid[metric].max()
        worst_groups = valid.loc[
            np.isclose(valid[metric], worst_value, rtol=1e-10, atol=1e-12),
            "group",
        ].tolist()

        # Record every tied worst group rather than arbitrarily selecting one.
        for group in worst_groups:
            worst_group_rows.append({
                "model": model,
                "sample_size": sample_size,
                "repetition": repetition,
                "metric": metric,
                "worst_group": group,
                "worst_metric_value": worst_value,
                "number_of_tied_worst_groups": len(worst_groups),
            })

worst_df = pd.DataFrame(worst_group_rows)

worst_frequency = (
    worst_df
    .groupby(["model", "sample_size", "metric", "worst_group"], as_index=False)
    .agg(
        worst_group_count=("repetition", "count"),
        mean_worst_metric_value=("worst_metric_value", "mean"),
    )
)

worst_frequency["worst_group_frequency"] = (
    worst_frequency["worst_group_count"] / REPETITIONS
)

worst_path = OUTPUT_DIR / "sample_size_sweep_worst_group_frequency.csv"
worst_frequency.to_csv(worst_path, index=False)


# ============================================================
# COMPLETION
# ============================================================

print("\nDONE")
print(f"Raw combined results: {combined_path}")
print(f"Summary statistics: {summary_path}")
print(f"Worst-group frequencies: {worst_path}")

for model in MODELS:
    print(f"{model}: {OUTPUT_DIR / f'sample_size_sweep_{model}.csv'}")

print(
    "\nMethodological note: samples are paired across models, but rows are "
    "still sampled at the comparison level. These intervals do not account "
    "for dependence induced by repeated subject identities."
)
