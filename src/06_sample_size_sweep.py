import pandas as pd
import numpy as np
from config import BFW_DATA_PATH, OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = ["vgg16", "resnet50", "senet50"]
THRESHOLD = 0.50
SAMPLE_SIZES = [100000, 50000, 20000, 10000, 5000]
REPETITIONS = 20
GROUP_COLUMN = "a1"
RANDOM_SEED = 42


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

df = pd.read_csv(BFW_DATA_PATH)

required_columns = {
    GROUP_COLUMN, "label", *MODELS
}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns in BFW data: {sorted(missing_columns)}"
    )

groups = sorted(df[GROUP_COLUMN].dropna().unique())
rng = np.random.default_rng(RANDOM_SEED)

# Preserve the original subgroup-proportion sampling design.
proportions = (
    df[GROUP_COLUMN]
    .value_counts(normalize=True)
    .sort_index()
)

results = []


def calculate_metrics(sample, model, sample_size, repetition):
    """Calculate group-level classification metrics for one sampled dataset."""
    predictions = (sample[model] >= THRESHOLD).astype(int)

    for group in groups:
        group_mask = sample[GROUP_COLUMN] == group
        group_labels = sample.loc[group_mask, "label"]
        group_predictions = predictions.loc[group_mask]

        tp = ((group_labels == 1) & (group_predictions == 1)).sum()
        fn = ((group_labels == 1) & (group_predictions == 0)).sum()
        fp = ((group_labels == 0) & (group_predictions == 1)).sum()
        tn = ((group_labels == 0) & (group_predictions == 0)).sum()

        positive_count = tp + fn
        negative_count = fp + tn

        # Guard against undefined rates if a sampled subgroup has no
        # positive or negative examples.
        tpr = tp / positive_count if positive_count else np.nan
        fnr = fn / positive_count if positive_count else np.nan
        fpr = fp / negative_count if negative_count else np.nan

        results.append({
            "model": model,
            "threshold": THRESHOLD,
            "sample_size": sample_size,
            "repetition": repetition,
            "group": group,
            "group_sample_count": int(group_mask.sum()),
            "positive_pairs": int(positive_count),
            "negative_pairs": int(negative_count),
            "TPR": tpr,
            "FNR": fnr,
            "FPR": fpr
        })


# ============================================================
# SAMPLE EACH SIZE REPEATEDLY FOR EACH MODEL
# ============================================================

for sample_size in SAMPLE_SIZES:
    # Allocate approximately equal proportions to the original group
    # composition; correct rounding to keep the requested total exact.
    counts = (proportions * sample_size).round().astype(int)
    difference = sample_size - int(counts.sum())

    if difference != 0:
        largest_group = counts.idxmax()
        counts.loc[largest_group] += difference

    # Validate requested sample size against available data per group.
    insufficient = {
        group: int(counts[group])
        for group in groups
        if counts[group] > int((df[GROUP_COLUMN] == group).sum())
    }
    if insufficient:
        raise ValueError(
            f"Cannot draw sample_size={sample_size} without replacement; "
            f"requested counts exceed available rows: {insufficient}"
        )

    for model in MODELS:
        print(f"\nModel: {model} | sample size: {sample_size}")

        for repetition in range(1, REPETITIONS + 1):
            sampled_parts = []

            for group in groups:
                group_df = df[df[GROUP_COLUMN] == group]
                sampled_group = group_df.sample(
                    n=int(counts[group]),
                    replace=False,
                    random_state=int(rng.integers(0, 1_000_000_000))
                )
                sampled_parts.append(sampled_group)

            sample = pd.concat(sampled_parts, ignore_index=True)
            calculate_metrics(sample, model, sample_size, repetition)

            print(f"  repetition {repetition}/{REPETITIONS}")

# ============================================================
# SAVE OUTPUTS
# ============================================================

results_df = pd.DataFrame(results)

combined_path = OUTPUT_DIR / "sample_size_sweep_all_models.csv"
results_df.to_csv(combined_path, index=False)

# Also write separate per-model files, leaving the original ResNet50
# output filename available for compatibility with existing workflows.
for model in MODELS:
    model_path = OUTPUT_DIR / f"sample_size_sweep_{model}.csv"
    results_df[results_df["model"] == model].to_csv(model_path, index=False)

print("\nDONE")
print(f"Combined results saved to: {combined_path}")
for model in MODELS:
    print(f"{model} results saved to: {OUTPUT_DIR / f'sample_size_sweep_{model}.csv'}")
