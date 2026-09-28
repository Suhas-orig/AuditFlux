import pandas as pd
import numpy as np
from config import BFW_DATA_PATH, OUTPUT_DIR


MODEL = "resnet50"

THRESHOLD = 0.50

SAMPLE_SIZES = [100000, 50000, 20000, 10000, 5000]

REPETITIONS = 20

GROUP_COLUMN = "a1"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(BFW_DATA_PATH)


# We need enough data in every subgroup for every sample size.

groups = sorted(df[GROUP_COLUMN].unique())

rng = np.random.default_rng(42)

results = []


def calculate_metrics(sample):

    sample = sample.copy()

    sample["prediction"] = (
        sample[MODEL] >= THRESHOLD
    ).astype(int)

    for group in groups:

        g = sample[sample[GROUP_COLUMN] == group]

        tp = ((g["label"] == 1) & (g["prediction"] == 1)).sum()
        fn = ((g["label"] == 1) & (g["prediction"] == 0)).sum()
        fp = ((g["label"] == 0) & (g["prediction"] == 1)).sum()
        tn = ((g["label"] == 0) & (g["prediction"] == 0)).sum()

        tpr = tp / (tp + fn)
        fnr = fn / (tp + fn)
        fpr = fp / (fp + tn)

        results.append({
            "sample_size": len(sample),
            "group": group,
            "TPR": tpr,
            "FNR": fnr,
            "FPR": fpr
        })


# ---------------------------------------------------------
# Sample each size repeatedly
# ---------------------------------------------------------

for sample_size in SAMPLE_SIZES:

    print(f"\nSample size: {sample_size}")

    # Allocate approximately equal numbers according
    # to the original subgroup proportions.

    proportions = (
        df[GROUP_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
    )

    counts = (
        proportions * sample_size
    ).round().astype(int)

    # Correct rounding so total is exactly sample_size.

    difference = sample_size - counts.sum()

    if difference != 0:
        largest_group = counts.idxmax()
        counts[largest_group] += difference

    for repetition in range(REPETITIONS):

        sampled_parts = []

        for group in groups:

            group_df = df[df[GROUP_COLUMN] == group]

            sampled_group = group_df.sample(
                n=counts[group],
                random_state=int(
                    rng.integers(0, 1_000_000_000)
                )
            )

            sampled_parts.append(sampled_group)

        sample = pd.concat(
            sampled_parts,
            ignore_index=True
        )

        calculate_metrics(sample)

        print(
            f"  repetition {repetition + 1}/{REPETITIONS}"
        )


results_df = pd.DataFrame(results)

OUTPUT_PATH = OUTPUT_DIR / "sample_size_sweep_resnet50.csv"

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nDONE")
print(f"Saved to: {OUTPUT_PATH}")