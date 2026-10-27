from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import BFW_DATA_PATH, OUTPUT_DIR

MODEL = "resnet50"
GROUP_COLUMN = "a1"
LABEL_COLUMN = "label"

THRESHOLDS = [0.60, 0.65, 0.70]
SAMPLE_SIZES = [5_000, 10_000, 20_000, 50_000, 100_000]
REPETITIONS = 100
RANDOM_SEED = 42
CHECKPOINT_EVERY = 10


def proportional_allocation(stratum_sizes, sample_size):
    sizes = np.asarray(stratum_sizes, dtype=np.int64)
    expected = sizes / sizes.sum() * sample_size
    allocation = np.floor(expected).astype(np.int64)
    remaining = int(sample_size - allocation.sum())
    if remaining:
        fractional = expected - allocation
        order = np.argsort(-fractional, kind="stable")
        for idx in order[:remaining]:
            allocation[idx] += 1
    if np.any(allocation > sizes):
        raise ValueError("Requested sample size exceeds an available stratum.")
    return allocation


def make_strata(labels, group_codes):
    strata = []
    for group_code in range(int(group_codes.max()) + 1):
        for label_value in (0, 1):
            indices = np.flatnonzero(
                (group_codes == group_code) & (labels == label_value)
            )
            if len(indices):
                strata.append((group_code, label_value, indices))
    return strata


def draw_stratified_sample(strata, sample_size, rng):
    sizes = [len(indices) for _, _, indices in strata]
    allocations = proportional_allocation(sizes, sample_size)
    chosen = []
    for (_, _, indices), count in zip(strata, allocations):
        if count:
            chosen.append(rng.choice(indices, size=int(count), replace=False))
    return np.concatenate(chosen)


def calculate_counts(labels, scores, group_codes, group_names, threshold):
    predictions = (scores >= threshold).astype(np.int8)
    # outcome = label * 2 + prediction: TN=0, FP=1, FN=2, TP=3
    outcomes = labels * 2 + predictions
    combined = group_codes * 4 + outcomes
    counts = np.bincount(
        combined, minlength=len(group_names) * 4
    ).reshape(len(group_names), 4)
    tn, fp, fn, tp = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    negatives = tn + fp
    fpr = np.divide(
        fp, negatives, out=np.full(len(group_names), np.nan, dtype=float),
        where=negatives > 0
    )
    return tn, fp, negatives, fpr


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {BFW_DATA_PATH} ...")
    df = pd.read_csv(BFW_DATA_PATH, usecols=[LABEL_COLUMN, MODEL, GROUP_COLUMN])
    df = df.dropna(subset=[LABEL_COLUMN, MODEL, GROUP_COLUMN]).reset_index(drop=True)

    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int8)
    scores = df[MODEL].to_numpy(dtype=float)
    group_names = np.sort(df[GROUP_COLUMN].unique())
    group_to_code = {name: i for i, name in enumerate(group_names)}
    group_codes = df[GROUP_COLUMN].map(group_to_code).to_numpy(dtype=np.int16)

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Expected binary labels encoded as 0 and 1.")
    strata = make_strata(labels, group_codes)
    rng = np.random.default_rng(RANDOM_SEED)

    group_rows = []
    summary_rows = []
    total = len(SAMPLE_SIZES) * REPETITIONS

    print(f"Rows: {len(df):,}; groups: {list(group_names)}")
    print(f"Sample sizes: {SAMPLE_SIZES}; repetitions per size: {REPETITIONS}")
    print(f"Thresholds: {THRESHOLDS}")
    print("Uses Experiment 10's sampling design and RNG sequence.")

    completed = 0
    for sample_size in SAMPLE_SIZES:
        for repetition in range(1, REPETITIONS + 1):
            sample_indices = draw_stratified_sample(strata, sample_size, rng)
            sample_labels = labels[sample_indices]
            sample_scores = scores[sample_indices]
            sample_groups = group_codes[sample_indices]

            for threshold in THRESHOLDS:
                tn, fp, negatives, fpr = calculate_counts(
                    sample_labels, sample_scores, sample_groups,
                    group_names, threshold
                )
                valid_fpr = fpr[~np.isnan(fpr)]
                max_fpr = float(np.max(valid_fpr)) if len(valid_fpr) else np.nan
                tied_worst = (
                    int(np.isclose(valid_fpr, max_fpr, rtol=0, atol=1e-12).sum())
                    if len(valid_fpr) else 0
                )
                all_zero_fp = bool(np.all(fp == 0))
                zero_fp_groups = int(np.sum(fp == 0))
                top_gap = (
                    float(np.sort(valid_fpr)[-1] - np.sort(valid_fpr)[-2])
                    if len(valid_fpr) >= 2 else np.nan
                )

                summary_rows.append({
                    "sample_size": sample_size,
                    "repetition": repetition,
                    "threshold": threshold,
                    "total_false_positives": int(fp.sum()),
                    "total_negatives": int(negatives.sum()),
                    "all_groups_zero_fp": int(all_zero_fp),
                    "groups_with_zero_fp": zero_fp_groups,
                    "groups_tied_at_max_fpr": tied_worst,
                    "unique_worst_group": int(tied_worst == 1),
                    "max_fpr": max_fpr,
                    "second_highest_fpr_gap": top_gap,
                })

                for i, group in enumerate(group_names):
                    group_rows.append({
                        "sample_size": sample_size,
                        "repetition": repetition,
                        "threshold": threshold,
                        "group": group,
                        "TN": int(tn[i]),
                        "FP": int(fp[i]),
                        "negative_denominator_TN_plus_FP": int(negatives[i]),
                        "FPR": float(fpr[i]) if not np.isnan(fpr[i]) else np.nan,
                        "zero_FP": int(fp[i] == 0),
                        "all_groups_zero_FP_in_repetition": int(all_zero_fp),
                    })

            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                pd.DataFrame(group_rows).to_csv(
                    output_dir / "false_positive_count_diagnostics_checkpoint.csv",
                    index=False,
                )
                pd.DataFrame(summary_rows).to_csv(
                    output_dir / "false_positive_count_repetition_summary_checkpoint.csv",
                    index=False,
                )
                print(f"  completed {completed}/{total} samples")

    group_df = pd.DataFrame(group_rows)
    summary_df = pd.DataFrame(summary_rows)
    group_path = output_dir / "false_positive_count_diagnostics.csv"
    summary_path = output_dir / "false_positive_count_repetition_summary.csv"
    group_df.to_csv(group_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    aggregate = (
        summary_df.groupby(["sample_size", "threshold"], as_index=False)
        .agg(
            repetitions=("repetition", "nunique"),
            all_groups_zero_fp_rate=("all_groups_zero_fp", "mean"),
            mean_groups_with_zero_fp=("groups_with_zero_fp", "mean"),
            mean_groups_tied_at_max_fpr=("groups_tied_at_max_fpr", "mean"),
            unique_worst_group_rate=("unique_worst_group", "mean"),
            mean_total_false_positives=("total_false_positives", "mean"),
            mean_total_negatives=("total_negatives", "mean"),
            mean_max_fpr=("max_fpr", "mean"),
            mean_top_gap=("second_highest_fpr_gap", "mean"),
        )
    )
    aggregate_path = output_dir / "false_positive_count_aggregate_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)

    print("\nFinished.")
    print(f"Group-level counts: {group_path}")
    print(f"Per-repetition summary: {summary_path}")
    print(f"Aggregate summary: {aggregate_path}")
    print("\nAggregate diagnostic summary:")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
