from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# Resolve the project root from this script's location:
# AuditFlux/scripts/10_sample_threshold_interaction.py -> AuditFlux/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import BFW_DATA_PATH, OUTPUT_DIR

# -------------------------------
# Experiment configuration
# -------------------------------
MODEL = "resnet50"
GROUP_COLUMN = "a1"
LABEL_COLUMN = "label"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
SAMPLE_SIZES = [5_000, 10_000, 20_000, 50_000, 100_000]
REPETITIONS = 100
RANDOM_SEED = 42
CHECKPOINT_EVERY = 10

METRICS = ["FNR", "FPR"]


def proportional_allocation(stratum_sizes, sample_size):
    """Allocate an exact sample size proportionally across strata."""
    sizes = np.asarray(stratum_sizes, dtype=np.int64)
    expected = sizes / sizes.sum() * sample_size
    allocation = np.floor(expected).astype(np.int64)
    remaining = int(sample_size - allocation.sum())

    if remaining:
        fractional = expected - allocation
        # Stable tie-breaking makes results reproducible.
        order = np.argsort(-fractional, kind="stable")
        for idx in order[:remaining]:
            allocation[idx] += 1

    if np.any(allocation > sizes):
        raise ValueError(
            "Requested sample size exceeds available data in at least one stratum."
        )
    return allocation


def calculate_group_metrics(labels, scores, group_codes, group_names, threshold):
    """Return FNR and FPR for each group using vectorized contingency counts."""
    predictions = (scores >= threshold).astype(np.int8)
    # Outcome encoding: 0=TN, 1=FP, 2=FN, 3=TP
    outcomes = labels * 2 + predictions
    combined = group_codes * 4 + outcomes
    counts = np.bincount(
        combined, minlength=len(group_names) * 4
    ).reshape(len(group_names), 4)

    tn, fp, fn, tp = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    positives = tp + fn
    negatives = tn + fp

    fnr = np.divide(
        fn, positives, out=np.full(len(group_names), np.nan), where=positives > 0
    )
    fpr = np.divide(
        fp, negatives, out=np.full(len(group_names), np.nan), where=negatives > 0
    )
    return {"FNR": fnr, "FPR": fpr}


def rank_groups(values):
    """Rank groups worst-to-best; rank 1 is the worst error rate."""
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        raise ValueError("A group metric is undefined; increase sample size or inspect data.")
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks, int(order[0])


def summarize_transition(values_from, values_to, group_names):
    ranks_from, worst_from = rank_groups(values_from)
    ranks_to, worst_to = rank_groups(values_to)
    tau = float(kendalltau(ranks_from, ranks_to).statistic)
    gap_from = float(np.max(values_from) - np.min(values_from))
    gap_to = float(np.max(values_to) - np.min(values_to))
    return {
        "worst_group_from": group_names[worst_from],
        "worst_group_to": group_names[worst_to],
        "worst_group_reversal": int(worst_from != worst_to),
        "kendall_tau": tau,
        "ranking_instability": 1.0 - tau,
        "gap_from": gap_from,
        "gap_to": gap_to,
        "magnitude_change": abs(gap_to - gap_from),
    }


def make_strata(labels, group_codes):
    """Return indices and labels for every observed group × verification-label stratum."""
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
    stratum_sizes = [len(indices) for _, _, indices in strata]
    allocations = proportional_allocation(stratum_sizes, sample_size)
    chosen = []
    for (_, _, indices), count in zip(strata, allocations):
        if count:
            chosen.append(rng.choice(indices, size=int(count), replace=False))
    return np.concatenate(chosen)


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {BFW_DATA_PATH} ...")
    df = pd.read_csv(
        BFW_DATA_PATH,
        usecols=[LABEL_COLUMN, MODEL, GROUP_COLUMN],
    )
    df = df.dropna(subset=[LABEL_COLUMN, MODEL, GROUP_COLUMN]).reset_index(drop=True)

    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int8)
    scores = df[MODEL].to_numpy(dtype=float)
    group_names = np.sort(df[GROUP_COLUMN].unique())
    group_to_code = {name: i for i, name in enumerate(group_names)}
    group_codes = df[GROUP_COLUMN].map(group_to_code).to_numpy(dtype=np.int16)

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Expected binary labels encoded as 0 and 1.")
    if len(group_names) < 2:
        raise ValueError("At least two demographic groups are required.")

    strata = make_strata(labels, group_codes)
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Rows: {len(df):,}")
    print(f"Groups: {list(group_names)}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Sample sizes: {SAMPLE_SIZES}")
    print(f"Repetitions per size: {REPETITIONS}")
    print("Sampling is stratified by demographic group × verification label.")
    print("The same sampled observations are reused across all thresholds.")

    raw_rows = []
    transition_rows = []
    transitions = list(zip(THRESHOLDS[:-1], THRESHOLDS[1:]))

    for sample_size in SAMPLE_SIZES:
        print(f"\nSample size: {sample_size:,}")

        for repetition in range(1, REPETITIONS + 1):
            sample_indices = draw_stratified_sample(strata, sample_size, rng)
            sample_labels = labels[sample_indices]
            sample_scores = scores[sample_indices]
            sample_groups = group_codes[sample_indices]

            metrics_by_threshold = {}

            for threshold in THRESHOLDS:
                metrics = calculate_group_metrics(
                    sample_labels, sample_scores, sample_groups,
                    group_names, threshold
                )
                metrics_by_threshold[threshold] = metrics

                for metric in METRICS:
                    values = metrics[metric]
                    ranks, worst = rank_groups(values)
                    for group_idx, group_name in enumerate(group_names):
                        raw_rows.append({
                            "sample_size": sample_size,
                            "repetition": repetition,
                            "threshold": threshold,
                            "metric": metric,
                            "group": group_name,
                            "value": float(values[group_idx]),
                            "rank_worst_to_best": int(ranks[group_idx]),
                            "is_worst_group": int(group_idx == worst),
                        })

            # Paired adjacent-threshold transitions: same sample at both thresholds.
            for metric in METRICS:
                for threshold_from, threshold_to in transitions:
                    stats = summarize_transition(
                        metrics_by_threshold[threshold_from][metric],
                        metrics_by_threshold[threshold_to][metric],
                        group_names,
                    )
                    transition_rows.append({
                        "sample_size": sample_size,
                        "repetition": repetition,
                        "metric": metric,
                        "threshold_from": threshold_from,
                        "threshold_to": threshold_to,
                        **stats,
                    })

            if repetition % CHECKPOINT_EVERY == 0:
                pd.DataFrame(raw_rows).to_csv(
                    output_dir / "sample_threshold_interaction_metrics_checkpoint.csv",
                    index=False,
                )
                pd.DataFrame(transition_rows).to_csv(
                    output_dir / "sample_threshold_interaction_transitions_checkpoint.csv",
                    index=False,
                )
                print(f"  completed {repetition}/{REPETITIONS} repetitions")

    raw_df = pd.DataFrame(raw_rows)
    transitions_df = pd.DataFrame(transition_rows)

    # Aggregate across repetitions, retaining each sample-size × metric × transition.
    summary_df = (
        transitions_df
        .groupby(
            ["sample_size", "metric", "threshold_from", "threshold_to"],
            as_index=False,
        )
        .agg(
            repetitions=("repetition", "nunique"),
            worst_group_reversal_rate=("worst_group_reversal", "mean"),
            mean_ranking_instability=("ranking_instability", "mean"),
            sd_ranking_instability=("ranking_instability", "std"),
            mean_kendall_tau=("kendall_tau", "mean"),
            mean_gap_from=("gap_from", "mean"),
            mean_gap_to=("gap_to", "mean"),
            mean_magnitude_change=("magnitude_change", "mean"),
            sd_magnitude_change=("magnitude_change", "std"),
        )
    )

    # Also report how often each group is worst at each sample-size × threshold × metric.
    worst_group_frequency = (
        raw_df[raw_df["is_worst_group"] == 1]
        .groupby(
            ["sample_size", "threshold", "metric", "group"],
            as_index=False,
        )
        .agg(worst_count=("repetition", "nunique"))
    )
    worst_group_frequency["worst_frequency"] = (
        worst_group_frequency["worst_count"] / REPETITIONS
    )
    worst_group_frequency.to_csv(
        output_dir / "sample_threshold_interaction_worst_group_frequency.csv",
        index=False,
    )

    raw_path = output_dir / "sample_threshold_interaction_metrics.csv"
    transitions_path = output_dir / "sample_threshold_interaction_transitions.csv"
    summary_path = output_dir / "sample_threshold_interaction_summary.csv"

    raw_df.to_csv(raw_path, index=False)
    transitions_df.to_csv(transitions_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nDONE")
    print(f"Group metrics: {raw_path}")
    print(f"Transition metrics: {transitions_path}")
    print(f"Transition summary: {summary_path}")
    print(
        "Worst-group frequencies: "
        f"{output_dir / 'sample_threshold_interaction_worst_group_frequency.csv'}"
    )
    print("\nSummary preview:")
    print(summary_df.head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    main()