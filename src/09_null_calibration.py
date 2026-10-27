
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import BFW_DATA_PATH, OUTPUT_DIR

# -------------------------------
# Configuration
# -------------------------------
MODEL = "resnet50"
GROUP_COLUMN = "a1"
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
METRICS = ["FNR", "FPR"]
PERMUTATION_REPETITIONS = 1000
RANDOM_SEED = 42
CHECKPOINT_EVERY = 100


def calculate_metrics(labels, scores, group_codes, group_names, threshold):
    """Calculate FNR/FPR for each group using integer category counts."""
    predictions = (scores >= threshold).astype(np.int8)
    # 0=TN, 1=FP, 2=FN, 3=TP
    outcomes = labels * 2 + predictions
    combined = group_codes * 4 + outcomes
    counts = np.bincount(
        combined, minlength=len(group_names) * 4
    ).reshape(len(group_names), 4)
    tn, fp, fn, tp = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    positives, negatives = tp + fn, tn + fp
    fnr = np.divide(fn, positives, out=np.full(len(group_names), np.nan), where=positives > 0)
    fpr = np.divide(fp, negatives, out=np.full(len(group_names), np.nan), where=negatives > 0)
    return {"FNR": fnr, "FPR": fpr}


def rank_groups(values, metric):
    """Return integer ranks in worst-to-best order (1 = worst)."""
    # FNR/FPR: larger value is worse. Stable tie handling by group code.
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks, int(order[0])


def transition_stats(values_from, values_to, metric):
    ranks_from, worst_from = rank_groups(values_from, metric)
    ranks_to, worst_to = rank_groups(values_to, metric)
    tau = float(kendalltau(ranks_from, ranks_to).statistic)
    gap_from = float(np.nanmax(values_from) - np.nanmin(values_from))
    gap_to = float(np.nanmax(values_to) - np.nanmin(values_to))
    return {
        "worst_group_from_code": worst_from,
        "worst_group_to_code": worst_to,
        "worst_group_reversal": int(worst_from != worst_to),
        "kendall_tau": tau,
        "ranking_instability": 1.0 - tau,
        "gap_from": gap_from,
        "gap_to": gap_to,
        "magnitude_change": abs(gap_to - gap_from),
    }


def main():
    print(f"Loading {BFW_DATA_PATH} ...")
    df = pd.read_csv(BFW_DATA_PATH, usecols=["label", MODEL, GROUP_COLUMN])
    df = df.dropna(subset=["label", MODEL, GROUP_COLUMN]).reset_index(drop=True)
    labels = df["label"].to_numpy(dtype=np.int8)
    scores = df[MODEL].to_numpy(dtype=float)
    group_names = np.sort(df[GROUP_COLUMN].unique())
    group_to_code = {name: i for i, name in enumerate(group_names)}
    original_group_codes = df[GROUP_COLUMN].map(group_to_code).to_numpy(dtype=np.int16)

    print(f"Rows: {len(df):,}; groups: {list(group_names)}")
    print(f"Permutations: {PERMUTATION_REPETITIONS:,}; seed: {RANDOM_SEED}")

    # Observed full-data metrics at each threshold.
    observed = {}
    observed_rows = []
    for threshold in THRESHOLDS:
        metrics = calculate_metrics(labels, scores, original_group_codes, group_names, threshold)
        observed[threshold] = metrics
        for metric in METRICS:
            vals = metrics[metric]
            ranks, worst = rank_groups(vals, metric)
            for i, group in enumerate(group_names):
                observed_rows.append({
                    "threshold": threshold, "metric": metric, "group": group,
                    "value": float(vals[i]), "rank_worst_to_best": int(ranks[i]),
                    "is_worst_group": int(i == worst),
                })

    observed_transitions = []
    for metric in METRICS:
        for t_from, t_to in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
            stats = transition_stats(observed[t_from][metric], observed[t_to][metric], metric)
            observed_transitions.append({
                "source": "observed_full_data", "permutation": -1,
                "metric": metric, "threshold_from": t_from, "threshold_to": t_to,
                "worst_group_from": group_names[stats.pop("worst_group_from_code")],
                "worst_group_to": group_names[stats.pop("worst_group_to_code")],
                **stats,
            })

    rng = np.random.default_rng(RANDOM_SEED)
    null_rows = []
    # Permute group assignments within each true-label stratum. One permuted
    # assignment is reused across all thresholds within each replicate.
    strata = [np.flatnonzero(labels == y) for y in np.unique(labels)]
    for rep in range(PERMUTATION_REPETITIONS):
        permuted_groups = original_group_codes.copy()
        for idx in strata:
            permuted_groups[idx] = rng.permutation(original_group_codes[idx])

        metrics_by_threshold = {}
        for threshold in THRESHOLDS:
            metrics_by_threshold[threshold] = calculate_metrics(
                labels, scores, permuted_groups, group_names, threshold
            )

        for metric in METRICS:
            for t_from, t_to in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
                stats = transition_stats(
                    metrics_by_threshold[t_from][metric],
                    metrics_by_threshold[t_to][metric], metric
                )
                null_rows.append({
                    "source": "permutation_null", "permutation": rep,
                    "metric": metric, "threshold_from": t_from, "threshold_to": t_to,
                    "worst_group_from": group_names[stats.pop("worst_group_from_code")],
                    "worst_group_to": group_names[stats.pop("worst_group_to_code")],
                    **stats,
                })

        if (rep + 1) % CHECKPOINT_EVERY == 0:
            print(f"Completed {rep + 1:,}/{PERMUTATION_REPETITIONS:,} permutations")
            pd.DataFrame(null_rows).to_csv(OUTPUT_DIR / "null_calibration_transitions_checkpoint.csv", index=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observed_df = pd.DataFrame(observed_rows)
    transitions_df = pd.concat(
        [pd.DataFrame(observed_transitions), pd.DataFrame(null_rows)], ignore_index=True
    )
    null_df = transitions_df[transitions_df["source"] == "permutation_null"]

    summary_rows = []
    for _, obs in pd.DataFrame(observed_transitions).iterrows():
        subset = null_df[
            (null_df["metric"] == obs["metric"])
            & (null_df["threshold_from"] == obs["threshold_from"])
            & (null_df["threshold_to"] == obs["threshold_to"])
        ]
        row = {
            "metric": obs["metric"],
            "threshold_from": obs["threshold_from"],
            "threshold_to": obs["threshold_to"],
            "observed_worst_group_from": obs["worst_group_from"],
            "observed_worst_group_to": obs["worst_group_to"],
            "observed_worst_group_reversal": obs["worst_group_reversal"],
            "observed_kendall_tau": obs["kendall_tau"],
            "observed_ranking_instability": obs["ranking_instability"],
            "observed_gap_from": obs["gap_from"],
            "observed_gap_to": obs["gap_to"],
            "observed_magnitude_change": obs["magnitude_change"],
        }
        for col in ["worst_group_reversal", "kendall_tau", "ranking_instability", "gap_from", "gap_to", "magnitude_change"]:
            vals = subset[col].to_numpy(dtype=float)
            row[f"null_{col}_mean"] = float(np.nanmean(vals))
            row[f"null_{col}_sd"] = float(np.nanstd(vals, ddof=1))
            row[f"null_{col}_q025"] = float(np.nanquantile(vals, 0.025))
            row[f"null_{col}_q50"] = float(np.nanquantile(vals, 0.50))
            row[f"null_{col}_q975"] = float(np.nanquantile(vals, 0.975))
        # One-sided empirical tail probabilities. For reversal/instability/change,
        # larger is more extreme; for tau, smaller is more extreme.
        for col in ["worst_group_reversal", "ranking_instability", "magnitude_change"]:
            row[f"empirical_p_{col}"] = (1 + np.sum(subset[col].to_numpy() >= obs[col])) / (len(subset) + 1)
        row["empirical_p_low_kendall_tau"] = (1 + np.sum(subset["kendall_tau"].to_numpy() <= obs["kendall_tau"])) / (len(subset) + 1)
        summary_rows.append(row)

    observed_df.to_csv(OUTPUT_DIR / "null_calibration_observed_metrics.csv", index=False)
    transitions_df.to_csv(OUTPUT_DIR / "null_calibration_transitions.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "null_calibration_summary.csv", index=False)

    print("\nSaved:")
    print(OUTPUT_DIR / "null_calibration_observed_metrics.csv")
    print(OUTPUT_DIR / "null_calibration_transitions.csv")
    print(OUTPUT_DIR / "null_calibration_summary.csv")
    print("\nImportant: this permutation null tests group-score association conditional on label.")
    print("It is distinct from bootstrap sampling uncertainty in paired_threshold_bootstrap_summary.csv.")


if __name__ == "__main__":
    main()
