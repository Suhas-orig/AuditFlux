from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import BFW_DATA_PATH, OUTPUT_DIR

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODELS = ["vgg16", "resnet50", "senet50"]
GROUP_COLUMN = "a1"
LABEL_COLUMN = "label"
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
METRICS = ["FNR", "FPR"]
PERMUTATION_REPETITIONS = 1000
RANDOM_SEED = 42
CHECKPOINT_EVERY = 100


def calculate_metrics(labels, scores, group_codes, group_count, threshold):
    """Calculate group-level FNR and FPR using vectorised category counts."""
    predictions = (scores >= threshold).astype(np.int8)
    # Outcome codes: 0=TN, 1=FP, 2=FN, 3=TP.
    outcomes = labels * 2 + predictions
    combined = group_codes * 4 + outcomes
    counts = np.bincount(combined, minlength=group_count * 4).reshape(group_count, 4)
    tn, fp, fn, tp = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    positives, negatives = tp + fn, tn + fp
    fnr = np.divide(fn, positives, out=np.full(group_count, np.nan), where=positives > 0)
    fpr = np.divide(fp, negatives, out=np.full(group_count, np.nan), where=negatives > 0)
    return {"FNR": fnr, "FPR": fpr}


def rank_groups(values):
    """Average ranks, with rank 1 assigned to the worst (largest) error rate."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    ranks = np.full(values.shape, np.nan, dtype=float)
    if valid.any():
        ranks[valid] = rankdata(-values[valid], method="average")
    if not valid.any():
        worst_indices = np.array([], dtype=int)
    else:
        worst_value = np.nanmax(values)
        worst_indices = np.flatnonzero(np.isclose(values, worst_value, rtol=1e-12, atol=1e-15))
    return ranks, worst_indices


def transition_stats(values_from, values_to, group_names):
    """Compare two group rankings while preserving ties and undefined cases."""
    ranks_from, worst_from = rank_groups(values_from)
    ranks_to, worst_to = rank_groups(values_to)

    valid_ranks = np.isfinite(ranks_from) & np.isfinite(ranks_to)
    tau = np.nan
    if valid_ranks.sum() >= 2:
        result = kendalltau(ranks_from[valid_ranks], ranks_to[valid_ranks], variant="b")
        if np.isfinite(result.statistic):
            tau = float(result.statistic)

    worst_defined = len(worst_from) > 0 and len(worst_to) > 0
    worst_changed = np.nan
    if worst_defined:
        worst_set_from = set(worst_from.tolist())
        worst_set_to = set(worst_to.tolist())
        worst_changed = int(worst_set_from != worst_set_to)

    finite_from = np.asarray(values_from, dtype=float)
    finite_to = np.asarray(values_to, dtype=float)
    finite_from = finite_from[np.isfinite(finite_from)]
    finite_to = finite_to[np.isfinite(finite_to)]
    gap_from = float(np.max(finite_from) - np.min(finite_from)) if len(finite_from) else np.nan
    gap_to = float(np.max(finite_to) - np.min(finite_to)) if len(finite_to) else np.nan

    return {
        "worst_group_from": "|".join(str(group_names[i]) for i in worst_from),
        "worst_group_to": "|".join(str(group_names[i]) for i in worst_to),
        "worst_group_from_count": int(len(worst_from)),
        "worst_group_to_count": int(len(worst_to)),
        "worst_group_comparison_defined": int(worst_defined),
        "worst_group_set_changed": worst_changed,
        "kendall_tau_b": tau,
        "ranking_instability": 1.0 - tau if np.isfinite(tau) else np.nan,
        "gap_from": gap_from,
        "gap_to": gap_to,
        "magnitude_change": abs(gap_to - gap_from) if np.isfinite(gap_from) and np.isfinite(gap_to) else np.nan,
    }


def safe_null_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": np.nan, "sd": np.nan, "q025": np.nan, "q50": np.nan, "q975": np.nan, "n_defined": 0}
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "q025": float(np.quantile(values, 0.025)),
        "q50": float(np.quantile(values, 0.50)),
        "q975": float(np.quantile(values, 0.975)),
        "n_defined": int(len(values)),
    }


def empirical_upper_p(null_values, observed_value):
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed_value) or len(values) == 0:
        return np.nan
    return float((1 + np.sum(values >= observed_value)) / (len(values) + 1))


def empirical_lower_p(null_values, observed_value):
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed_value) or len(values) == 0:
        return np.nan
    return float((1 + np.sum(values <= observed_value)) / (len(values) + 1))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {BFW_DATA_PATH} ...")
    usecols = [LABEL_COLUMN, GROUP_COLUMN, *MODELS]
    df = pd.read_csv(BFW_DATA_PATH, usecols=usecols)
    df = df.dropna(subset=usecols).reset_index(drop=True)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(np.int8)

    unexpected_labels = sorted(set(df[LABEL_COLUMN].unique()) - {0, 1})
    if unexpected_labels:
        raise ValueError(f"Expected labels 0 and 1; found {unexpected_labels}")

    group_names = np.sort(df[GROUP_COLUMN].unique())
    group_to_code = {name: i for i, name in enumerate(group_names)}
    original_group_codes = df[GROUP_COLUMN].map(group_to_code).to_numpy(dtype=np.int16)
    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int8)
    scores_by_model = {model: df[model].to_numpy(dtype=float) for model in MODELS}
    group_count = len(group_names)

    print(f"Rows: {len(df):,}; groups: {list(group_names)}; models: {MODELS}")
    print(f"Permutations: {PERMUTATION_REPETITIONS:,}; seed: {RANDOM_SEED}")
    print("Null: group assignments permuted within each true-label stratum.")

    # Full-data observed results.
    observed_metric_rows = []
    observed_transition_rows = []
    observed_by_model = {}

    for model in MODELS:
        scores = scores_by_model[model]
        observed_by_model[model] = {}
        for threshold in THRESHOLDS:
            metrics = calculate_metrics(labels, scores, original_group_codes, group_count, threshold)
            observed_by_model[model][threshold] = metrics
            for metric in METRICS:
                values = metrics[metric]
                ranks, worst_indices = rank_groups(values)
                for i, group in enumerate(group_names):
                    observed_metric_rows.append({
                        "model": model,
                        "threshold": threshold,
                        "metric": metric,
                        "group": group,
                        "value": float(values[i]) if np.isfinite(values[i]) else np.nan,
                        "rank_worst_to_best": float(ranks[i]) if np.isfinite(ranks[i]) else np.nan,
                        "is_worst_group": int(i in set(worst_indices.tolist())),
                        "worst_group_tied_count": int(len(worst_indices)),
                    })

        for metric in METRICS:
            for t_from, t_to in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
                stats = transition_stats(
                    observed_by_model[model][t_from][metric],
                    observed_by_model[model][t_to][metric],
                    group_names,
                )
                observed_transition_rows.append({
                    "source": "observed_full_data",
                    "permutation": -1,
                    "model": model,
                    "metric": metric,
                    "threshold_from": t_from,
                    "threshold_to": t_to,
                    **stats,
                })

    # Permutation null. A single permuted group assignment is reused across
    # all thresholds and all three models within each replicate (paired design).
    rng = np.random.default_rng(RANDOM_SEED)
    strata = [np.flatnonzero(labels == y) for y in np.unique(labels)]
    null_rows = []

    for rep in range(PERMUTATION_REPETITIONS):
        permuted_groups = original_group_codes.copy()
        for idx in strata:
            permuted_groups[idx] = rng.permutation(original_group_codes[idx])

        for model in MODELS:
            scores = scores_by_model[model]
            metrics_by_threshold = {
                threshold: calculate_metrics(labels, scores, permuted_groups, group_count, threshold)
                for threshold in THRESHOLDS
            }
            for metric in METRICS:
                for t_from, t_to in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
                    stats = transition_stats(
                        metrics_by_threshold[t_from][metric],
                        metrics_by_threshold[t_to][metric],
                        group_names,
                    )
                    null_rows.append({
                        "source": "permutation_null",
                        "permutation": rep + 1,
                        "model": model,
                        "metric": metric,
                        "threshold_from": t_from,
                        "threshold_to": t_to,
                        **stats,
                    })

        if (rep + 1) % CHECKPOINT_EVERY == 0:
            checkpoint_path = OUTPUT_DIR / "null_calibration_transitions_checkpoint.csv"
            pd.DataFrame(null_rows).to_csv(checkpoint_path, index=False)
            print(f"Completed {rep + 1:,}/{PERMUTATION_REPETITIONS:,} permutations")

    observed_df = pd.DataFrame(observed_metric_rows)
    observed_transitions_df = pd.DataFrame(observed_transition_rows)
    null_df = pd.DataFrame(null_rows)
    transitions_df = pd.concat([observed_transitions_df, null_df], ignore_index=True)

    # Compare each observed transition with its matching permutation null.
    summary_rows = []
    summary_metrics = [
        "worst_group_set_changed", "kendall_tau_b", "ranking_instability",
        "gap_from", "gap_to", "magnitude_change",
    ]
    for _, obs in observed_transitions_df.iterrows():
        subset = null_df[
            (null_df["model"] == obs["model"])
            & (null_df["metric"] == obs["metric"])
            & (null_df["threshold_from"] == obs["threshold_from"])
            & (null_df["threshold_to"] == obs["threshold_to"])
        ]
        row = {
            "model": obs["model"],
            "metric": obs["metric"],
            "threshold_from": obs["threshold_from"],
            "threshold_to": obs["threshold_to"],
            "observed_worst_group_from": obs["worst_group_from"],
            "observed_worst_group_to": obs["worst_group_to"],
            "observed_worst_group_comparison_defined": obs["worst_group_comparison_defined"],
            "observed_worst_group_set_changed": obs["worst_group_set_changed"],
            "observed_kendall_tau_b": obs["kendall_tau_b"],
            "observed_ranking_instability": obs["ranking_instability"],
            "observed_gap_from": obs["gap_from"],
            "observed_gap_to": obs["gap_to"],
            "observed_magnitude_change": obs["magnitude_change"],
            "null_permutations": int(len(subset)),
        }
        for col in summary_metrics:
            stats = safe_null_summary(subset[col].to_numpy(dtype=float))
            for stat_name, stat_value in stats.items():
                row[f"null_{col}_{stat_name}"] = stat_value

        row["empirical_p_worst_group_set_changed"] = empirical_upper_p(
            subset["worst_group_set_changed"].to_numpy(dtype=float), obs["worst_group_set_changed"]
        )
        row["empirical_p_ranking_instability"] = empirical_upper_p(
            subset["ranking_instability"].to_numpy(dtype=float), obs["ranking_instability"]
        )
        row["empirical_p_low_kendall_tau_b"] = empirical_lower_p(
            subset["kendall_tau_b"].to_numpy(dtype=float), obs["kendall_tau_b"]
        )
        row["empirical_p_magnitude_change"] = empirical_upper_p(
            subset["magnitude_change"].to_numpy(dtype=float), obs["magnitude_change"]
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    observed_df.to_csv(OUTPUT_DIR / "null_calibration_observed_metrics_all_models.csv", index=False)
    transitions_df.to_csv(OUTPUT_DIR / "null_calibration_transitions_all_models.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "null_calibration_summary_all_models.csv", index=False)

    # Model-specific convenience files, matching the project's existing pattern.
    for model in MODELS:
        observed_df[observed_df["model"] == model].to_csv(
            OUTPUT_DIR / f"null_calibration_observed_metrics_{model}.csv", index=False
        )
        transitions_df[transitions_df["model"] == model].to_csv(
            OUTPUT_DIR / f"null_calibration_transitions_{model}.csv", index=False
        )
        summary_df[summary_df["model"] == model].to_csv(
            OUTPUT_DIR / f"null_calibration_summary_{model}.csv", index=False
        )

    print("\nSaved combined outputs:")
    print(OUTPUT_DIR / "null_calibration_observed_metrics_all_models.csv")
    print(OUTPUT_DIR / "null_calibration_transitions_all_models.csv")
    print(OUTPUT_DIR / "null_calibration_summary_all_models.csv")
    print("\nInterpretation note:")
    print("This is a row-level conditional permutation null: group labels are shuffled")
    print("within true-label strata, preserving group-label counts while breaking")
    print("the association between group assignment and model scores.")
    print("It does not account for dependence between pairs sharing identities.")
    print("Undefined rankings (for example, all groups tied) are retained as NaN,")
    print("rather than being assigned artificial ranks.")


if __name__ == "__main__":
    main()
