from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path

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
PERMUTATION_REPETITIONS = 5_000
RANDOM_SEED = 42
CHECKPOINT_EVERY = 100
TIE_TOLERANCE = 1e-12

RUN_NAME = "null_calibration_v2"
CHECKPOINT_DIR = OUTPUT_DIR / f"{RUN_NAME}_checkpoints"


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
    """Average ranks with tolerance-aware ties; rank 1 is the worst error rate."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    ranks = np.full(values.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return ranks, np.array([], dtype=int)

    # Round only for ranking so near-equal floating-point rates are treated
    # consistently with the explicit tie tolerance used for worst-group sets.
    rounded = np.round(values[valid], decimals=12)
    ranks[valid] = rankdata(-rounded, method="average")

    # Use the same rounded values for both ranks and worst-group ties.
    worst_indices = valid_indices[rounded == np.max(rounded)]
    return ranks, worst_indices


def transition_stats(values_from, values_to, group_names):
    """Compare rankings, worst-group sets and disparity gaps across thresholds."""
    ranks_from, worst_from = rank_groups(values_from)
    ranks_to, worst_to = rank_groups(values_to)

    valid_ranks = np.isfinite(ranks_from) & np.isfinite(ranks_to)
    tau = np.nan
    if valid_ranks.sum() >= 2:
        result = kendalltau(
            ranks_from[valid_ranks], ranks_to[valid_ranks], variant="b"
        )
        if np.isfinite(result.statistic):
            tau = float(result.statistic)

    worst_defined = len(worst_from) > 0 and len(worst_to) > 0
    worst_changed = np.nan
    if worst_defined:
        worst_changed = int(set(worst_from.tolist()) != set(worst_to.tolist()))

    finite_from = np.asarray(values_from, dtype=float)
    finite_to = np.asarray(values_to, dtype=float)
    finite_from = finite_from[np.isfinite(finite_from)]
    finite_to = finite_to[np.isfinite(finite_to)]
    gap_from = (
        float(np.max(finite_from) - np.min(finite_from))
        if len(finite_from) else np.nan
    )
    gap_to = (
        float(np.max(finite_to) - np.min(finite_to))
        if len(finite_to) else np.nan
    )
    signed_change = gap_to - gap_from if np.isfinite(gap_from) and np.isfinite(gap_to) else np.nan
    magnitude_change = abs(signed_change) if np.isfinite(signed_change) else np.nan

    return {
        "worst_group_from": "|".join(str(group_names[i]) for i in worst_from),
        "worst_group_to": "|".join(str(group_names[i]) for i in worst_to),
        "worst_group_from_count": int(len(worst_from)),
        "worst_group_to_count": int(len(worst_to)),
        "worst_group_comparison_defined": int(worst_defined),
        "worst_group_set_changed": worst_changed,
        "kendall_tau_b": tau,
        # Original scale is retained: 1-tau ranges from 0 to 2.
        "ranking_instability": 1.0 - tau if np.isfinite(tau) else np.nan,
        "ranking_instability_normalized": (1.0 - tau) / 2.0 if np.isfinite(tau) else np.nan,
        "gap_from": gap_from,
        "gap_to": gap_to,
        "signed_gap_change": signed_change,
        "magnitude_change": magnitude_change,
    }


def safe_null_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "mean": np.nan, "sd": np.nan, "q025": np.nan,
            "q50": np.nan, "q975": np.nan, "n_defined": 0,
        }
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


def dataset_fingerprint(labels, group_codes, scores_by_model, group_names):
    """Fingerprint the actual arrays used, not merely the CSV path and row count."""
    digest = hashlib.sha256()
    for array in [labels, group_codes, *[scores_by_model[m] for m in MODELS]]:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    digest.update("|".join(map(str, group_names)).encode("utf-8"))
    return digest.hexdigest()


def configuration_fingerprint(data_fingerprint):
    payload = {
        "run_name": RUN_NAME,
        "data_fingerprint": data_fingerprint,
        "models": MODELS,
        "group_column": GROUP_COLUMN,
        "label_column": LABEL_COLUMN,
        "thresholds": THRESHOLDS,
        "metrics": METRICS,
        "permutation_repetitions": PERMUTATION_REPETITIONS,
        "random_seed": RANDOM_SEED,
        "tie_tolerance": TIE_TOLERANCE,
        "null_design": "group labels permuted within true-label strata; paired across models and thresholds",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def atomic_json_write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    temporary.replace(path)


def apply_holm(summary_df, p_column, adjusted_column):
    p_values = summary_df[p_column].to_numpy(dtype=float)
    valid = np.isfinite(p_values)
    adjusted = np.full(len(summary_df), np.nan, dtype=float)
    if valid.any():
        original_indices = np.flatnonzero(valid)
        order = np.argsort(p_values[valid])
        sorted_p = p_values[valid][order]
        m = len(sorted_p)
        holm_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
        holm_sorted = np.clip(holm_sorted, 0.0, 1.0)
        adjusted[original_indices[order]] = holm_sorted
    summary_df[adjusted_column] = adjusted


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {BFW_DATA_PATH} ...")
    usecols = [LABEL_COLUMN, GROUP_COLUMN, *MODELS]
    df = pd.read_csv(BFW_DATA_PATH, usecols=usecols)

    missing = df[usecols].isna().sum()
    if missing.any():
        raise ValueError(
            "Missing values found in required columns; refusing to silently drop rows:\n"
            + missing[missing > 0].to_string()
        )

    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int8)
    unexpected_labels = sorted(set(np.unique(labels)) - {0, 1})
    if unexpected_labels:
        raise ValueError(f"Expected labels 0 and 1; found {unexpected_labels}")

    group_names, original_group_codes = np.unique(
        df[GROUP_COLUMN].to_numpy(), return_inverse=True
    )
    original_group_codes = original_group_codes.astype(np.int16, copy=False)
    scores_by_model = {
        model: df[model].to_numpy(dtype=np.float64) for model in MODELS
    }
    for model, scores in scores_by_model.items():
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite scores found for {model}.")

    n = len(df)
    group_count = len(group_names)
    del df
    gc.collect()

    data_fp = dataset_fingerprint(labels, original_group_codes, scores_by_model, group_names)
    config_fp = configuration_fingerprint(data_fp)

    print(f"Rows: {n:,}; groups: {list(group_names)}; models: {MODELS}")
    print(f"Permutations: {PERMUTATION_REPETITIONS:,}; seed: {RANDOM_SEED}")
    print("Null: group assignments permuted within each true-label stratum.")
    print("Pairing: one permuted assignment reused across all models and thresholds.")

    # Full-data observed results.
    observed_metric_rows = []
    observed_transition_rows = []
    observed_by_model = {}

    for model in MODELS:
        observed_by_model[model] = {}
        for threshold in THRESHOLDS:
            metrics = calculate_metrics(
                labels, scores_by_model[model], original_group_codes,
                group_count, threshold
            )
            observed_by_model[model][threshold] = metrics
            for metric in METRICS:
                values = metrics[metric]
                ranks, worst_indices = rank_groups(values)
                worst_set = set(worst_indices.tolist())
                for i, group in enumerate(group_names):
                    observed_metric_rows.append({
                        "model": model,
                        "threshold": threshold,
                        "metric": metric,
                        "group": group,
                        "value": float(values[i]) if np.isfinite(values[i]) else np.nan,
                        "rank_worst_to_best": float(ranks[i]) if np.isfinite(ranks[i]) else np.nan,
                        "is_worst_group": int(i in worst_set),
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

    observed_df = pd.DataFrame(observed_metric_rows)
    observed_transitions_df = pd.DataFrame(observed_transition_rows)

    # -----------------------------------------------------------------
    # Resumable permutation null, saved as independent 100-repetition chunks.
    # The checkpoint metadata is written only after its chunk is safely saved.
    # -----------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    metadata_path = CHECKPOINT_DIR / "checkpoint_metadata.json"
    completed_repetitions = 0

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("configuration_fingerprint") != config_fp:
            raise RuntimeError(
                "Existing checkpoints do not match the current data/configuration. "
                f"Use a new RUN_NAME or remove {CHECKPOINT_DIR} before starting fresh."
            )
        completed_repetitions = int(metadata["completed_repetitions"])
        rng.bit_generator.state = metadata["rng_state"]
        print(f"Resuming after {completed_repetitions:,} completed permutations.")
    else:
        # Do not silently mix orphaned chunks from an interrupted first run.
        for old_chunk in CHECKPOINT_DIR.glob("null_chunk_*.csv"):
            old_chunk.unlink()
        print("Starting a new permutation run.")

    strata = [np.flatnonzero(labels == y) for y in np.unique(labels)]

    for batch_start in range(completed_repetitions + 1, PERMUTATION_REPETITIONS + 1, CHECKPOINT_EVERY):
        batch_end = min(batch_start + CHECKPOINT_EVERY - 1, PERMUTATION_REPETITIONS)
        batch_rows = []

        for rep in range(batch_start, batch_end + 1):
            permuted_groups = original_group_codes.copy()
            for idx in strata:
                permuted_groups[idx] = rng.permutation(original_group_codes[idx])

            for model in MODELS:
                scores = scores_by_model[model]
                metrics_by_threshold = {
                    threshold: calculate_metrics(
                        labels, scores, permuted_groups, group_count, threshold
                    )
                    for threshold in THRESHOLDS
                }
                for metric in METRICS:
                    for t_from, t_to in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
                        stats = transition_stats(
                            metrics_by_threshold[t_from][metric],
                            metrics_by_threshold[t_to][metric],
                            group_names,
                        )
                        batch_rows.append({
                            "source": "permutation_null",
                            "permutation": rep,
                            "model": model,
                            "metric": metric,
                            "threshold_from": t_from,
                            "threshold_to": t_to,
                            **stats,
                        })

            if rep % 100 == 0:
                print(f"Completed {rep:,}/{PERMUTATION_REPETITIONS:,} permutations")

        chunk_path = CHECKPOINT_DIR / f"null_chunk_{batch_end:05d}.csv"
        temporary_chunk = chunk_path.with_suffix(".csv.tmp")
        pd.DataFrame(batch_rows).to_csv(temporary_chunk, index=False)
        temporary_chunk.replace(chunk_path)

        atomic_json_write(metadata_path, {
            "configuration_fingerprint": config_fp,
            "data_fingerprint": data_fp,
            "completed_repetitions": batch_end,
            "rng_state": rng.bit_generator.state,
        })
        del batch_rows
        gc.collect()

    # Read only chunks confirmed by checkpoint metadata.
    chunk_paths = sorted(CHECKPOINT_DIR.glob("null_chunk_*.csv"))
    expected_chunks = int(np.ceil(PERMUTATION_REPETITIONS / CHECKPOINT_EVERY))
    if len(chunk_paths) != expected_chunks:
        raise RuntimeError(
            f"Expected {expected_chunks} checkpoint chunks, found {len(chunk_paths)}."
        )
    null_df = pd.concat((pd.read_csv(path) for path in chunk_paths), ignore_index=True)
    expected_null_rows = PERMUTATION_REPETITIONS * len(MODELS) * len(METRICS) * (len(THRESHOLDS) - 1)
    if len(null_df) != expected_null_rows:
        raise RuntimeError(f"Expected {expected_null_rows} null rows, found {len(null_df)}.")

    # Compare each observed transition with its matching permutation null.
    summary_rows = []
    summary_metrics = [
        "worst_group_set_changed", "kendall_tau_b", "ranking_instability",
        "ranking_instability_normalized", "gap_from", "gap_to",
        "signed_gap_change", "magnitude_change",
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
            "observed_ranking_instability_normalized": obs["ranking_instability_normalized"],
            "observed_gap_from": obs["gap_from"],
            "observed_gap_to": obs["gap_to"],
            "observed_signed_gap_change": obs["signed_gap_change"],
            "observed_magnitude_change": obs["magnitude_change"],
            "null_permutations": int(len(subset)),
        }
        for col in summary_metrics:
            stats = safe_null_summary(subset[col].to_numpy(dtype=float))
            for stat_name, stat_value in stats.items():
                row[f"null_{col}_{stat_name}"] = stat_value

        null_worst = subset["worst_group_set_changed"].to_numpy(dtype=float)
        obs_worst = float(obs["worst_group_set_changed"])
        row["empirical_p_worst_change_upper"] = empirical_upper_p(null_worst, obs_worst)
        row["empirical_p_worst_change_lower"] = empirical_lower_p(null_worst, obs_worst)
        row["empirical_p_worst_change_two_sided"] = min(
            1.0,
            2.0 * min(
                row["empirical_p_worst_change_upper"],
                row["empirical_p_worst_change_lower"],
            ),
        ) if np.isfinite(obs_worst) else np.nan

        null_tau = subset["kendall_tau_b"].to_numpy(dtype=float)
        row["empirical_p_low_kendall_tau_b"] = empirical_lower_p(
            null_tau, float(obs["kendall_tau_b"])
        )
        row["empirical_p_magnitude_change_upper"] = empirical_upper_p(
            subset["magnitude_change"].to_numpy(dtype=float),
            float(obs["magnitude_change"]),
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Control family-wise error across the 48 model/metric/transition tests
    # separately for each pre-defined inferential endpoint.
    apply_holm(summary_df, "empirical_p_worst_change_upper", "holm_p_worst_change_upper")
    apply_holm(summary_df, "empirical_p_worst_change_lower", "holm_p_worst_change_lower")
    apply_holm(summary_df, "empirical_p_worst_change_two_sided", "holm_p_worst_change_two_sided")
    apply_holm(summary_df, "empirical_p_low_kendall_tau_b", "holm_p_low_kendall_tau_b")
    apply_holm(summary_df, "empirical_p_magnitude_change_upper", "holm_p_magnitude_change_upper")

    # Save combined outputs. Previous experiment outputs are not overwritten.
    observed_path = OUTPUT_DIR / f"{RUN_NAME}_observed_metrics_all_models.csv"
    transitions_path = OUTPUT_DIR / f"{RUN_NAME}_transitions_all_models.csv"
    summary_path = OUTPUT_DIR / f"{RUN_NAME}_summary_all_models.csv"
    null_path = OUTPUT_DIR / f"{RUN_NAME}_null_transitions_all_models.csv"

    transitions_df = pd.concat([observed_transitions_df, null_df], ignore_index=True)
    observed_df.to_csv(observed_path, index=False)
    transitions_df.to_csv(transitions_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    null_df.to_csv(null_path, index=False)

    for model in MODELS:
        observed_df[observed_df["model"] == model].to_csv(
            OUTPUT_DIR / f"{RUN_NAME}_observed_metrics_{model}.csv", index=False
        )
        transitions_df[transitions_df["model"] == model].to_csv(
            OUTPUT_DIR / f"{RUN_NAME}_transitions_{model}.csv", index=False
        )
        summary_df[summary_df["model"] == model].to_csv(
            OUTPUT_DIR / f"{RUN_NAME}_summary_{model}.csv", index=False
        )

    print("\nSaved revised null-calibration outputs:")
    print(observed_path)
    print(transitions_path)
    print(summary_path)
    print(null_path)
    print("\nInterpretation notes:")
    print("- This is a row-level conditional permutation null.")
    print("- Group assignments are shuffled within true-label strata, preserving group counts per label.")
    print("- Each permuted assignment is shared across all models and thresholds within a replicate.")
    print("- Ranking instability = 1 - Kendall tau-b (range 0-2); normalized version ranges 0-1.")
    print("- Kendall lower-tail p-value and ranking-instability upper-tail p-value are complements and are not both reported.")
    print("- Holm adjustment is applied separately to each endpoint across the 48 transitions.")
    print("- Row-level permutation does not preserve identity-level dependence or pair structure.")
    print("- These results calibrate observed statistics against this conditional null; they do not establish causation.")


if __name__ == "__main__":
    main()
