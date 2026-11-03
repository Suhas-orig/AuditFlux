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
IDENTITY_COLUMN = "id1"
FOLD_COLUMN = "fold"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
METRICS = ["FNR", "FPR"]

PERMUTATION_REPETITIONS = 5000
RANDOM_SEED = 42
CHECKPOINT_EVERY = 100
TIE_DECIMALS = 12

PREFIX = "identity_null_calibration_v1"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def calculate_metrics(labels, scores, group_codes, group_count, threshold):
    predictions = (scores >= threshold).astype(np.int8)
    outcomes = labels * 2 + predictions  # 0=TN, 1=FP, 2=FN, 3=TP
    combined = group_codes * 4 + outcomes
    counts = np.bincount(
        combined, minlength=group_count * 4
    ).reshape(group_count, 4)

    tn, fp, fn, tp = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    positives = tp + fn
    negatives = tn + fp

    fnr = np.divide(
        fn, positives,
        out=np.full(group_count, np.nan, dtype=float),
        where=positives > 0,
    )
    fpr = np.divide(
        fp, negatives,
        out=np.full(group_count, np.nan, dtype=float),
        where=negatives > 0,
    )
    return {"FNR": fnr, "FPR": fpr}


def rank_groups(values):
    """Use the same rounded values for average ranks and worst-group ties."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Undefined group metric encountered in identity permutation.")

    tie_values = np.round(values, decimals=TIE_DECIMALS)
    ranks = rankdata(-tie_values, method="average")
    worst_indices = np.flatnonzero(tie_values == tie_values.max())
    return ranks, worst_indices


def transition_stats(values_from, values_to, group_names):
    ranks_from, worst_from = rank_groups(values_from)
    ranks_to, worst_to = rank_groups(values_to)

    tau_result = kendalltau(ranks_from, ranks_to, variant="b")
    tau = float(tau_result.statistic) if np.isfinite(tau_result.statistic) else np.nan

    set_from = set(worst_from.tolist())
    set_to = set(worst_to.tolist())
    worst_changed = int(set_from != set_to)

    gap_from = float(np.max(values_from) - np.min(values_from))
    gap_to = float(np.max(values_to) - np.min(values_to))
    signed_change = gap_to - gap_from

    return {
        "worst_group_from": "|".join(str(group_names[i]) for i in worst_from),
        "worst_group_to": "|".join(str(group_names[i]) for i in worst_to),
        "worst_group_from_count": int(len(worst_from)),
        "worst_group_to_count": int(len(worst_to)),
        "worst_group_set_changed": worst_changed,
        "kendall_tau_b": tau,
        "ranking_instability": 1.0 - tau if np.isfinite(tau) else np.nan,
        "ranking_instability_normalized": (
            (1.0 - tau) / 2.0 if np.isfinite(tau) else np.nan
        ),
        "gap_from": gap_from,
        "gap_to": gap_to,
        "signed_gap_change": signed_change,
        "magnitude_change": abs(signed_change),
    }


def empirical_upper(null_values, observed):
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    return float((1 + np.sum(values >= observed)) / (len(values) + 1))


def empirical_lower(null_values, observed):
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    return float((1 + np.sum(values <= observed)) / (len(values) + 1))


def empirical_two_sided(null_values, observed):
    p_upper = empirical_upper(null_values, observed)
    p_lower = empirical_lower(null_values, observed)
    return min(1.0, 2.0 * min(p_upper, p_lower))


def holm_adjust(p_values):
    """Holm step-down adjusted p-values; no statsmodels dependency required."""
    p_values = np.asarray(p_values, dtype=float)
    if not np.isfinite(p_values).all():
        raise ValueError("Holm correction received non-finite p-values.")
    count = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    factors = count - np.arange(count)
    adjusted_sorted = np.maximum.accumulate(factors * sorted_p)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def build_transitions_for_assignment(
    labels, scores_by_model, group_codes, group_names, source, permutation_id
):
    rows = []
    for model in MODELS:
        threshold_metrics = {
            threshold: calculate_metrics(
                labels, scores_by_model[model], group_codes,
                len(group_names), threshold
            )
            for threshold in THRESHOLDS
        }

        for metric in METRICS:
            for t_from, t_to in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
                stats = transition_stats(
                    threshold_metrics[t_from][metric],
                    threshold_metrics[t_to][metric],
                    group_names,
                )
                rows.append({
                    "source": source,
                    "permutation": permutation_id,
                    "model": model,
                    "metric": metric,
                    "threshold_from": t_from,
                    "threshold_to": t_to,
                    **stats,
                })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required_columns = [
        FOLD_COLUMN, IDENTITY_COLUMN, LABEL_COLUMN, GROUP_COLUMN, *MODELS
    ]
    print(f"Loading {BFW_DATA_PATH} ...")
    df = pd.read_csv(BFW_DATA_PATH, usecols=required_columns)

    if df[required_columns].isna().any().any():
        missing = df[required_columns].isna().sum()
        raise ValueError(f"Missing values found:\n{missing[missing > 0]}")

    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int8)
    unexpected_labels = sorted(set(np.unique(labels)) - {0, 1})
    if unexpected_labels:
        raise ValueError(f"Expected labels 0 and 1; found {unexpected_labels}")

    group_names = np.sort(df[GROUP_COLUMN].unique())
    group_to_code = {name: i for i, name in enumerate(group_names)}
    original_group_codes = df[GROUP_COLUMN].map(group_to_code).to_numpy(dtype=np.int16)

    scores_by_model = {
        model: df[model].to_numpy(dtype=np.float64) for model in MODELS
    }

    # Build and validate the identity-level demographic table.
    identity_table = (
        df[[IDENTITY_COLUMN, FOLD_COLUMN, GROUP_COLUMN]]
        .drop_duplicates()
        .copy()
    )
    if identity_table[IDENTITY_COLUMN].duplicated().any():
        raise ValueError(
            "An id1 identity maps to multiple folds or demographic groups. "
            "Identity-level permutation is not valid with this mapping."
        )

    identity_table = identity_table.sort_values(
        [FOLD_COLUMN, IDENTITY_COLUMN]
    ).reset_index(drop=True)

    identity_table["group_code"] = (
        identity_table[GROUP_COLUMN].map(group_to_code).astype(np.int16)
    )
    identity_to_position = {
        identity: position
        for position, identity in enumerate(identity_table[IDENTITY_COLUMN])
    }
    row_identity_positions = (
        df[IDENTITY_COLUMN].map(identity_to_position).to_numpy(dtype=np.int32)
    )

    identity_counts = (
        identity_table.groupby([FOLD_COLUMN, GROUP_COLUMN])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=group_names, fill_value=0)
    )

    # Require a complete, balanced identity allocation within each fold.
    if identity_table[IDENTITY_COLUMN].nunique() != 800:
        raise ValueError(
            "Expected 800 unique id1 identities based on the inspected BFW "
            f"structure; found {identity_table[IDENTITY_COLUMN].nunique()}."
        )
    if not (identity_counts == identity_counts.iloc[0, 0]).all().all():
        raise ValueError(
            "Identity demographic counts are not balanced within every fold. "
            "Review the allocation before running this permutation design."
        )

    folds = np.sort(identity_table[FOLD_COLUMN].unique())
    if len(folds) != 5:
        raise ValueError(f"Expected five BFW folds; found {len(folds)}.")

    identity_group_codes = identity_table["group_code"].to_numpy(dtype=np.int16)
    identity_fold_values = identity_table[FOLD_COLUMN].to_numpy()
    fold_identity_positions = {
        fold: np.flatnonzero(identity_fold_values == fold) for fold in folds
    }

    dataset_hash = sha256_file(Path(BFW_DATA_PATH))
    config = {
        "dataset_sha256": dataset_hash,
        "rows": int(len(df)),
        "identity_count": int(len(identity_table)),
        "models": MODELS,
        "group_column": GROUP_COLUMN,
        "identity_column": IDENTITY_COLUMN,
        "fold_column": FOLD_COLUMN,
        "thresholds": THRESHOLDS,
        "metrics": METRICS,
        "permutations": PERMUTATION_REPETITIONS,
        "seed": RANDOM_SEED,
        "design": "permute identity demographic labels within fold",
    }
    config_fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    print(f"Rows: {len(df):,}")
    print(f"Identities: {len(identity_table):,}")
    print(f"Folds: {list(folds)}")
    print(f"Groups: {list(group_names)}")
    print("Identity counts per fold and group:")
    print(identity_counts)
    print(f"Permutations: {PERMUTATION_REPETITIONS:,}")
    print("Null: permute identity-level group assignments within each fold.")
    print("The same permuted identity assignments are reused across all models and thresholds.")

    # Save an audit of the identity structure.
    identity_audit_path = OUTPUT_DIR / f"{PREFIX}_identity_audit.csv"
    identity_audit = (
        identity_table.groupby([FOLD_COLUMN, GROUP_COLUMN])
        .agg(identity_count=(IDENTITY_COLUMN, "nunique"))
        .reset_index()
    )
    identity_audit.to_csv(identity_audit_path, index=False)

    # Full-data observed reference.
    observed_rows = build_transitions_for_assignment(
        labels, scores_by_model, original_group_codes,
        group_names, "observed_full_data", -1
    )
    observed_df = pd.DataFrame(observed_rows)
    observed_path = OUTPUT_DIR / f"{PREFIX}_observed_transitions.csv"
    observed_df.to_csv(observed_path, index=False)

    checkpoint_path = OUTPUT_DIR / f"{PREFIX}_null_checkpoint.csv"
    state_path = OUTPUT_DIR / f"{PREFIX}_checkpoint_state.json"
    null_path = OUTPUT_DIR / f"{PREFIX}_null_transitions.csv"
    summary_path = OUTPUT_DIR / f"{PREFIX}_summary.csv"

    rng = np.random.default_rng(RANDOM_SEED)
    completed = 0

    # Resume only when both checkpoint and state match this exact dataset/config.
    if checkpoint_path.exists() and state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)

        if state.get("config_fingerprint") == config_fingerprint:
            completed = int(state["completed_permutations"])
            rng.bit_generator.state = state["rng_state"]
            checkpoint = pd.read_csv(checkpoint_path)
            checkpoint = checkpoint[
                checkpoint["permutation"] <= completed
            ]
            expected_rows = completed * len(MODELS) * len(METRICS) * (len(THRESHOLDS) - 1)
            if len(checkpoint) != expected_rows:
                raise ValueError(
                    "Checkpoint row count does not match its saved progress. "
                    "Move the checkpoint files aside and start a fresh run."
                )
            # If interruption occurred after writing the CSV batch but before
            # updating the state file, discard rows beyond the saved RNG state.
            checkpoint.to_csv(checkpoint_path, index=False)
            print(f"Resuming after {completed:,} complete permutations.")
        else:
            raise ValueError(
                "Checkpoint configuration or dataset fingerprint differs. "
                "Move the old checkpoint files aside before starting a new run."
            )
    elif checkpoint_path.exists() or state_path.exists():
        raise ValueError(
            "Only one checkpoint component exists. Move the incomplete "
            "checkpoint files aside before starting a fresh run."
        )

    batch_rows = []
    for permutation_id in range(completed + 1, PERMUTATION_REPETITIONS + 1):
        permuted_identity_groups = identity_group_codes.copy()

        # Permute group labels among identities within each original fold.
        # This preserves identity-level group counts within every fold.
        for fold in folds:
            positions = fold_identity_positions[fold]
            permuted_identity_groups[positions] = rng.permutation(
                identity_group_codes[positions]
            )

        # Each comparison receives the permuted group of its first identity.
        permuted_row_groups = permuted_identity_groups[row_identity_positions]

        rows = build_transitions_for_assignment(
            labels, scores_by_model, permuted_row_groups,
            group_names, "identity_permutation_null", permutation_id
        )
        batch_rows.extend(rows)

        if permutation_id % CHECKPOINT_EVERY == 0:
            batch_df = pd.DataFrame(batch_rows)
            write_header = not checkpoint_path.exists()
            batch_df.to_csv(
                checkpoint_path,
                mode="a",
                header=write_header,
                index=False,
            )
            batch_rows.clear()

            state = {
                "config_fingerprint": config_fingerprint,
                "completed_permutations": permutation_id,
                "rng_state": rng.bit_generator.state,
            }
            temporary_state_path = state_path.with_suffix(".tmp")
            with temporary_state_path.open("w", encoding="utf-8") as handle:
                json.dump(state, handle)
            temporary_state_path.replace(state_path)

            print(
                f"Completed {permutation_id:,}/{PERMUTATION_REPETITIONS:,} "
                "identity permutations."
            )
            gc.collect()

    # Save any final partial batch.
    if batch_rows:
        batch_df = pd.DataFrame(batch_rows)
        write_header = not checkpoint_path.exists()
        batch_df.to_csv(
            checkpoint_path,
            mode="a",
            header=write_header,
            index=False,
        )
        batch_rows.clear()

    # Final checkpoint state.
    state = {
        "config_fingerprint": config_fingerprint,
        "completed_permutations": PERMUTATION_REPETITIONS,
        "rng_state": rng.bit_generator.state,
    }
    temporary_state_path = state_path.with_suffix(".tmp")
    with temporary_state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle)
    temporary_state_path.replace(state_path)

    null_df = pd.read_csv(checkpoint_path)
    expected_null_rows = (
        PERMUTATION_REPETITIONS * len(MODELS) * len(METRICS)
        * (len(THRESHOLDS) - 1)
    )
    if len(null_df) != expected_null_rows:
        raise ValueError(
            f"Expected {expected_null_rows:,} null rows; found {len(null_df):,}."
        )
    if null_df.duplicated(
        ["permutation", "model", "metric", "threshold_from", "threshold_to"]
    ).any():
        raise ValueError("Duplicate permutation transition records detected.")

    null_df.to_csv(null_path, index=False)

    # Compare observed transitions against the matching identity-level null.
    summary_rows = []
    test_columns = [
        "worst_group_set_changed",
        "kendall_tau_b",
        "magnitude_change",
    ]

    for _, observed in observed_df.iterrows():
        subset = null_df[
            (null_df["model"] == observed["model"])
            & (null_df["metric"] == observed["metric"])
            & (null_df["threshold_from"] == observed["threshold_from"])
            & (null_df["threshold_to"] == observed["threshold_to"])
        ]

        row = {
            "model": observed["model"],
            "metric": observed["metric"],
            "threshold_from": observed["threshold_from"],
            "threshold_to": observed["threshold_to"],
            "observed_worst_group_from": observed["worst_group_from"],
            "observed_worst_group_to": observed["worst_group_to"],
            "observed_worst_group_set_changed": observed["worst_group_set_changed"],
            "observed_kendall_tau_b": observed["kendall_tau_b"],
            "observed_ranking_instability": observed["ranking_instability"],
            "observed_ranking_instability_normalized": observed[
                "ranking_instability_normalized"
            ],
            "observed_gap_from": observed["gap_from"],
            "observed_gap_to": observed["gap_to"],
            "observed_signed_gap_change": observed["signed_gap_change"],
            "observed_magnitude_change": observed["magnitude_change"],
            "null_permutations": int(len(subset)),
        }

        for column in test_columns:
            values = subset[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"null_{column}_mean"] = float(np.mean(finite))
            row[f"null_{column}_sd"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
            )
            row[f"null_{column}_q025"] = float(np.quantile(finite, 0.025))
            row[f"null_{column}_median"] = float(np.quantile(finite, 0.50))
            row[f"null_{column}_q975"] = float(np.quantile(finite, 0.975))
            row[f"null_{column}_n_defined"] = int(len(finite))

        worst_values = subset["worst_group_set_changed"].to_numpy(dtype=float)
        tau_values = subset["kendall_tau_b"].to_numpy(dtype=float)
        magnitude_values = subset["magnitude_change"].to_numpy(dtype=float)

        row["empirical_p_worst_change_upper"] = empirical_upper(
            worst_values, observed["worst_group_set_changed"]
        )
        row["empirical_p_worst_change_lower"] = empirical_lower(
            worst_values, observed["worst_group_set_changed"]
        )
        row["empirical_p_worst_change_two_sided"] = empirical_two_sided(
            worst_values, observed["worst_group_set_changed"]
        )
        row["empirical_p_low_kendall_tau_b"] = empirical_lower(
            tau_values, observed["kendall_tau_b"]
        )
        row["empirical_p_magnitude_change_upper"] = empirical_upper(
            magnitude_values, observed["magnitude_change"]
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Holm correction across the 48 model/metric/threshold-transition tests
    # for each inferential endpoint.
    p_columns = [
        "empirical_p_worst_change_upper",
        "empirical_p_worst_change_lower",
        "empirical_p_worst_change_two_sided",
        "empirical_p_low_kendall_tau_b",
        "empirical_p_magnitude_change_upper",
    ]
    for column in p_columns:
        summary_df[f"holm_{column}"] = holm_adjust(
            summary_df[column].to_numpy(dtype=float)
        )

    summary_df.to_csv(summary_path, index=False)

    print("\nSaved outputs:")
    print(identity_audit_path)
    print(observed_path)
    print(null_path)
    print(summary_path)

    print("\nValidation:")
    print(f"Null transition rows: {len(null_df):,}")
    print(f"Expected null transition rows: {expected_null_rows:,}")
    print(f"Summary rows: {len(summary_df)}")
    print(
        "Magnitude tests significant after Holm correction:",
        int((summary_df["holm_empirical_p_magnitude_change_upper"] < 0.05).sum()),
        "/",
        len(summary_df),
    )
    print(
        "Worst-group two-sided tests significant after Holm correction:",
        int((summary_df["holm_empirical_p_worst_change_two_sided"] < 0.05).sum()),
        "/",
        len(summary_df),
    )
    print(
        "Low-Kendall-tau tests significant after Holm correction:",
        int((summary_df["holm_empirical_p_low_kendall_tau_b"] < 0.05).sum()),
        "/",
        len(summary_df),
    )

    print("\nInterpretation note:")
    print(
        "This is an identity-level permutation sensitivity analysis. "
        "Demographic group labels are permuted among identities within each "
        "predefined fold, preserving identity-level group counts per fold."
    )
    print(
        "Each comparison is assigned the permuted group of its first identity "
        "(id1), matching the project's a1 grouping convention."
    )
    print(
        "This preserves repeated id1 assignments and fold structure, but it is "
        "a different null hypothesis from the earlier row-level, label-stratified "
        "permutation. Compare the two null designs; do not treat them as "
        "interchangeable."
    )
    print(
        "The design does not fully model dependence through second identities "
        "(id2), nor does it establish population-level causal inference."
    )


if __name__ == "__main__":
    main()
