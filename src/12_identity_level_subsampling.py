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
# Experiment 12: identity-level induced-subgraph subsampling
# ---------------------------------------------------------------------
MODELS = ["vgg16", "resnet50", "senet50"]
GROUPS = ["AF", "AM", "BF", "BM", "IF", "IM", "WF", "WM"]
GROUP_COLUMN = "a1"  # Preserve the established first-endpoint audit convention.
SECOND_GROUP_COLUMN = "a2"
ID1_COLUMN = "id1"
ID2_COLUMN = "id2"
FOLD_COLUMN = "fold"
LABEL_COLUMN = "label"
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
METRICS = ["FNR", "FPR"]

# Each original fold contains 20 identities in every demographic group.
# Select exactly k identities from each fold x demographic-group stratum.
IDENTITIES_PER_STRATUM = [10, 13, 16, 18]
REPETITIONS = 500
RANDOM_SEED = 20261103
TIE_DECIMALS = 12
PREFIX = "12_identity_subsampling"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def group_metrics(labels, scores, group_codes, n_groups, threshold):
    pred = (scores >= threshold).astype(np.int8)
    outcome = labels * 2 + pred  # 0=TN, 1=FP, 2=FN, 3=TP
    counts = np.bincount(
        group_codes * 4 + outcome, minlength=n_groups * 4
    ).reshape(n_groups, 4)
    tn, fp, fn, tp = counts.T
    positives = tp + fn
    negatives = tn + fp
    fnr = np.divide(fn, positives, out=np.full(n_groups, np.nan), where=positives > 0)
    fpr = np.divide(fp, negatives, out=np.full(n_groups, np.nan), where=negatives > 0)
    return {"FNR": fnr, "FPR": fpr, "counts": counts}


def rank_and_worst(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("A subgroup metric is undefined; check retained pair counts.")
    rounded = np.round(values, TIE_DECIMALS)
    ranks = rankdata(-rounded, method="average")
    worst = np.flatnonzero(rounded == rounded.max())
    return ranks, worst


def gap(values):
    return float(np.max(values) - np.min(values))


def transition_summary(values_from, values_to):
    ranks_from, worst_from = rank_and_worst(values_from)
    ranks_to, worst_to = rank_and_worst(values_to)
    tau = kendalltau(ranks_from, ranks_to, variant="b").statistic
    return {
        "gap_from": gap(values_from),
        "gap_to": gap(values_to),
        "signed_gap_change": gap(values_to) - gap(values_from),
        "absolute_gap_change": abs(gap(values_to) - gap(values_from)),
        "kendall_tau_b_between_thresholds": float(tau) if np.isfinite(tau) else np.nan,
        "ranking_instability_normalized": (1.0 - float(tau)) / 2.0 if np.isfinite(tau) else np.nan,
        "worst_group_from": "|".join(GROUPS[i] for i in worst_from),
        "worst_group_to": "|".join(GROUPS[i] for i in worst_to),
        "worst_group_set_changed": int(set(worst_from.tolist()) != set(worst_to.tolist())),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    needed = [FOLD_COLUMN, ID1_COLUMN, ID2_COLUMN, GROUP_COLUMN,
              SECOND_GROUP_COLUMN, LABEL_COLUMN, *MODELS]
    print(f"Loading {BFW_DATA_PATH}")
    df = pd.read_csv(BFW_DATA_PATH, usecols=needed)
    if df[needed].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int8)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must be binary 0/1.")
    group_names = sorted(df[GROUP_COLUMN].unique().tolist())
    if group_names != GROUPS:
        raise ValueError(f"Unexpected group set: {group_names}")
    group_to_code = {g: i for i, g in enumerate(GROUPS)}
    group_codes = df[GROUP_COLUMN].map(group_to_code).to_numpy(dtype=np.int16)
    folds = sorted(df[FOLD_COLUMN].unique().tolist())

    # Build a canonical identity table from BOTH endpoints, and verify that
    # every identity has one fold and one demographic group.
    left = df[[ID1_COLUMN, FOLD_COLUMN, GROUP_COLUMN]].rename(
        columns={ID1_COLUMN: "identity", GROUP_COLUMN: "group"}
    )
    right = df[[ID2_COLUMN, FOLD_COLUMN, SECOND_GROUP_COLUMN]].rename(
        columns={ID2_COLUMN: "identity", SECOND_GROUP_COLUMN: "group"}
    )
    identity_table = pd.concat([left, right], ignore_index=True).drop_duplicates()
    if identity_table["identity"].duplicated().any():
        raise ValueError("An identity maps to multiple folds or demographic groups.")
    identity_table = identity_table.sort_values([FOLD_COLUMN, "group", "identity"]).reset_index(drop=True)
    if identity_table["identity"].nunique() != 800:
        raise ValueError(f"Expected 800 identities; found {identity_table['identity'].nunique()}")

    stratum_sizes = identity_table.groupby([FOLD_COLUMN, "group"]).size()
    if not (stratum_sizes == 20).all() or len(stratum_sizes) != 40:
        raise ValueError("Expected exactly 20 identities in each of 5 x 8 strata.")

    id_to_pos = {identity: i for i, identity in enumerate(identity_table["identity"].tolist())}
    id1_pos = df[ID1_COLUMN].map(id_to_pos).to_numpy(dtype=np.int32)
    id2_pos = df[ID2_COLUMN].map(id_to_pos).to_numpy(dtype=np.int32)
    if not np.array_equal(
        identity_table.iloc[id1_pos][FOLD_COLUMN].to_numpy(),
        df[FOLD_COLUMN].to_numpy()
    ):
        raise ValueError("id1 fold mapping inconsistency.")
    if not np.array_equal(
        identity_table.iloc[id2_pos][FOLD_COLUMN].to_numpy(),
        df[FOLD_COLUMN].to_numpy()
    ):
        raise ValueError("id2 fold mapping inconsistency; cross-fold edges detected.")

    scores = {m: df[m].to_numpy(dtype=np.float64) for m in MODELS}
    n_groups = len(GROUPS)
    n_rows = len(df)
    n_ids = len(identity_table)

    # Full-data reference: group metrics at every threshold and derived transitions.
    reference_group_rows = []
    reference_transition_rows = []
    reference_vectors = {}
    for model in MODELS:
        per_threshold = {}
        for threshold in THRESHOLDS:
            result = group_metrics(labels, scores[model], group_codes, n_groups, threshold)
            per_threshold[threshold] = result
            for metric in METRICS:
                vals = result[metric]
                ranks, worst = rank_and_worst(vals)
                for gi, group in enumerate(GROUPS):
                    reference_group_rows.append({
                        "model": model, "threshold": threshold, "metric": metric,
                        "group": group, "value": float(vals[gi]),
                        "rank": float(ranks[gi]), "is_worst": int(gi in worst),
                    })
                reference_vectors[(model, metric, threshold)] = vals.copy()
        for metric in METRICS:
            for t_from, t_to in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
                reference_transition_rows.append({
                    "model": model, "metric": metric,
                    "threshold_from": t_from, "threshold_to": t_to,
                    **transition_summary(per_threshold[t_from][metric], per_threshold[t_to][metric]),
                })

    reference_group_df = pd.DataFrame(reference_group_rows)
    reference_transition_df = pd.DataFrame(reference_transition_rows)
    reference_group_df.to_csv(OUTPUT_DIR / f"{PREFIX}_full_data_reference_groups.csv", index=False)
    reference_transition_df.to_csv(OUTPUT_DIR / f"{PREFIX}_full_data_reference_transitions.csv", index=False)

    # Precompute positions in each fold x group stratum.
    strata = {}
    id_folds = identity_table[FOLD_COLUMN].to_numpy()
    id_groups = identity_table["group"].to_numpy()
    for fold in folds:
        for group in GROUPS:
            strata[(fold, group)] = np.flatnonzero((id_folds == fold) & (id_groups == group))

    rng = np.random.default_rng(RANDOM_SEED)
    audit_rows = []
    baseline_rows = []
    transition_rows = []
    total_runs = len(IDENTITIES_PER_STRATUM) * REPETITIONS
    run_counter = 0

    print(f"Rows: {n_rows:,}; identities: {n_ids}; folds: {len(folds)}")
    print(f"Running {REPETITIONS} repetitions at each identity sample size: {IDENTITIES_PER_STRATUM}")

    for k in IDENTITIES_PER_STRATUM:
        for rep in range(1, REPETITIONS + 1):
            selected = np.zeros(n_ids, dtype=bool)
            for key, positions in strata.items():
                chosen = rng.choice(positions, size=k, replace=False)
                selected[chosen] = True

            # Induced-subgraph rule: retain a comparison only when BOTH identities
            # participating in it are selected. This preserves shared-node structure.
            keep = selected[id1_pos] & selected[id2_pos]
            retained = int(keep.sum())
            if retained == 0:
                raise ValueError("Identity subsample retained no comparison rows.")

            sampled_labels = labels[keep]
            sampled_groups = group_codes[keep]
            selected_count = int(selected.sum())
            run_counter += 1
            audit_rows.append({
                "identities_per_fold_group": k,
                "identity_fraction": k / 20.0,
                "repetition": rep,
                "selected_identities": selected_count,
                "source_rows": n_rows,
                "retained_comparisons": retained,
                "retained_row_fraction": retained / n_rows,
                "genuine_comparisons": int(np.sum(sampled_labels == 1)),
                "impostor_comparisons": int(np.sum(sampled_labels == 0)),
            })

            for model in MODELS:
                sampled_scores = scores[model][keep]
                per_threshold = {
                    t: group_metrics(sampled_labels, sampled_scores, sampled_groups, n_groups, t)
                    for t in THRESHOLDS
                }

                # Baseline .50: compare identity-subsample group gap, ranking and
                # worst-group membership against the full-data reference.
                for metric in METRICS:
                    vals = per_threshold[0.50][metric]
                    ranks, worst = rank_and_worst(vals)
                    ref_vals = reference_vectors[(model, metric, 0.50)]
                    ref_ranks, ref_worst = rank_and_worst(ref_vals)
                    tau_ref = kendalltau(ranks, ref_ranks, variant="b").statistic
                    baseline_rows.append({
                        "identities_per_fold_group": k,
                        "identity_fraction": k / 20.0,
                        "repetition": rep,
                        "model": model,
                        "metric": metric,
                        "group_gap": gap(vals),
                        "full_data_group_gap": gap(ref_vals),
                        "absolute_gap_error_vs_full_data": abs(gap(vals) - gap(ref_vals)),
                        "kendall_tau_b_vs_full_data_ranking": float(tau_ref) if np.isfinite(tau_ref) else np.nan,
                        "normalized_ranking_disagreement_vs_full_data": (1.0 - float(tau_ref)) / 2.0 if np.isfinite(tau_ref) else np.nan,
                        "worst_group_set": "|".join(GROUPS[i] for i in worst),
                        "full_data_worst_group_set": "|".join(GROUPS[i] for i in ref_worst),
                        "worst_group_set_matches_full_data": int(set(worst.tolist()) == set(ref_worst.tolist())),
                    })

                    for t_from, t_to in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
                        sampled_transition = transition_summary(
                            per_threshold[t_from][metric], per_threshold[t_to][metric]
                        )
                        ref_transition = reference_transition_df[
                            (reference_transition_df.model == model)
                            & (reference_transition_df.metric == metric)
                            & (reference_transition_df.threshold_from == t_from)
                            & (reference_transition_df.threshold_to == t_to)
                        ].iloc[0]
                        transition_rows.append({
                            "identities_per_fold_group": k,
                            "identity_fraction": k / 20.0,
                            "repetition": rep,
                            "model": model,
                            "metric": metric,
                            "threshold_from": t_from,
                            "threshold_to": t_to,
                            **sampled_transition,
                            "full_data_absolute_gap_change": float(ref_transition.absolute_gap_change),
                            "absolute_gap_change_error_vs_full_data": abs(
                                sampled_transition["absolute_gap_change"]
                                - float(ref_transition.absolute_gap_change)
                            ),
                            "full_data_worst_group_set_changed": int(ref_transition.worst_group_set_changed),
                            "worst_group_transition_matches_full_data": int(
                                sampled_transition["worst_group_set_changed"]
                                == int(ref_transition.worst_group_set_changed)
                            ),
                        })

            if rep % 50 == 0:
                print(f"  k={k:2d}/20, repetition {rep}/{REPETITIONS} complete")

        print(f"Completed identity fraction {k/20:.0%} ({k} identities per fold-group).")
        gc.collect()

    audit_df = pd.DataFrame(audit_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    transition_df = pd.DataFrame(transition_rows)

    # Summaries are distributions over identity-level induced-subgraph samples,
    # NOT conventional bootstrap confidence intervals.
    baseline_summary = baseline_df.groupby(
        ["identities_per_fold_group", "identity_fraction", "model", "metric"], as_index=False
    ).agg(
        repetitions=("repetition", "count"),
        group_gap_mean=("group_gap", "mean"),
        group_gap_sd=("group_gap", "std"),
        group_gap_p2_5=("group_gap", lambda x: x.quantile(0.025)),
        group_gap_median=("group_gap", "median"),
        group_gap_p97_5=("group_gap", lambda x: x.quantile(0.975)),
        mean_absolute_gap_error=("absolute_gap_error_vs_full_data", "mean"),
        mean_tau_vs_full_data=("kendall_tau_b_vs_full_data_ranking", "mean"),
        mean_normalized_ranking_disagreement=("normalized_ranking_disagreement_vs_full_data", "mean"),
        worst_group_match_frequency=("worst_group_set_matches_full_data", "mean"),
    )
    transition_summary_df = transition_df.groupby(
        ["identities_per_fold_group", "identity_fraction", "model", "metric", "threshold_from", "threshold_to"],
        as_index=False,
    ).agg(
        repetitions=("repetition", "count"),
        absolute_gap_change_mean=("absolute_gap_change", "mean"),
        absolute_gap_change_sd=("absolute_gap_change", "std"),
        absolute_gap_change_p2_5=("absolute_gap_change", lambda x: x.quantile(0.025)),
        absolute_gap_change_median=("absolute_gap_change", "median"),
        absolute_gap_change_p97_5=("absolute_gap_change", lambda x: x.quantile(0.975)),
        mean_absolute_gap_change_error=("absolute_gap_change_error_vs_full_data", "mean"),
        mean_tau_between_thresholds=("kendall_tau_b_between_thresholds", "mean"),
        worst_group_transition_match_frequency=("worst_group_transition_matches_full_data", "mean"),
        worst_group_set_changed_frequency=("worst_group_set_changed", "mean"),
    )

    # Compare the .50 full-data row-bootstrap gap SD from Program 08 with the
    # identity-level induced-subgraph gap SD here. This is a diagnostic comparison,
    # not a claim that the two resampling distributions estimate identical targets.
    row_boot_path = OUTPUT_DIR / "bootstrap_baseline_summary_all_models.csv"
    if row_boot_path.exists():
        row_boot = pd.read_csv(row_boot_path)
        baseline_summary = baseline_summary.merge(
            row_boot[["model", "metric", "group_gap_sd", "group_gap_p2_5", "group_gap_p97_5"]]
            .rename(columns={
                "group_gap_sd": "row_bootstrap_group_gap_sd",
                "group_gap_p2_5": "row_bootstrap_group_gap_p2_5",
                "group_gap_p97_5": "row_bootstrap_group_gap_p97_5",
            }),
            on=["model", "metric"], how="left",
        )
        baseline_summary["identity_to_row_gap_sd_ratio"] = (
            baseline_summary["group_gap_sd"] / baseline_summary["row_bootstrap_group_gap_sd"]
        )

    audit_df.to_csv(OUTPUT_DIR / f"{PREFIX}_split_audit.csv", index=False)
    baseline_df.to_csv(OUTPUT_DIR / f"{PREFIX}_baseline_repetitions.csv", index=False)
    transition_df.to_csv(OUTPUT_DIR / f"{PREFIX}_transition_repetitions.csv", index=False)
    baseline_summary.to_csv(OUTPUT_DIR / f"{PREFIX}_baseline_summary.csv", index=False)
    transition_summary_df.to_csv(OUTPUT_DIR / f"{PREFIX}_transition_summary.csv", index=False)

    manifest = {
        "experiment": "Identity-level induced-subgraph subsampling sensitivity analysis",
        "dataset_path": str(BFW_DATA_PATH),
        "dataset_sha256": sha256_file(BFW_DATA_PATH),
        "source_rows": n_rows,
        "unique_identities": n_ids,
        "folds": folds,
        "groups": GROUPS,
        "models": MODELS,
        "thresholds": THRESHOLDS,
        "metrics": METRICS,
        "identities_per_fold_group": IDENTITIES_PER_STRATUM,
        "repetitions_per_size": REPETITIONS,
        "random_seed": RANDOM_SEED,
        "sampling_design": "Within each original fold x demographic-group stratum, sample k of 20 identities without replacement; retain a comparison only when both endpoint identities are selected.",
        "estimand_note": "This is identity-level induced-subgraph subsampling sensitivity analysis. Its quantiles are not presented as conventional bootstrap confidence intervals or population-level inferential intervals.",
        "a1_convention": "Subgroup metrics use a1 (the first endpoint group), consistent with the established AuditFlux audit convention; both id1 and id2 must be selected for a row to remain.",
        "limitations": [
            "Sampling is conditional on the five predefined folds and fixed demographic counts within each fold.",
            "The induced-subgraph design preserves shared-identity dependence among retained comparisons but does not by itself establish a universally valid confidence interval procedure.",
            "The first-endpoint subgroup convention is retained for comparability with prior experiments.",
            "Results describe sensitivity to identity composition within this BFW table, not external-dataset generalization."
        ],
        "outputs": [
            f"{PREFIX}_full_data_reference_groups.csv",
            f"{PREFIX}_full_data_reference_transitions.csv",
            f"{PREFIX}_split_audit.csv",
            f"{PREFIX}_baseline_repetitions.csv",
            f"{PREFIX}_transition_repetitions.csv",
            f"{PREFIX}_baseline_summary.csv",
            f"{PREFIX}_transition_summary.csv",
            f"{PREFIX}_manifest.json",
        ],
    }
    with (OUTPUT_DIR / f"{PREFIX}_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nEXPERIMENT 12 COMPLETE")
    print(f"Identity subsamples: {len(audit_df):,}")
    print(f"Baseline repetition records: {len(baseline_df):,}")
    print(f"Transition repetition records: {len(transition_df):,}")
    print("Outputs saved to:", OUTPUT_DIR)
    print("\nBaseline summary:")
    print(baseline_summary.to_string(index=False))
    print("\nTransition summary preview:")
    print(transition_summary_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
