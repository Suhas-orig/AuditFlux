"""AuditFlux Program 19: matched-size row vs identity-induced subsampling.

Research question
-----------------
When an identity-induced BFW subsample is smaller than the full comparison
 table, how much of its observed audit variability is associated with the
 reduced number/composition of comparison rows, and how much additional
 variability is associated with retaining the identity-sharing structure?

Design
------
For each repetition and identity sample size k:
  1. Select k identities without replacement in every original fold x a1-group
     stratum (the established first-endpoint audit convention).
  2. Form the induced comparison subgraph: retain a row only when BOTH id1
     and id2 are selected.
  3. Count retained rows in each exact (fold, a1, a2, label) stratum.
  4. Draw a row-level comparison sample WITHOUT replacement from the original
     BFW table, matching those exact stratum counts.

Thus, each identity sample is paired with a row sample having exactly the
same total number of rows and the same fold x endpoint-demographic x label
composition. The principal design difference is the sampling unit: the identity sample
conditions on a coherent selected set of identities, whereas the comparator
samples individual rows. The sampled rows can still share identities by chance;
what is removed is the vertex-based inclusion mechanism, not all overlap.

Interpretation is conditional on the observed BFW comparison table. This is
not a population confidence-interval procedure and does not make the sampled
impostor graph complete. FNR and FPR are descriptive audit statistics at the
pre-specified threshold 0.50.

Requires: pandas, numpy, scipy. Uses the project's src/config.py.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import BFW_DATA_PATH, OUTPUT_DIR

# ----------------------------- DESIGN -----------------------------
PREFIX = "19_matched_size_comparison"
MODELS = ["vgg16", "resnet50", "senet50"]
GROUPS = ["AF", "AM", "BF", "BM", "IF", "IM", "WF", "WM"]
IDENTITIES_PER_FOLD_GROUP = [10, 13, 16, 18]
REPETITIONS = 500
THRESHOLD = 0.50
RANDOM_SEED = 20261110
TIE_DECIMALS = 12
STRATUM_COLUMNS = ["fold", "a1", "a2", "label"]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def safe_tau(values: np.ndarray, reference: np.ndarray) -> float:
    result = kendalltau(
        rankdata(-np.round(values, TIE_DECIMALS), method="average"),
        rankdata(-np.round(reference, TIE_DECIMALS), method="average"),
        variant="b",
    ).statistic
    return float(result) if np.isfinite(result) else np.nan


def summarize_metric(labels: np.ndarray, scores: np.ndarray,
                     group_codes: np.ndarray) -> dict[str, np.ndarray]:
    predictions = (scores >= THRESHOLD).astype(np.int8)
    outcomes = labels * 2 + predictions  # TN=0, FP=1, FN=2, TP=3
    counts = np.bincount(
        group_codes * 4 + outcomes, minlength=len(GROUPS) * 4
    ).reshape(len(GROUPS), 4)
    tn, fp, fn, tp = counts.T
    positives = tp + fn
    negatives = tn + fp
    fnr = np.divide(fn, positives, out=np.full(len(GROUPS), np.nan), where=positives > 0)
    fpr = np.divide(fp, negatives, out=np.full(len(GROUPS), np.nan), where=negatives > 0)
    if not (np.isfinite(fnr).all() and np.isfinite(fpr).all()):
        raise ValueError("A subgroup has no positive or negative comparisons in a sample.")
    return {"FNR": fnr, "FPR": fpr, "counts": counts}


def worst_set(values: np.ndarray) -> str:
    maximum = float(np.max(values))
    members = [GROUPS[i] for i, value in enumerate(values)
               if np.isclose(value, maximum, rtol=0, atol=10 ** (-TIE_DECIMALS))]
    return "|".join(members)


def metric_record(values: np.ndarray, reference: np.ndarray) -> dict:
    tau = safe_tau(values, reference)
    return {
        "group_gap": float(np.max(values) - np.min(values)),
        "absolute_gap_error_vs_full_data": float(abs(
            (np.max(values) - np.min(values)) - (np.max(reference) - np.min(reference))
        )),
        "kendall_tau_b_vs_full_data": tau,
        "normalized_ranking_disagreement_vs_full_data": (1.0 - tau) / 2.0 if np.isfinite(tau) else np.nan,
        "worst_group_set": worst_set(values),
        "worst_group_set_matches_full_data": int(worst_set(values) == worst_set(reference)),
    }


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    required = ["fold", "id1", "id2", "a1", "a2", "label", *MODELS]
    print(f"Loading {BFW_DATA_PATH}")
    df = pd.read_csv(BFW_DATA_PATH, usecols=required)
    if df[required].isna().any().any():
        raise ValueError("Missing values found in required columns.")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Expected binary labels 0 and 1.")
    if sorted(df["a1"].astype(str).unique().tolist()) != GROUPS:
        raise ValueError("Unexpected a1 demographic groups.")

    # Validate a single fold/group assignment for every identity using both endpoints.
    left = df[["id1", "fold", "a1"]].rename(columns={"id1": "identity", "a1": "group"})
    right = df[["id2", "fold", "a2"]].rename(columns={"id2": "identity", "a2": "group"})
    identity_table = pd.concat([left, right], ignore_index=True).drop_duplicates()
    if identity_table["identity"].duplicated().any():
        raise ValueError("An identity maps to multiple folds or demographic groups.")
    identity_table = identity_table.sort_values(["fold", "group", "identity"]).reset_index(drop=True)
    if identity_table["identity"].nunique() != 800:
        raise ValueError(f"Expected 800 identities; found {identity_table['identity'].nunique()}.")
    folds = sorted(df["fold"].unique().tolist())
    identity_stratum_sizes = identity_table.groupby(["fold", "group"]).size()
    if len(folds) != 5 or len(identity_stratum_sizes) != 40 or not (identity_stratum_sizes == 20).all():
        raise ValueError("Expected 5 folds x 8 groups x 20 identities.")

    id_to_pos = {identity: pos for pos, identity in enumerate(identity_table["identity"].tolist())}
    id1_pos = df["id1"].map(id_to_pos).to_numpy(dtype=np.int32)
    id2_pos = df["id2"].map(id_to_pos).to_numpy(dtype=np.int32)
    id_folds = identity_table["fold"].to_numpy()
    id_groups = identity_table["group"].astype(str).to_numpy()
    if not np.array_equal(id_folds[id1_pos], df["fold"].to_numpy()):
        raise ValueError("id1 fold inconsistency.")
    if not np.array_equal(id_folds[id2_pos], df["fold"].to_numpy()):
        raise ValueError("id2 fold inconsistency; cross-fold edges detected.")
    if not np.array_equal(id_groups[id1_pos], df["a1"].astype(str).to_numpy()):
        raise ValueError("id1/a1 identity mapping inconsistency.")
    if not np.array_equal(id_groups[id2_pos], df["a2"].astype(str).to_numpy()):
        raise ValueError("id2/a2 identity mapping inconsistency.")

    # Numeric arrays shared across all models; every method uses the same rows
    # for VGG16, ResNet50 and SENet50 within a repetition.
    labels = df["label"].to_numpy(dtype=np.int8)
    group_to_code = {group: i for i, group in enumerate(GROUPS)}
    group_codes = df["a1"].astype(str).map(group_to_code).to_numpy(dtype=np.int16)
    scores = {model: df[model].to_numpy(dtype=np.float64) for model in MODELS}
    n_rows = len(df)
    n_ids = len(identity_table)

    # Full-data reference at threshold .50.
    reference_vectors: dict[tuple[str, str], np.ndarray] = {}
    reference_group_rows = []
    for model in MODELS:
        ref = summarize_metric(labels, scores[model], group_codes)
        for metric in ["FNR", "FPR"]:
            values = ref[metric]
            reference_vectors[(model, metric)] = values.copy()
            for gi, group in enumerate(GROUPS):
                reference_group_rows.append({
                    "model": model, "metric": metric, "group": group,
                    "reference_value": float(values[gi]),
                    "reference_rank": float(rankdata(-np.round(values, TIE_DECIMALS), method="average")[gi]),
                    "reference_is_worst": int(group in worst_set(values).split("|")),
                    "threshold": THRESHOLD,
                })
    reference_df = pd.DataFrame(reference_group_rows)
    reference_df.to_csv(OUTPUT_DIR / f"{PREFIX}_full_data_reference.csv", index=False)

    # Precompute identity positions per fold x group.
    identity_strata = {}
    for fold in folds:
        for group in GROUPS:
            identity_strata[(fold, group)] = np.flatnonzero((id_folds == fold) & (id_groups == group))

    # Precompute source row indices per exact matching stratum. Exact matching
    # on fold, a1, a2 and label prevents composition differences from masquerading
    # as a dependence effect.
    stratum_index = pd.MultiIndex.from_frame(df[STRATUM_COLUMNS])
    row_stratum_codes, unique_strata = pd.factorize(stratum_index, sort=True)
    row_stratum_codes = row_stratum_codes.astype(np.int32)
    source_rows_by_stratum = [np.flatnonzero(row_stratum_codes == code)
                              for code in range(len(unique_strata))]
    stratum_descriptions = [tuple(item) for item in unique_strata.tolist()]

    rng = np.random.default_rng(RANDOM_SEED)
    repetition_rows = []
    group_rows = []
    audit_rows = []
    total = len(IDENTITIES_PER_FOLD_GROUP) * REPETITIONS
    run = 0
    print(f"Rows={n_rows:,}; identities={n_ids}; repetitions={REPETITIONS} per k")

    for k in IDENTITIES_PER_FOLD_GROUP:
        for repetition in range(1, REPETITIONS + 1):
            run += 1
            selected = np.zeros(n_ids, dtype=bool)
            for key, positions in identity_strata.items():
                selected[rng.choice(positions, size=k, replace=False)] = True
            identity_keep = selected[id1_pos] & selected[id2_pos]
            identity_indices = np.flatnonzero(identity_keep)
            if identity_indices.size == 0:
                raise ValueError("Identity-induced sample retained no comparison rows.")

            # Exact stratum counts of the identity-induced graph define the
            # matched row-level comparator. Sampling is without replacement.
            matched_counts = np.bincount(
                row_stratum_codes[identity_indices], minlength=len(unique_strata)
            )
            row_parts = []
            for stratum_code, count in enumerate(matched_counts):
                if count:
                    source = source_rows_by_stratum[stratum_code]
                    if count > source.size:
                        raise ValueError("Matched stratum count exceeds source population.")
                    row_parts.append(rng.choice(source, size=int(count), replace=False))
            row_indices = np.concatenate(row_parts).astype(np.int64, copy=False)
            if row_indices.size != identity_indices.size:
                raise AssertionError("Matched row sample does not have identical size.")
            row_counts_check = np.bincount(
                row_stratum_codes[row_indices], minlength=len(unique_strata)
            )
            if not np.array_equal(row_counts_check, matched_counts):
                raise AssertionError("Exact stratum matching failed.")

            selected_count = int(selected.sum())
            retained = int(identity_indices.size)
            audit_rows.append({
                "identities_per_fold_group": k,
                "identity_fraction": k / 20.0,
                "repetition": repetition,
                "selected_identities": selected_count,
                "expected_selected_identities": 40 * k,
                "identity_induced_rows": retained,
                "matched_row_rows": int(row_indices.size),
                "row_count_difference": int(row_indices.size - retained),
                "identity_induced_row_fraction": retained / n_rows,
                "matched_strata_count": int(np.count_nonzero(matched_counts)),
                "exact_fold_a1_a2_label_match": 1,
            })

            for method, indices in [("identity_induced", identity_indices),
                                    ("matched_row", row_indices)]:
                sample_labels = labels[indices]
                sample_groups = group_codes[indices]
                for model in MODELS:
                    result = summarize_metric(sample_labels, scores[model][indices], sample_groups)
                    for metric in ["FNR", "FPR"]:
                        values = result[metric]
                        reference = reference_vectors[(model, metric)]
                        stats = metric_record(values, reference)
                        repetition_rows.append({
                            "identities_per_fold_group": k,
                            "identity_fraction": k / 20.0,
                            "repetition": repetition,
                            "method": method,
                            "model": model,
                            "metric": metric,
                            "sample_rows": int(indices.size),
                            "threshold": THRESHOLD,
                            **stats,
                        })
                        for gi, group in enumerate(GROUPS):
                            group_rows.append({
                                "identities_per_fold_group": k,
                                "repetition": repetition,
                                "method": method,
                                "model": model,
                                "metric": metric,
                                "group": group,
                                "value": float(values[gi]),
                                "reference_value": float(reference[gi]),
                                "absolute_error_vs_reference": float(abs(values[gi] - reference[gi])),
                            })
            if run % 100 == 0 or run == total:
                print(f"Completed {run:,}/{total:,} identity-panel repetitions")

    repetitions_df = pd.DataFrame(repetition_rows)
    groups_df = pd.DataFrame(group_rows)
    audit_df = pd.DataFrame(audit_rows)

    # Summaries are descriptive distributions across paired resampling runs.
    summary_rows = []
    group_summary_rows = []
    keys = ["identities_per_fold_group", "method", "model", "metric"]
    for key, part in repetitions_df.groupby(keys, sort=True):
        k, method, model, metric = key
        summary_rows.append({
            "identities_per_fold_group": int(k),
            "identity_fraction": float(k / 20.0),
            "method": method,
            "model": model,
            "metric": metric,
            "repetitions": int(part["repetition"].nunique()),
            "mean_sample_rows": float(part["sample_rows"].mean()),
            "mean_group_gap": float(part["group_gap"].mean()),
            "sd_group_gap": float(part["group_gap"].std(ddof=1)),
            "gap_p2_5": float(part["group_gap"].quantile(0.025)),
            "gap_p97_5": float(part["group_gap"].quantile(0.975)),
            "mean_absolute_gap_error_vs_full_data": float(part["absolute_gap_error_vs_full_data"].mean()),
            "mean_kendall_tau_b_vs_full_data": float(part["kendall_tau_b_vs_full_data"].mean()),
            "mean_normalized_ranking_disagreement": float(part["normalized_ranking_disagreement_vs_full_data"].mean()),
            "worst_group_set_match_frequency": float(part["worst_group_set_matches_full_data"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)

    for key, part in groups_df.groupby(keys + ["group"], sort=True):
        k, method, model, metric, group = key
        group_summary_rows.append({
            "identities_per_fold_group": int(k),
            "method": method,
            "model": model,
            "metric": metric,
            "group": group,
            "mean_value": float(part["value"].mean()),
            "sd_value": float(part["value"].std(ddof=1)),
            "p2_5": float(part["value"].quantile(0.025)),
            "p97_5": float(part["value"].quantile(0.975)),
            "mean_absolute_error_vs_reference": float(part["absolute_error_vs_reference"].mean()),
        })
    group_summary_df = pd.DataFrame(group_summary_rows)

    # Paired method comparison: each identity-induced draw is paired with its
    # exact matched-row draw at the same k and repetition.
    paired_rows = []
    for (k, model, metric), part in repetitions_df.groupby(
        ["identities_per_fold_group", "model", "metric"], sort=True
    ):
        wide = part.pivot(index="repetition", columns="method", values=[
            "group_gap", "absolute_gap_error_vs_full_data",
            "normalized_ranking_disagreement_vs_full_data",
            "worst_group_set_matches_full_data",
        ])
        for outcome in ["group_gap", "absolute_gap_error_vs_full_data",
                        "normalized_ranking_disagreement_vs_full_data",
                        "worst_group_set_matches_full_data"]:
            difference = wide[(outcome, "identity_induced")] - wide[(outcome, "matched_row")]
            paired_rows.append({
                "identities_per_fold_group": int(k),
                "model": model,
                "metric": metric,
                "outcome": outcome,
                "identity_induced_mean": float(wide[(outcome, "identity_induced")].mean()),
                "matched_row_mean": float(wide[(outcome, "matched_row")].mean()),
                "mean_paired_difference_identity_minus_row": float(difference.mean()),
                "sd_paired_difference": float(difference.std(ddof=1)),
                "identity_induced_sd": float(wide[(outcome, "identity_induced")].std(ddof=1)),
                "matched_row_sd": float(wide[(outcome, "matched_row")].std(ddof=1)),
                "sd_ratio_identity_to_row": (
                    float(wide[(outcome, "identity_induced")].std(ddof=1) /
                          wide[(outcome, "matched_row")].std(ddof=1))
                    if wide[(outcome, "matched_row")].std(ddof=1) > 0 else np.nan
                ),
                "paired_repetitions": int(len(difference)),
            })

    paired_df = pd.DataFrame(paired_rows)
    audit_df.to_csv(OUTPUT_DIR / f"{PREFIX}_split_audit.csv", index=False)
    repetitions_df.to_csv(OUTPUT_DIR / f"{PREFIX}_repetitions.csv", index=False)
    groups_df.to_csv(OUTPUT_DIR / f"{PREFIX}_group_repetitions.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / f"{PREFIX}_summary.csv", index=False)
    group_summary_df.to_csv(OUTPUT_DIR / f"{PREFIX}_group_summary.csv", index=False)
    paired_df.to_csv(OUTPUT_DIR / f"{PREFIX}_paired_method_comparison.csv", index=False)

    manifest = {
        "experiment": "Program 19 - Matched-Size Row-versus-Identity Subsampling Comparison",
        "input": str(BFW_DATA_PATH),
        "input_rows": int(n_rows),
        "input_identities": int(n_ids),
        "input_sha256": sha256_file(BFW_DATA_PATH),
        "design": {
            "folds": [str(x) for x in folds],
            "demographic_groups": GROUPS,
            "identities_per_fold_group": IDENTITIES_PER_FOLD_GROUP,
            "repetitions_per_identity_sample_size": REPETITIONS,
            "threshold": THRESHOLD,
            "models": MODELS,
            "identity_method": "Select k identities without replacement within each original fold x a1-group stratum; retain a comparison only when both id1 and id2 are selected.",
            "matched_row_method": "For each identity-induced sample, sample the exact same number of rows without replacement from the original table, matching counts in every fold x a1 x a2 x label stratum.",
            "pairing": "Each identity-induced sample is paired with a matched row sample at the same k and repetition; all models use the same selected rows within each method.",
            "audit_convention": "a1 (first endpoint) is the audited demographic group; a2 is retained in matching strata.",
        },
        "primary_estimands": [
            "Across-repetition SD and percentile range of max-minus-min FNR/FPR group gaps.",
            "Absolute error of sampled group gaps relative to the full observed BFW table.",
            "Kendall tau-b ranking agreement and worst-group-set agreement with the full observed BFW reference.",
            "Paired differences between identity-induced and exact-stratum matched row samples.",
        ],
        "interpretation": {
            "main_comparison": "At equal total row count and exact fold x a1 x a2 x label composition, compare identity-panel induced-graph sampling with row-unit sampling. The contrast reflects the sampling design and induced identity composition, not a pure universal dependence parameter.",
            "resampling_percentiles": "Descriptive sensitivity distributions conditional on the observed BFW table, not population confidence intervals.",
            "sd_ratio": "Ratio of across-repetition standard deviations; descriptive and not a universal design effect.",
        },
        "limitations": [
            "The comparison is conditional on the observed BFW comparison table and its sampled impostor graph.",
            "The matched row comparator controls total row count and fold x endpoint-demographic x label composition, but samples rows as units rather than conditioning on a selected identity panel. Shared identities can still occur incidentally in the row sample.",
            "Identity-induced samples preserve dependence among retained comparisons through shared identities, but only within the induced subgraph of the observed comparisons.",
            "The BFW impostor comparison graph is sampled/incomplete; no missing comparisons are imputed.",
            "The experiment assesses sensitivity across the predefined five folds and eight demographic groups; it does not establish population-level confidence intervals or universal variance corrections.",
            "FNR and FPR are evaluated at threshold 0.50; threshold sensitivity is addressed in earlier experiments.",
            "a1 remains the audited first-endpoint group for comparability with prior AuditFlux experiments.",
        ],
        "random_seed": RANDOM_SEED,
        "outputs": {
            "full_data_reference": f"{PREFIX}_full_data_reference.csv",
            "split_audit": f"{PREFIX}_split_audit.csv",
            "repetitions": f"{PREFIX}_repetitions.csv",
            "group_repetitions": f"{PREFIX}_group_repetitions.csv",
            "summary": f"{PREFIX}_summary.csv",
            "group_summary": f"{PREFIX}_group_summary.csv",
            "paired_method_comparison": f"{PREFIX}_paired_method_comparison.csv",
        },
    }
    manifest_path = OUTPUT_DIR / f"{PREFIX}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nProgram 19 complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Elapsed: {time.time() - start:.1f} seconds")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
