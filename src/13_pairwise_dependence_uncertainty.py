"""AuditFlux Program 13: Pairwise-Dependence-Aware Uncertainty Quantification.

Part A: image-level, within-identity recentered bootstrap for genuine-pair FNR.
Part B: identity-level vertex/dyadic bootstrap for impostor-pair FPR.

IMPORTANT SCOPE:
- Part A applies the recentering identity to the complete within-identity
  genuine-pair graph available in BFW. It is not a claim that every detail of
  the published procedure has been reproduced for all verification metrics.
- Part B resamples identities within original fold x demographic group and
  weights each observed directed impostor comparison by m(id1)*m(id2).
  It is a vertex-resampling approximation for the observed, sampled impostor
  graph; it does not create missing image comparisons.
- These are two different resampling designs. Their intervals must not be
  described as interchangeable or as unconditional population guarantees.

Requires: pandas, numpy, scipy, and numba (recommended for practical runtime).
If numba is unavailable, a slower pure-Python fallback is used.
"""

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    prange = range
    def njit(*args, **kwargs):
        def decorate(func):
            return func
        return decorate

from config import BFW_DATA_PATH, OUTPUT_DIR

if not NUMBA_AVAILABLE:
    raise ImportError("Program 13 requires numba for practical runtime. Install it with: pip install numba")

# ----------------------------- CONFIG -----------------------------
MODELS = ["vgg16", "resnet50", "senet50"]
THRESHOLDS = np.array([0.50], dtype=np.float64)  # Baseline operating point; threshold sensitivity is covered by Programs 03/07/12.
BOOTSTRAP_REPETITIONS = 1000  # 1,000 repetitions for stable percentile summaries.
RANDOM_SEED = 20261103
GROUP_COLUMN = "a1"
ALPHA = 0.05

# ----------------------------- KERNELS ----------------------------
@njit(cache=True, parallel=True)
def fnr_recenter_kernel(multiplicities, pair_errors, identity_group_codes, n_groups):
    """Return recentered FNR: [rep, model_threshold, group].

    pair_errors: [identity, model_threshold, 300] binary FN indicators for
    each unique unordered genuine image pair.
    multiplicities: [rep, identity, 25] image draw counts.

    For each identity, U* excludes identical bootstrap draw positions;
    V* includes the diagonal. Since a self-comparison has similarity 1 and
    all thresholds <= .70, its FN indicator is zero.
    Recentered identity statistic = U_original + U_boot - V_original.
    The group statistic averages equally over identities. This equals the
    original pooled pair statistic because each identity has 300 pairs.
    """
    n_rep, n_identity, n_image = multiplicities.shape
    n_mt = pair_errors.shape[1]
    out = np.zeros((n_rep, n_mt, n_groups), dtype=np.float64)
    group_n = np.zeros(n_groups, dtype=np.int64)
    for i in range(n_identity):
        group_n[identity_group_codes[i]] += 1

    for r in prange(n_rep):
        for i in range(n_identity):
            g = identity_group_codes[i]
            n = n_image
            denom_u = n * (n - 1)
            denom_v = n * n
            for mt in range(n_mt):
                weighted_offdiag = 0.0
                original_sum = 0.0
                p = 0
                for a in range(n_image - 1):
                    ma = multiplicities[r, i, a]
                    for b in range(a + 1, n_image):
                        h = pair_errors[i, mt, p]
                        original_sum += h
                        weighted_offdiag += int(ma) * int(multiplicities[r, i, b]) * h
                        p += 1
                u_original = original_sum / 300.0
                u_boot = (2.0 * weighted_offdiag) / denom_u
                # IMPORTANT: recenter against the V-statistic of the ORIGINAL
                # evaluation sample, not the V-statistic of this bootstrap draw.
                # The paper's correction is U_original + U_bootstrap - V_original.
                # Using V_bootstrap here makes V_bootstrap=(n-1)/n*U_bootstrap
                # when diagonal FN indicators are zero, collapsing the bootstrap
                # deviation to approximately U_bootstrap/n and severely
                # understating image-level uncertainty.
                v_original = ((n - 1.0) / n) * u_original
                out[r, mt, g] += u_original + u_boot - v_original

    for r in prange(n_rep):
        for mt in range(n_mt):
            for g in range(n_groups):
                if group_n[g] > 0:
                    out[r, mt, g] /= group_n[g]
    return out


@njit(cache=True)
def fpr_dyadic_kernel(identity_multiplicities, edge_id1_codes, edge_id2_codes,
                      edge_group_codes, edge_denominators, edge_fp_counts,
                      n_groups):
    """Weighted FPR for each rep x model-threshold x a1 group.

    edge_fp_counts: [edge, model_threshold]. Each observed directed identity
    edge aggregates all comparison rows with that (fold,id1,id2,a1) key.
    Each edge receives vertex-bootstrap weight m_i*m_j.
    """
    n_rep = identity_multiplicities.shape[0]
    n_mt = edge_fp_counts.shape[1]
    n_edge = edge_id1_codes.shape[0]
    numer = np.zeros((n_rep, n_mt, n_groups), dtype=np.float64)
    denom = np.zeros((n_rep, n_groups), dtype=np.float64)

    for r in range(n_rep):
        for e in range(n_edge):
            w = (int(identity_multiplicities[r, edge_id1_codes[e]]) *
                 int(identity_multiplicities[r, edge_id2_codes[e]]))
            if w != 0:
                g = edge_group_codes[e]
                denom[r, g] += w * edge_denominators[e]
                for mt in range(n_mt):
                    numer[r, mt, g] += w * edge_fp_counts[e, mt]

    out = np.empty((n_rep, n_mt, n_groups), dtype=np.float64)
    for r in range(n_rep):
        for mt in range(n_mt):
            for g in range(n_groups):
                if denom[r, g] > 0:
                    out[r, mt, g] = numer[r, mt, g] / denom[r, g]
                else:
                    out[r, mt, g] = np.nan
    return out


# ----------------------------- HELPERS ----------------------------
def rank_info(values):
    values = np.asarray(values, dtype=float)
    ranks = rankdata(-values, method="average")
    worst = np.flatnonzero(np.isclose(values, np.max(values), rtol=0.0, atol=1e-12))
    gap = float(np.max(values) - np.min(values))
    tau = np.nan
    return ranks, worst, gap, tau


def joined(items):
    return " | ".join(sorted(str(x) for x in items))


def safe_tau(a, b):
    """Fast Kendall tau-b for the eight-group rankings used here."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    concordant = discordant = ties_a = ties_b = 0
    for i in range(len(a) - 1):
        for j in range(i + 1, len(a)):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da * db > 0:
                concordant += 1
            else:
                discordant += 1
    denom = (concordant + discordant + ties_a) * (concordant + discordant + ties_b)
    if denom == 0:
        return np.nan
    return float((concordant - discordant) / np.sqrt(denom))


def summarize_metric(values, group_names, reference_ranks, reference_worst):
    """Summarize [rep, group] bootstrap metric matrix."""
    n_rep = values.shape[0]
    rows = []
    worst_counts = {g: 0 for g in group_names}
    gaps = np.empty(n_rep, dtype=float)
    taus = np.empty(n_rep, dtype=float)
    worst_match = np.zeros(n_rep, dtype=np.int8)
    for r in range(n_rep):
        v = values[r]
        ranks, worst_idx, gap, _ = rank_info(v)
        gaps[r] = gap
        taus[r] = safe_tau(reference_ranks, ranks)
        worst_set = {group_names[i] for i in worst_idx}
        for g in worst_set:
            worst_counts[g] += 1
        worst_match[r] = int(worst_set == reference_worst)
    return gaps, taus, worst_match, worst_counts


# ----------------------------- LOAD / VALIDATE -----------------------------
start = time.time()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
required = ["fold", "p1", "p2", "label", "id1", "id2", "a1", "a2", *MODELS]
print("Loading BFW comparison table...")
df = pd.read_csv(BFW_DATA_PATH, usecols=required)
if df[required].isna().any().any():
    raise ValueError("Missing values in required columns.")
if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Expected label values 0 and 1.")
if not np.array_equal(df.loc[df.label == 1, "a1"].to_numpy(),
                      df.loc[df.label == 1, "a2"].to_numpy()):
    raise ValueError("Genuine comparisons have inconsistent a1/a2 groups.")

models = MODELS
groups = sorted(df[GROUP_COLUMN].astype(str).unique())
group_to_code = {g: i for i, g in enumerate(groups)}
n_groups = len(groups)
mt_models = np.repeat(np.arange(len(models)), len(THRESHOLDS))
mt_thresholds = np.tile(THRESHOLDS, len(models))
n_mt = len(mt_thresholds)
print(f"Rows: {len(df):,}; groups: {groups}; models: {models}")
print(f"Numba available: {NUMBA_AVAILABLE}; repetitions: {BOOTSTRAP_REPETITIONS:,}")

# Identity metadata: every identity must map to one fold and one a1 group.
identity_meta = pd.concat([
    df[["id1", "fold", "a1"]].rename(columns={"id1": "identity"}),
    df[["id2", "fold", "a2"]].rename(columns={"id2": "identity", "a2": "a1"}),
], ignore_index=True).drop_duplicates()
if identity_meta.groupby("identity").size().max() != 1:
    raise ValueError("An identity maps to multiple fold/group combinations.")
identity_meta = identity_meta.sort_values(["fold", "a1", "identity"]).reset_index(drop=True)
identity_ids = identity_meta["identity"].to_numpy()
identity_to_code = {v: i for i, v in enumerate(identity_ids)}
identity_group_codes = identity_meta["a1"].astype(str).map(group_to_code).to_numpy(np.int16)
identity_folds = identity_meta["fold"].to_numpy()

# ----------------------------- FULL DATA REFERENCES -----------------------------
reference_rows = []
reference_by_model_threshold = {}
for mi, model in enumerate(models):
    scores = df[model].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.int8)
    group_codes_rows = df["a1"].astype(str).map(group_to_code).to_numpy(np.int16)
    for ti, threshold in enumerate(THRESHOLDS):
        pred = scores >= threshold
        fnr = np.array([
            np.mean((labels[group_codes_rows == gi] == 1) & (~pred[group_codes_rows == gi])) /
            np.mean(labels[group_codes_rows == gi] == 1)
            for gi in range(n_groups)
        ], dtype=float)
        fpr = np.array([
            np.sum((labels[group_codes_rows == gi] == 0) & pred[group_codes_rows == gi]) /
            np.sum(labels[group_codes_rows == gi] == 0)
            for gi in range(n_groups)
        ], dtype=float)
        for metric, vals in [("FNR", fnr), ("FPR", fpr)]:
            ranks, worst_idx, gap, _ = rank_info(vals)
            key = (model, float(threshold), metric)
            reference_by_model_threshold[key] = {
                "values": vals, "ranks": ranks,
                "worst": {groups[i] for i in worst_idx}, "gap": gap,
            }
            row = {"model": model, "threshold": threshold, "metric": metric,
                   "gap": gap, "worst_groups": joined(groups[i] for i in worst_idx),
                   "ranking_tau_reference": 1.0}
            row.update({f"{metric.lower()}_{groups[i]}": vals[i] for i in range(n_groups)})
            reference_rows.append(row)

pd.DataFrame(reference_rows).to_csv(OUTPUT_DIR / "13_pairwise_uncertainty_reference.csv", index=False)

# ----------------------------- PART A: IMAGE-LEVEL FNR -----------------------------
print("\nPreparing complete genuine within-identity image-pair graph...")
genuine = df[(df.label == 1) & (df.id1 == df.id2) & (df.p1 != df.p2)].copy()
genuine["image_lo"] = np.where(genuine.p1 < genuine.p2, genuine.p1, genuine.p2)
genuine["image_hi"] = np.where(genuine.p1 < genuine.p2, genuine.p2, genuine.p1)
# Keep one record per unordered image pair. Scores should agree across reversed duplicates.
genuine_pair_keys = ["id1", "image_lo", "image_hi"]
# Reversed duplicate rows must carry the same precomputed similarity scores.
score_ranges = genuine.groupby(genuine_pair_keys)[models].agg(lambda x: float(x.max() - x.min()))
max_duplicate_score_range = float(score_ranges.to_numpy().max())
if max_duplicate_score_range > 1e-8:
    raise ValueError(f"Reversed genuine-pair scores disagree (max range={max_duplicate_score_range}).")
genuine = genuine.sort_values(["id1", "image_lo", "image_hi"]).drop_duplicates(
    genuine_pair_keys, keep="first")

# Build exactly 25 images and 300 unique pairs per identity.
identity_images = {}
for identity, sub in pd.concat([
    df[["id1", "p1"]].rename(columns={"id1": "identity", "p1": "image"}),
    df[["id2", "p2"]].rename(columns={"id2": "identity", "p2": "image"}),
], ignore_index=True).drop_duplicates().groupby("identity"):
    identity_images[identity] = sorted(sub["image"].tolist())

n_identity = len(identity_ids)
if n_identity != 800:
    warnings.warn(f"Expected 800 identities from prior audit; found {n_identity}.")
image_index = {identity: {p: j for j, p in enumerate(identity_images[identity])}
               for identity in identity_ids}
if any(len(identity_images[i]) != 25 for i in identity_ids):
    raise ValueError("Image-level bootstrap requires exactly 25 unique images per identity.")

pair_errors = np.zeros((n_identity, n_mt, 300), dtype=np.uint8)
pair_counts = np.zeros(n_identity, dtype=np.int16)
for row in genuine.itertuples(index=False):
    ii = identity_to_code[row.id1]
    a = image_index[row.id1][row.image_lo]
    b = image_index[row.id1][row.image_hi]
    if a > b:
        a, b = b, a
    # triangular index for a < b
    pair_idx = a * (49 - a) // 2 + (b - a - 1)
    pair_counts[ii] += 1
    for mt in range(n_mt):
        score = getattr(row, models[mt_models[mt]])
        pair_errors[ii, mt, pair_idx] = int(score < mt_thresholds[mt])

if not np.all(pair_counts == 300):
    bad = np.flatnonzero(pair_counts != 300)[:10]
    raise ValueError(f"Expected 300 unique genuine image pairs per identity; bad identity indices: {bad}")

print("Generating within-identity image resamples...")
rng = np.random.default_rng(RANDOM_SEED)
# Multiplicity cube: repetitions x identities x 25 images. uint8 is sufficient (max 25).
image_mult = np.zeros((BOOTSTRAP_REPETITIONS, n_identity, 25), dtype=np.uint8)
image_probabilities = np.full(25, 1.0 / 25.0)
for i in range(n_identity):
    image_mult[:, i, :] = rng.multinomial(25, image_probabilities, size=BOOTSTRAP_REPETITIONS).astype(np.uint8)

print("Computing recentered FNR bootstrap (U* - V* + U)...")
fnr_boot = fnr_recenter_kernel(image_mult, pair_errors, identity_group_codes, n_groups)
# Reference FNR from complete unique genuine pair graph, not all duplicated rows.
fnr_reference = np.zeros((n_mt, n_groups), dtype=float)
for mt in range(n_mt):
    for g in range(n_groups):
        idx = np.flatnonzero(identity_group_codes == g)
        fnr_reference[mt, g] = pair_errors[idx, mt, :].mean(axis=1).mean()

# Replace the initial row-table FNR reference with the exact unique-pair
# U-statistic used by Part A. The conventional baseline includes self/reversed
# records; this Part A estimand deliberately uses 300 unique unordered pairs
# per identity so that image-level U/V recentering is well-defined.
for mt in range(n_mt):
    model = models[mt_models[mt]]
    threshold = float(mt_thresholds[mt])
    vals = fnr_reference[mt]
    ranks, worst_idx, gap, _ = rank_info(vals)
    reference_by_model_threshold[(model, threshold, "FNR")] = {
        "values": vals.copy(), "ranks": ranks,
        "worst": {groups[i] for i in worst_idx}, "gap": gap,
    }

# Rewrite reference file now that the FNR reference uses the exact Part A U-statistic.
reference_rows = []
for (model, threshold, metric), ref in reference_by_model_threshold.items():
    vals = ref["values"]
    row = {"model": model, "threshold": threshold, "metric": metric,
           "gap": ref["gap"], "worst_groups": joined(ref["worst"]),
           "reference_definition": "unique genuine unordered image-pair U-statistic" if metric == "FNR" else "original observed directed impostor comparison rows"}
    row.update({f"{metric.lower()}_{groups[i]}": float(vals[i]) for i in range(n_groups)})
    reference_rows.append(row)
pd.DataFrame(reference_rows).to_csv(OUTPUT_DIR / "13_pairwise_uncertainty_reference.csv", index=False)

# ----------------------------- PART B: DYADIC FPR -----------------------------
print("\nPreparing directed impostor identity-pair graph...")
impostor = df[df.label == 0].copy()
if (impostor["id1"].to_numpy() == impostor["id2"].to_numpy()).any():
    raise ValueError("An impostor comparison has identical endpoint identities.")
impostor["edge_group_code"] = impostor["a1"].astype(str).map(group_to_code).astype(np.int16)
# Aggregate each directed (fold,id1,id2,a1) edge. Duplicate/reversed rows remain
# represented in the row count because the original a1 estimand is directional.
edge_keys = ["fold", "id1", "id2", "edge_group_code"]
edge_groupby = impostor.groupby(edge_keys, sort=False, observed=True)
edge_index = edge_groupby.ngroup().to_numpy(np.int32)
edge_table = edge_groupby.size().reset_index(name="edge_denominator")
edge_count = len(edge_table)
edge_id1_codes = edge_table["id1"].map(identity_to_code).to_numpy(np.int32)
edge_id2_codes = edge_table["id2"].map(identity_to_code).to_numpy(np.int32)
edge_group_codes = edge_table["edge_group_code"].to_numpy(np.int16)
edge_denominators = edge_table["edge_denominator"].to_numpy(np.float64)
if not np.array_equal(np.bincount(edge_index, minlength=edge_count).astype(np.float64), edge_denominators):
    raise ValueError("Impostor edge aggregation order/count validation failed.")
if not np.array_equal(identity_group_codes[edge_id1_codes], edge_group_codes):
    raise ValueError("An impostor edge's a1 group does not match its first identity's group.")

edge_fp_counts = np.zeros((edge_count, n_mt), dtype=np.float64)
labels_imp = impostor["label"].to_numpy(dtype=np.int8)
edge_codes = edge_index.astype(np.int32)
for mt in range(n_mt):
    model = models[mt_models[mt]]
    is_fp = (impostor[model].to_numpy(dtype=float) >= mt_thresholds[mt]).astype(np.float64)
    edge_fp_counts[:, mt] = np.bincount(edge_codes, weights=is_fp, minlength=edge_count)

# Bootstrap identities with replacement inside each original fold x a1 group.
# Every identity is sampled 20 times per stratum, preserving stratum size.
identity_strata = {}
for i, (fold, group) in enumerate(zip(identity_folds, identity_meta["a1"].astype(str))):
    identity_strata.setdefault((fold, group), []).append(i)
if any(len(v) != 20 for v in identity_strata.values()):
    raise ValueError("Expected exactly 20 identities per fold x demographic group.")

print("Generating fold/group-stratified identity multiplicities...")
identity_mult = np.zeros((BOOTSTRAP_REPETITIONS, n_identity), dtype=np.uint8)
for members in identity_strata.values():
    members = np.asarray(members, dtype=np.int32)
    stratum_counts = rng.multinomial(len(members), np.full(len(members), 1.0 / len(members)),
                                     size=BOOTSTRAP_REPETITIONS).astype(np.uint8)
    identity_mult[:, members] = stratum_counts

print("Computing dyadic FPR bootstrap (edge weight = m_i * m_j)...")
# Vectorized per-repetition edge weighting is substantially faster than a
# Python-level loop over every edge and threshold. Matrix-vector products are
# performed by NumPy/BLAS; the same identity multiplicities are shared by all models.
fpr_boot = np.empty((BOOTSTRAP_REPETITIONS, n_mt, n_groups), dtype=np.float64)
for r in range(BOOTSTRAP_REPETITIONS):
    edge_weights = (identity_mult[r, edge_id1_codes].astype(np.int32) *
                    identity_mult[r, edge_id2_codes].astype(np.int32)).astype(np.float64)
    for g in range(n_groups):
        mask = edge_group_codes == g
        w = edge_weights[mask]
        denominator = float(np.dot(edge_denominators[mask], w))
        if denominator <= 0:
            raise ValueError(f"Zero FPR denominator in repetition {r + 1}, group {groups[g]}.")
        fpr_boot[r, :, g] = (edge_fp_counts[mask].T @ w) / denominator
    if (r + 1) % 100 == 0:
        print(f"  Dyadic bootstrap repetitions: {r + 1:,}/{BOOTSTRAP_REPETITIONS:,}")

# ----------------------------- LONG OUTPUTS -----------------------------
print("\nWriting bootstrap repetitions and summaries...")
raw_rows = []
summary_rows = []
worst_rows = []

for mt in range(n_mt):
    model = models[mt_models[mt]]
    threshold = float(mt_thresholds[mt])
    for metric, boot_array, ref_array in [
        ("FNR", fnr_boot, fnr_reference),
        ("FPR", fpr_boot, None),
    ]:
        ref = reference_by_model_threshold[(model, threshold, metric)]
        values = boot_array[:, mt, :]
        if np.isnan(values).any():
            raise ValueError(f"Undefined bootstrap metric in {model} {threshold} {metric}.")
        gaps, taus, worst_match, worst_counts = summarize_metric(
            values, groups, ref["ranks"], ref["worst"]
        )
        # Raw group metrics: 2000 x 8 per model/threshold/metric.
        for r in range(BOOTSTRAP_REPETITIONS):
            row = {"repetition": r + 1, "model": model,
                   "threshold": threshold, "metric": metric,
                   "fairness_gap": gaps[r], "kendall_tau_b_vs_full_reference": taus[r],
                   "worst_group_set_matches_reference": int(worst_match[r])}
            row.update({groups[g]: float(values[r, g]) for g in range(n_groups)})
            raw_rows.append(row)
        summary_rows.append({
            "model": model, "threshold": threshold, "metric": metric,
            "bootstrap_method": "within_identity_image_recentered" if metric == "FNR" else "fold_group_stratified_identity_dyadic",
            "repetitions": BOOTSTRAP_REPETITIONS,
            "gap_mean": float(np.mean(gaps)), "gap_sd": float(np.std(gaps, ddof=1)),
            "gap_p2_5": float(np.quantile(gaps, 0.025)),
            "gap_p97_5": float(np.quantile(gaps, 0.975)),
            "tau_mean": float(np.nanmean(taus)),
            "tau_p2_5": float(np.nanquantile(taus, 0.025)),
            "tau_p97_5": float(np.nanquantile(taus, 0.975)),
            "worst_group_set_match_frequency": float(np.mean(worst_match)),
            "reference_gap": ref["gap"],
            "reference_worst_groups": joined(ref["worst"]),
        })
        for g in groups:
            worst_rows.append({
                "model": model, "threshold": threshold, "metric": metric,
                "group": g, "worst_group_frequency": worst_counts[g] / BOOTSTRAP_REPETITIONS,
                "worst_group_count": worst_counts[g],
            })

raw_path = OUTPUT_DIR / "13_pairwise_uncertainty_repetitions.csv"
summary_path = OUTPUT_DIR / "13_pairwise_uncertainty_summary.csv"
worst_path = OUTPUT_DIR / "13_pairwise_uncertainty_worst_group_frequency.csv"
pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
pd.DataFrame(worst_rows).to_csv(worst_path, index=False)

# Group-level percentile intervals for each metric.
group_summary_rows = []
raw_df = pd.DataFrame(raw_rows)
for (model, threshold, metric), sub in raw_df.groupby(["model", "threshold", "metric"], sort=False):
    for group in groups:
        vals = sub[group].to_numpy(dtype=float)
        group_summary_rows.append({
            "model": model, "threshold": threshold, "metric": metric, "group": group,
            "bootstrap_method": "within_identity_image_recentered" if metric == "FNR" else "fold_group_stratified_identity_dyadic",
            "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
            "p2_5": float(np.quantile(vals, 0.025)), "p97_5": float(np.quantile(vals, 0.975)),
        })
group_summary_df = pd.DataFrame(group_summary_rows)
group_summary_df.to_csv(OUTPUT_DIR / "13_pairwise_uncertainty_group_summary.csv", index=False)

# Optional direct comparison with Program 08's conventional row-bootstrap
# group summaries, if those outputs are present in the same OUTPUT_DIR.
row_bootstrap_path = OUTPUT_DIR / "bootstrap_group_metric_summary_all_models.csv"
method_comparison_path = OUTPUT_DIR / "13_pairwise_uncertainty_vs_row_bootstrap.csv"
if row_bootstrap_path.exists():
    row_bootstrap_df = pd.read_csv(row_bootstrap_path)
    comparison = group_summary_df.merge(
        row_bootstrap_df[["model", "metric", "group", "repetitions", "sd", "p2_5", "p97_5"]],
        on=["model", "metric", "group"], how="left", suffixes=("_pairwise", "_row")
    )
    comparison["row_to_pairwise_sd_ratio"] = comparison["sd_row"] / comparison["sd_pairwise"]
    comparison["row_interval_width"] = comparison["p97_5_row"] - comparison["p2_5_row"]
    comparison["pairwise_interval_width"] = comparison["p97_5_pairwise"] - comparison["p2_5_pairwise"]
    comparison["row_to_pairwise_interval_width_ratio"] = (
        comparison["row_interval_width"] / comparison["pairwise_interval_width"]
    )
    comparison.to_csv(method_comparison_path, index=False)
else:
    print("Program 08 group summary not found; skipping optional row-bootstrap comparison.")

# ----------------------------- MANIFEST -----------------------------
manifest = {
    "experiment": "Program 13 - Pairwise-Dependence-Aware Uncertainty Quantification",
    "dataset_rows": int(len(df)), "identity_count": int(n_identity),
    "unique_image_count": int(sum(len(v) for v in identity_images.values())),
    "genuine_unique_unordered_pairs": int(len(genuine)),
    "max_score_range_across_reversed_genuine_duplicates": max_duplicate_score_range,
    "directed_impostor_identity_edges": int(edge_count),
    "models": models, "thresholds": THRESHOLDS.tolist(),
    "group_column": GROUP_COLUMN, "groups": groups,
    "bootstrap_repetitions": BOOTSTRAP_REPETITIONS, "seed": RANDOM_SEED,
    "part_a": {
        "metric": "FNR", "method": "within-identity image resampling with recentering",
        "formula": "U_original + U_bootstrap - V_original (corrected)",
        "sampling": "25 images sampled with replacement within each identity; identity and fold composition fixed",
        "pair_graph": "300 unique genuine unordered image pairs per identity",
        "diagonal": "self-comparison FN indicator set to zero because self-score is 1 and thresholds are <= 0.70",
        "aggregation": "equal identity average within each a1 group; equivalent to pooled pair rate because every identity has 300 genuine pairs",
        "qualification": "recentered image-level FNR procedure adapted to the complete genuine-pair graph available in BFW; V_original is the original-sample V-statistic, not the bootstrap-sample V-statistic; no claim of full reproduction of every paper-specific theorem or procedure"
    },
    "part_b": {
        "metric": "FPR", "method": "fold x a1-group stratified identity vertex/dyadic bootstrap",
        "sampling": "20 identities sampled with replacement within each original fold x demographic group",
        "edge_weight": "multiplicity(id1) * multiplicity(id2)",
        "comparison_graph": "observed directed impostor identity-pair edges only; no missing image comparisons are imputed",
        "qualification": "identity-level dyadic resampling approximation for the sampled BFW impostor graph; not the exact recentered bootstrap from the ICLR paper"
    },
    "common": {
        "models_paired": True,
        "thresholds_evaluated": THRESHOLDS.tolist(),
        "ranking": "Kendall tau-b against full-data group ranking",
        "gap": "max(group metric) - min(group metric)",
        "intervals": "percentile summaries of the specified resampling distributions; interpretation conditional on BFW design and each resampling scheme",
        "limitations": [
            "FNR and FPR use different resampling designs and should not be interpreted as identical inferential procedures.",
            "FPR procedure conditions on the observed sampled impostor graph and cannot recover unobserved image-pair scores.",
            "Identity resampling preserves dependence induced by shared identity endpoints but does not fully resample images within identities for FPR.",
            "Original fold and demographic group counts are held fixed.",
            "Results do not establish external population generalization or causal demographic effects."
        ]
    },
    "outputs": [raw_path.name, summary_path.name, worst_path.name,
                "13_pairwise_uncertainty_group_summary.csv",
                "13_pairwise_uncertainty_reference.csv",
                "13_pairwise_uncertainty_vs_row_bootstrap.csv (if Program 08 group summary exists)"]
}
with open(OUTPUT_DIR / "13_pairwise_uncertainty_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\n" + "=" * 72)
print("PROGRAM 13 COMPLETE")
print("=" * 72)
print(f"Genuine unique pairs: {len(genuine):,}; identities with 300 pairs: {(pair_counts == 300).sum():,}")
print(f"Directed impostor identity edges: {edge_count:,}")
print(f"Raw repetitions: {raw_path}")
print(f"Summary: {summary_path}")
print(f"Worst-group frequency: {worst_path}")
print(f"Elapsed minutes: {(time.time() - start) / 60:.2f}")
