"""AuditFlux Program 13B: FPR vertex-bootstrap centring correction.

This is a supplementary FPR-only program. It does NOT rerun or modify the
accepted image-level recentered FNR analysis in Program 13.

Primary method:
  Fold x a1-group stratified multinomial identity bootstrap, with edge weights
  m_i*m_j. For distinct endpoints in the same stratum of size n=20,
  E[m_i*m_j]=(n-1)/n=0.95. Multiply such edge weights by n/(n-1)=20/19
  so their expected weight is one, matching edges whose endpoints are in
  different strata (whose multiplicities are independent).

Sensitivity method:
  Independent Poisson(1) identity weights, with the same product edge weights.
  This avoids the within-stratum multinomial negative dependence, but does not
  preserve exactly 20 draws per fold x group stratum.

Both methods condition on the observed, sampled directed impostor graph. They
retain shared-identity dependence through common vertex weights, but do not
impute unobserved impostor image comparisons or provide unconditional
population guarantees.

Requires: pandas, numpy, scipy. Uses project config.py for data/output paths.
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import BFW_DATA_PATH, OUTPUT_DIR

# ----------------------------- CONFIG -----------------------------
MODELS = ["vgg16", "resnet50", "senet50"]
THRESHOLD = 0.50
REPETITIONS = 1000
SEED = 20261104
GROUP_COLUMN = "a1"
STRATUM_SIZE_EXPECTED = 20
ALPHA = 0.05


def safe_tau(a, b):
    """Kendall tau-b for small group rankings, including ties."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = d = ta = tb = 0
    for i in range(len(a) - 1):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da * db > 0:
                c += 1
            else:
                d += 1
    den = (c + d + ta) * (c + d + tb)
    return np.nan if den == 0 else float((c - d) / np.sqrt(den))


def worst_set(values, groups):
    mx = np.max(values)
    return {groups[i] for i in np.flatnonzero(np.isclose(values, mx, rtol=0, atol=1e-12))}


def rank_vector(values):
    return rankdata(-np.asarray(values, dtype=float), method="average")


def joined(items):
    return " | ".join(sorted(map(str, items)))


start = time.time()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
required = ["fold", "label", "id1", "id2", "a1", "a2", *MODELS]
print("Loading BFW table...")
df = pd.read_csv(BFW_DATA_PATH, usecols=required)
if df[required].isna().any().any():
    raise ValueError("Missing values in required columns.")
if not set(df.label.unique()).issubset({0, 1}):
    raise ValueError("Expected binary labels 0 and 1.")

impostor = df.loc[df.label == 0].copy()
if (impostor.id1.to_numpy() == impostor.id2.to_numpy()).any():
    raise ValueError("Impostor comparison has identical endpoint identities.")
impostor["group_code"] = impostor[GROUP_COLUMN].astype(str)
groups = sorted(df[GROUP_COLUMN].astype(str).unique())
group_to_code = {g: i for i, g in enumerate(groups)}
impostor["group_code"] = impostor.group_code.map(group_to_code).astype(np.int16)

# Identity metadata: each identity must have one fold and one demographic group.
meta = pd.concat([
    df[["id1", "fold", "a1"]].rename(columns={"id1": "identity", "a1": "group"}),
    df[["id2", "fold", "a2"]].rename(columns={"id2": "identity", "a2": "group"}),
], ignore_index=True).drop_duplicates()
if meta.groupby("identity").size().max() != 1:
    raise ValueError("Identity maps to multiple fold/group combinations.")
meta = meta.sort_values(["fold", "group", "identity"]).reset_index(drop=True)
ids = meta.identity.to_numpy()
id_to_code = {v: i for i, v in enumerate(ids)}
id_fold = meta.fold.to_numpy()
id_group = meta.group.astype(str).to_numpy()

strata = {}
for i, (fold, group) in enumerate(zip(id_fold, id_group)):
    strata.setdefault((fold, group), []).append(i)
stratum_sizes = {key: len(members) for key, members in strata.items()}
if len(set(stratum_sizes.values())) != 1 or next(iter(set(stratum_sizes.values()))) != STRATUM_SIZE_EXPECTED:
    raise ValueError(f"Expected {STRATUM_SIZE_EXPECTED} identities in every fold x group stratum; got {sorted(set(stratum_sizes.values()))}.")

# Aggregate observed directed impostor comparisons into identity-pair edges.
edge_keys = ["fold", "id1", "id2", "group_code"]
gb = impostor.groupby(edge_keys, sort=False, observed=True)
edge = gb.size().reset_index(name="denom")
edge["i"] = edge.id1.map(id_to_code)
edge["j"] = edge.id2.map(id_to_code)
if edge[["i", "j"]].isna().any().any():
    raise ValueError("Could not map all edge endpoints to identity metadata.")
i_code = edge.i.to_numpy(np.int32)
j_code = edge.j.to_numpy(np.int32)
g_code = edge.group_code.to_numpy(np.int16)
denom = edge.denom.to_numpy(np.float64)

# Model-specific false-positive row counts for each aggregated edge.
fp_counts = np.zeros((len(edge), len(MODELS)), dtype=np.float64)
edge_ids = gb.ngroup().to_numpy(np.int32)
for mi, model in enumerate(MODELS):
    fp = (impostor[model].to_numpy(dtype=float) >= THRESHOLD).astype(float)
    fp_counts[:, mi] = np.bincount(edge_ids, weights=fp, minlength=len(edge))
if not np.array_equal(np.bincount(edge_ids, minlength=len(edge)), denom):
    raise ValueError("Edge aggregation consistency check failed.")
if not np.array_equal(meta.group.astype(str).map(group_to_code).to_numpy()[i_code], g_code):
    raise ValueError("Edge a1 group does not match first endpoint group.")

n_groups = len(groups)
n_models = len(MODELS)
# Full observed FPR reference, grouped by first endpoint a1.
ref = np.zeros((n_models, n_groups), dtype=float)
for mi, model in enumerate(MODELS):
    pred = impostor[model].to_numpy(dtype=float) >= THRESHOLD
    for gi in range(n_groups):
        mask = impostor.group_code.to_numpy() == gi
        ref[mi, gi] = float(pred[mask].mean())
ref_rank = np.vstack([rank_vector(ref[mi]) for mi in range(n_models)])
ref_worst = [worst_set(ref[mi], groups) for mi in range(n_models)]

# Identify edges whose endpoints are in the same fold x group resampling stratum.
same_stratum = ((id_fold[i_code] == id_fold[j_code]) & (id_group[i_code] == id_group[j_code]))
correction = np.ones(len(edge), dtype=np.float64)
correction[same_stratum] = STRATUM_SIZE_EXPECTED / (STRATUM_SIZE_EXPECTED - 1.0)
print(f"Rows={len(df):,}; identities={len(ids):,}; impostor rows={len(impostor):,}; directed edges={len(edge):,}")
print(f"Edges with endpoints in same resampling stratum: {same_stratum.sum():,}/{len(edge):,}")

# Output arrays: [method, repetition, model, group].
methods = ["corrected_stratified_multinomial", "independent_poisson_vertex"]
boot = np.full((2, REPETITIONS, n_models, n_groups), np.nan, dtype=np.float64)
rng_multi = np.random.default_rng(SEED)
rng_pois = np.random.default_rng(SEED + 1)

print("Running corrected stratified multinomial vertex bootstrap...")
for r in range(REPETITIONS):
    mult = np.zeros(len(ids), dtype=np.int16)
    for members in strata.values():
        members = np.asarray(members, dtype=np.int32)
        mult[members] = rng_multi.multinomial(len(members), np.full(len(members), 1 / len(members)))
    w = mult[i_code].astype(np.float64) * mult[j_code].astype(np.float64) * correction
    for gi in range(n_groups):
        mask = g_code == gi
        den_r = float(np.dot(denom[mask], w[mask]))
        if den_r <= 0:
            raise RuntimeError(f"Zero corrected multinomial denominator: repetition {r+1}, group {groups[gi]}")
        boot[0, r, :, gi] = (fp_counts[mask].T @ w[mask]) / den_r
    if (r + 1) % 100 == 0:
        print(f"  corrected multinomial: {r+1:,}/{REPETITIONS:,}")

print("Running independent Poisson(1) vertex bootstrap...")
for r in range(REPETITIONS):
    # One common vertex-weight realization must be valid for every group so
    # all group estimates in a repetition remain jointly comparable.
    for attempt in range(100):
        mult = rng_pois.poisson(1.0, size=len(ids)).astype(np.int16)
        w = mult[i_code].astype(np.float64) * mult[j_code].astype(np.float64)
        denominators = np.array([
            float(np.dot(denom[g_code == gi], w[g_code == gi]))
            for gi in range(n_groups)
        ])
        if np.all(denominators > 0):
            break
    else:
        raise RuntimeError("Poisson bootstrap repeatedly yielded a zero group denominator.")
    for gi in range(n_groups):
        mask = g_code == gi
        boot[1, r, :, gi] = (fp_counts[mask].T @ w[mask]) / denominators[gi]
    if (r + 1) % 100 == 0:
        print(f"  Poisson vertex: {r+1:,}/{REPETITIONS:,}")

# Long repetition output.
rep_rows = []
summary_rows = []
group_rows = []
worst_rows = []
for meth_i, method in enumerate(methods):
    for mi, model in enumerate(MODELS):
        vals = boot[meth_i, :, mi, :]
        if not np.isfinite(vals).all():
            raise ValueError(f"Non-finite bootstrap values for {method}/{model}.")
        gaps = vals.max(axis=1) - vals.min(axis=1)
        taus = np.array([safe_tau(ref_rank[mi], rank_vector(v)) for v in vals])
        worst_matches = np.zeros(REPETITIONS, dtype=np.int8)
        worst_counts = {g: 0 for g in groups}
        for r, v in enumerate(vals):
            ws = worst_set(v, groups)
            worst_matches[r] = int(ws == ref_worst[mi])
            for g in ws:
                worst_counts[g] += 1
            row = {"method": method, "repetition": r + 1, "model": model,
                   "threshold": THRESHOLD, "fairness_gap": float(gaps[r]),
                   "kendall_tau_b_vs_full_reference": float(taus[r]),
                   "worst_group_set_matches_reference": int(worst_matches[r])}
            row.update({groups[g]: float(v[g]) for g in range(n_groups)})
            rep_rows.append(row)
        summary_rows.append({
            "method": method, "model": model, "threshold": THRESHOLD,
            "repetitions": REPETITIONS,
            "gap_mean": float(gaps.mean()), "gap_sd": float(gaps.std(ddof=1)),
            "gap_p2_5": float(np.quantile(gaps, .025)), "gap_p97_5": float(np.quantile(gaps, .975)),
            "tau_mean": float(np.nanmean(taus)), "tau_p2_5": float(np.nanquantile(taus, .025)),
            "tau_p97_5": float(np.nanquantile(taus, .975)),
            "worst_group_set_match_frequency": float(worst_matches.mean()),
            "reference_gap": float(ref[mi].max() - ref[mi].min()),
            "reference_worst_groups": joined(ref_worst[mi]),
        })
        for gi, group in enumerate(groups):
            x = vals[:, gi]
            group_rows.append({"method": method, "model": model, "threshold": THRESHOLD,
                               "group": group, "reference_fpr": float(ref[mi, gi]),
                               "mean": float(x.mean()), "bias_vs_reference": float(x.mean() - ref[mi, gi]),
                               "sd": float(x.std(ddof=1)), "p2_5": float(np.quantile(x, .025)),
                               "p97_5": float(np.quantile(x, .975)),
                               "reference_inside_percentile_interval": bool(np.quantile(x,.025) <= ref[mi,gi] <= np.quantile(x,.975))})
            worst_rows.append({"method": method, "model": model, "threshold": THRESHOLD,
                               "group": group, "worst_group_count": worst_counts[group],
                               "worst_group_frequency": worst_counts[group] / REPETITIONS})

# Compare against existing Program 13 original multinomial summary if present.
comparison_rows = []
for mi, model in enumerate(MODELS):
    for method_i, method in enumerate(methods):
        vals = boot[method_i, :, mi, :]
        for gi, group in enumerate(groups):
            comparison_rows.append({"model": model, "group": group, "method": method,
                                    "mean_fpr": float(vals[:, gi].mean()),
                                    "sd_fpr": float(vals[:, gi].std(ddof=1)),
                                    "bias_vs_full_data": float(vals[:, gi].mean() - ref[mi, gi])})
comparison_df = pd.DataFrame(comparison_rows)
old_group_path = OUTPUT_DIR / "13_pairwise_uncertainty_group_summary.csv"
if old_group_path.exists():
    old = pd.read_csv(old_group_path)
    old = old[(old.metric == "FPR") & (old.threshold == THRESHOLD)][["model", "group", "mean", "sd"]]
    old = old.rename(columns={"mean": "original_multinomial_mean", "sd": "original_multinomial_sd"})
    comparison_df = comparison_df.merge(old, on=["model", "group"], how="left")
    comparison_df["original_multinomial_bias_vs_full_data"] = comparison_df["original_multinomial_mean"] - comparison_df.merge(
        pd.DataFrame({"model": np.repeat(MODELS, n_groups), "group": groups * n_models,
                      "reference_fpr": ref.reshape(-1)}), on=["model", "group"], how="left")["reference_fpr"].to_numpy()

# Persist references and outputs.
ref_rows = []
for mi, model in enumerate(MODELS):
    row = {"model": model, "threshold": THRESHOLD,
           "reference_gap": float(ref[mi].max() - ref[mi].min()),
           "reference_worst_groups": joined(ref_worst[mi])}
    row.update({f"fpr_{groups[g]}": float(ref[mi, g]) for g in range(n_groups)})
    ref_rows.append(row)

paths = {
    "reference": OUTPUT_DIR / "13B_fpr_reference.csv",
    "repetitions": OUTPUT_DIR / "13B_fpr_repetitions.csv",
    "summary": OUTPUT_DIR / "13B_fpr_summary.csv",
    "group_summary": OUTPUT_DIR / "13B_fpr_group_summary.csv",
    "worst_group_frequency": OUTPUT_DIR / "13B_fpr_worst_group_frequency.csv",
    "method_comparison": OUTPUT_DIR / "13B_fpr_method_comparison.csv",
    "manifest": OUTPUT_DIR / "13B_fpr_manifest.json",
}
pd.DataFrame(ref_rows).to_csv(paths["reference"], index=False)
pd.DataFrame(rep_rows).to_csv(paths["repetitions"], index=False)
pd.DataFrame(summary_rows).to_csv(paths["summary"], index=False)
pd.DataFrame(group_rows).to_csv(paths["group_summary"], index=False)
pd.DataFrame(worst_rows).to_csv(paths["worst_group_frequency"], index=False)
comparison_df.to_csv(paths["method_comparison"], index=False)

manifest = {
    "experiment": "Program 13B - FPR Vertex Bootstrap Centering Sensitivity",
    "dataset_rows": int(len(df)), "impostor_comparison_rows": int(len(impostor)),
    "identity_count": int(len(ids)), "directed_observed_impostor_edges": int(len(edge)),
    "models": MODELS, "threshold": THRESHOLD, "group_column": GROUP_COLUMN,
    "repetitions_per_method": REPETITIONS, "seed": SEED,
    "strata": "original fold x demographic group; 20 identities per stratum",
    "primary_method": {
        "name": "corrected_stratified_multinomial_vertex_bootstrap",
        "sampling": "20 identities sampled with replacement within each original fold x a1-group stratum",
        "base_edge_weight": "m_i * m_j",
        "centering_correction": "for distinct endpoints in the same fold x group stratum, multiply edge weight by 20/19; cross-stratum edges unchanged",
        "justification": "under multinomial sampling of n=20 vertices, E[m_i*m_j]=(n-1)/n=19/20 for distinct vertices in the same stratum, while independent-stratum endpoint multiplicities have expected product 1; the 20/19 correction equalizes expected edge weights at 1",
        "estimand": "FPR on the observed sampled directed impostor comparison graph, grouped by first endpoint a1"
    },
    "sensitivity_method": {
        "name": "independent_poisson_vertex_bootstrap",
        "sampling": "independent Poisson(1) multiplicity per identity",
        "edge_weight": "m_i * m_j",
        "qualification": "does not preserve exactly 20 sampled identities per fold x group stratum; used as a sensitivity analysis"
    },
    "paired_models": True,
    "reference": "full observed FPR by first endpoint a1 at threshold 0.50",
    "intervals": "percentile summaries of the specified resampling distributions; not asserted as universally calibrated 95% population confidence intervals",
    "limitations": [
        "The observed impostor graph is sampled and incomplete; unobserved image comparisons are not imputed.",
        "Vertex weights preserve dependence from shared identity endpoints in the observed graph, but the bootstrap remains conditional on the BFW graph and fixed fold/group strata.",
        "The corrected factor equalizes expected edge weights; it does not establish exact finite-sample unbiasedness or nominal interval coverage for the ratio estimator.",
        "The Poisson method is a sensitivity analysis with random stratum representation, not a replacement for the fixed-stratum primary design.",
        "FPR resampling is distinct from the accepted image-level recentered FNR procedure in Program 13."
    ],
    "outputs": {k: v.name for k, v in paths.items()},
}
with open(paths["manifest"], "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nPROGRAM 13B COMPLETE")
print(pd.DataFrame(summary_rows).to_string(index=False))
print(f"\nOutputs written to: {OUTPUT_DIR}")
print(f"Elapsed minutes: {(time.time()-start)/60:.2f}")
