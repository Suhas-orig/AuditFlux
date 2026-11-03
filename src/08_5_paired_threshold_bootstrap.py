"""Paired threshold bootstrap for AuditFlux.

Resamples BFW comparison rows with replacement. Each bootstrap sample is
reused across every model and threshold. This estimates conditional
row-resampling variability in the empirical comparison table; it is NOT an
identity-clustered population bootstrap.
"""

import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import BFW_DATA_PATH, OUTPUT_DIR


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = ["vgg16", "resnet50", "senet50"]
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
METRICS = ["FNR", "FPR"]
BOOTSTRAP_REPETITIONS = 5_000
RANDOM_SEED = 42
CHECKPOINT_EVERY = 100
TIE_TOLERANCE = 1e-12
EXPERIMENT_VERSION = "08.5-v2"

OUTPUT_STEM = "paired_threshold_bootstrap_v2"
WORK_DIR = OUTPUT_DIR / f"{OUTPUT_STEM}_checkpoints"
RAW_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_raw.csv"
TRANSITION_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_transitions.csv"
SUMMARY_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_summary.csv"
REFERENCE_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}_reference.csv"
MANIFEST_PATH = WORK_DIR / "manifest.json"

RAW_COLUMNS = ["bootstrap_id", "model", "threshold", "group", "TPR", "FNR", "FPR"]
TRANSITION_COLUMNS = [
    "bootstrap_id", "model", "metric", "threshold_from", "threshold_to",
    "worst_group_from", "worst_group_to", "worst_group_reversal",
    "worst_group_comparison_defined", "worst_group_set_changed",
    "worst_group_complete_reversal", "worst_group_jaccard_similarity",
    "kendall_tau", "ranking_instability", "ranking_instability_normalized",
    "ranking_comparison_status", "gap_from", "gap_to", "signed_gap_change",
    "magnitude_change",
]


def calculate_metrics_fast(labels, scores, group_codes, group_names, threshold):
    predictions = (scores >= threshold).astype(np.int8)
    # 0=TN, 1=FP, 2=FN, 3=TP.
    outcome_codes = labels * 2 + predictions
    combined_codes = group_codes * 4 + outcome_codes
    counts = np.bincount(
        combined_codes, minlength=len(group_names) * 4
    ).reshape(len(group_names), 4)

    tn, fp, fn, tp = counts.T
    positives = tp + fn
    negatives = fp + tn
    tpr = np.divide(tp, positives, out=np.full(len(group_names), np.nan), where=positives > 0)
    fnr = np.divide(fn, positives, out=np.full(len(group_names), np.nan), where=positives > 0)
    fpr = np.divide(fp, negatives, out=np.full(len(group_names), np.nan), where=negatives > 0)

    if np.isnan(fnr).any() or np.isnan(fpr).any():
        raise ValueError("A bootstrap sample contains a group with no positive or negative examples.")

    return pd.DataFrame({"group": group_names, "TPR": tpr, "FNR": fnr, "FPR": fpr})


def tolerance_aware_average_ranks(values, tolerance=TIE_TOLERANCE):
    """Descending average ranks; values within tolerance share a rank."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values encountered while ranking groups.")

    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and abs(sorted_values[end] - sorted_values[start]) <= tolerance:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def ranking_and_worst_groups(metrics_df, metric):
    values = metrics_df[metric].to_numpy(dtype=float)
    ranks = tolerance_aware_average_ranks(values)
    maximum = values.max()
    worst = set(metrics_df.loc[np.abs(values - maximum) <= TIE_TOLERANCE, "group"].tolist())
    return ranks, worst


def gap(values):
    values = np.asarray(values, dtype=float)
    return float(values.max() - values.min())


def wilson_interval(successes, trials, z=1.959963984540054):
    """Monte Carlo interval for a bootstrap event probability, not a population CI."""
    if trials == 0:
        return np.nan, np.nan
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * np.sqrt((p * (1 - p) / trials) + (z * z / (4 * trials * trials))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def atomic_json_write(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def config_fingerprint(n, group_names):
    config = {
        "version": EXPERIMENT_VERSION,
        "dataset_path": str(BFW_DATA_PATH),
        "dataset_rows": int(n),
        "groups": [str(x) for x in group_names],
        "models": MODELS,
        "thresholds": THRESHOLDS,
        "metrics": METRICS,
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": RANDOM_SEED,
        "tie_tolerance": TIE_TOLERANCE,
        "sampling": "paired row bootstrap with replacement; sample reused across models and thresholds",
    }
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), config


def write_chunk(rows, columns, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(temp, index=False)
    os.replace(temp, path)


def main():
    print("=" * 76)
    print("PAIRED THRESHOLD BOOTSTRAP — REVISED 5,000-REPETITION RUN")
    print("=" * 76)

    required = ["label", "a1"] + MODELS
    df = pd.read_csv(BFW_DATA_PATH, usecols=required)
    labels = df["label"].to_numpy(dtype=np.int8)
    group_names, group_codes = np.unique(df["a1"].to_numpy(), return_inverse=True)
    group_codes = group_codes.astype(np.int8, copy=False)
    scores_by_model = {m: df[m].to_numpy(dtype=np.float64) for m in MODELS}
    n = len(df)
    del df
    gc.collect()

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be encoded as 0/1.")
    if any(not np.isfinite(x).all() for x in scores_by_model.values()):
        raise ValueError("Non-finite model scores detected.")

    print(f"Dataset rows: {n:,} | Groups: {list(group_names)}")
    print(f"Models: {MODELS} | Thresholds: {THRESHOLDS}")
    print(f"Bootstrap repetitions: {BOOTSTRAP_REPETITIONS:,} | Seed: {RANDOM_SEED}")
    print("Design: one row-bootstrap sample reused across every model and threshold.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint, config = config_fingerprint(n, group_names)

    # Reference audits are deterministic and are regenerated for this run.
    reference_rows = []
    for model in MODELS:
        for threshold in THRESHOLDS:
            met = calculate_metrics_fast(labels, scores_by_model[model], group_codes, group_names, threshold)
            for row in met.itertuples(index=False):
                reference_rows.append({"model": model, "threshold": threshold,
                                       "group": row.group, "TPR": row.TPR,
                                       "FNR": row.FNR, "FPR": row.FPR})
    pd.DataFrame(reference_rows).to_csv(REFERENCE_PATH, index=False)

    # Resume only from a manifest whose full configuration fingerprint matches.
    rng = np.random.default_rng(RANDOM_SEED)
    completed = 0
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("fingerprint") == fingerprint:
            completed = int(manifest.get("completed_repetitions", 0))
            if completed > BOOTSTRAP_REPETITIONS:
                raise RuntimeError("Checkpoint exceeds configured repetition count.")
            if completed:
                rng.bit_generator.state = manifest["rng_state"]
                print(f"Resuming from verified checkpoint: {completed:,}/{BOOTSTRAP_REPETITIONS:,}.")
        else:
            print("Existing checkpoint configuration differs; starting a clean v2 run.")
            for p in WORK_DIR.glob("*.csv"):
                p.unlink()
            MANIFEST_PATH.unlink()
    else:
        print("No compatible checkpoint found; starting a fresh run.")

    def chunk_path(kind, start, end):
        return WORK_DIR / f"{kind}_{start:06d}_{end:06d}.csv"

    while completed < BOOTSTRAP_REPETITIONS:
        start = completed + 1
        end = min(completed + CHECKPOINT_EVERY, BOOTSTRAP_REPETITIONS)
        raw_chunk, transition_chunk = [], []

        for bootstrap_id in range(start, end + 1):
            sample_indices = rng.integers(0, n, size=n)
            boot_labels = labels[sample_indices]
            boot_groups = group_codes[sample_indices]

            for model in MODELS:
                boot_scores = scores_by_model[model][sample_indices]
                threshold_metrics = {}
                for threshold in THRESHOLDS:
                    met = calculate_metrics_fast(boot_labels, boot_scores, boot_groups, group_names, threshold)
                    threshold_metrics[threshold] = met
                    for row in met.itertuples(index=False):
                        raw_chunk.append({
                            "bootstrap_id": bootstrap_id, "model": model,
                            "threshold": threshold, "group": row.group,
                            "TPR": row.TPR, "FNR": row.FNR, "FPR": row.FPR,
                        })

                for metric in METRICS:
                    ranks_by_threshold, gaps_by_threshold, worst_by_threshold = {}, {}, {}
                    for threshold in THRESHOLDS:
                        met = threshold_metrics[threshold]
                        ranks, worst = ranking_and_worst_groups(met, metric)
                        ranks_by_threshold[threshold] = ranks
                        gaps_by_threshold[threshold] = gap(met[metric].to_numpy())
                        worst_by_threshold[threshold] = worst

                    for t1, t2 in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
                        tau, _ = kendalltau(ranks_by_threshold[t1], ranks_by_threshold[t2], variant="b")
                        tau = float(tau) if np.isfinite(tau) else np.nan
                        ranking_status = "defined" if np.isfinite(tau) else "undefined_constant_ranking"
                        instability = 1.0 - tau if np.isfinite(tau) else np.nan
                        normalized_instability = instability / 2.0 if np.isfinite(instability) else np.nan

                        w1, w2 = worst_by_threshold[t1], worst_by_threshold[t2]
                        singleton_defined = len(w1) == 1 and len(w2) == 1
                        singleton_reversal = int(w1 != w2) if singleton_defined else np.nan
                        union = w1 | w2
                        jaccard = len(w1 & w2) / len(union) if union else np.nan
                        g1, g2 = gaps_by_threshold[t1], gaps_by_threshold[t2]
                        signed_change = g2 - g1

                        transition_chunk.append({
                            "bootstrap_id": bootstrap_id, "model": model, "metric": metric,
                            "threshold_from": t1, "threshold_to": t2,
                            "worst_group_from": "|".join(sorted(w1)),
                            "worst_group_to": "|".join(sorted(w2)),
                            "worst_group_reversal": singleton_reversal,
                            "worst_group_comparison_defined": bool(singleton_defined),
                            "worst_group_set_changed": bool(w1 != w2),
                            "worst_group_complete_reversal": bool(not (w1 & w2)),
                            "worst_group_jaccard_similarity": jaccard,
                            "kendall_tau": tau,
                            "ranking_instability": instability,
                            "ranking_instability_normalized": normalized_instability,
                            "ranking_comparison_status": ranking_status,
                            "gap_from": g1, "gap_to": g2,
                            "signed_gap_change": signed_change,
                            "magnitude_change": abs(signed_change),
                        })

            del sample_indices, boot_labels, boot_groups

        # Write both data chunks first. The manifest is committed last; a crash
        # before that leaves uncommitted chunks which are safely overwritten.
        raw_file = chunk_path("raw", start, end)
        transition_file = chunk_path("transitions", start, end)
        write_chunk(raw_chunk, RAW_COLUMNS, raw_file)
        write_chunk(transition_chunk, TRANSITION_COLUMNS, transition_file)
        completed = end
        atomic_json_write(MANIFEST_PATH, {
            "fingerprint": fingerprint,
            "configuration": config,
            "completed_repetitions": completed,
            "rng_state": rng.bit_generator.state,
        })
        print(f"Checkpoint committed: {completed:,}/{BOOTSTRAP_REPETITIONS:,}")
        del raw_chunk, transition_chunk
        gc.collect()

    # Assemble final files from committed chunks in deterministic order.
    raw_parts = sorted(WORK_DIR.glob("raw_*.csv"))
    transition_parts = sorted(WORK_DIR.glob("transitions_*.csv"))
    if len(raw_parts) != BOOTSTRAP_REPETITIONS // CHECKPOINT_EVERY or len(transition_parts) != len(raw_parts):
        # Include a possible shorter final chunk if repetitions are not divisible by checkpoint size.
        expected_chunks = (BOOTSTRAP_REPETITIONS + CHECKPOINT_EVERY - 1) // CHECKPOINT_EVERY
        if len(raw_parts) != expected_chunks or len(transition_parts) != expected_chunks:
            raise RuntimeError("Committed checkpoint chunks are missing; refusing to create incomplete outputs.")

    for parts, final_path, columns in [
        (raw_parts, RAW_PATH, RAW_COLUMNS),
        (transition_parts, TRANSITION_PATH, TRANSITION_COLUMNS),
    ]:
        temp = final_path.with_suffix(final_path.suffix + ".tmp")
        first = True
        with open(temp, "w", encoding="utf-8", newline="") as out:
            for part in parts:
                frame = pd.read_csv(part)
                frame.to_csv(out, index=False, header=first, columns=columns)
                first = False
        os.replace(temp, final_path)

    # Summaries: one row per model × metric × adjacent threshold transition.
    transitions = pd.read_csv(TRANSITION_PATH)
    summary_rows = []
    for model in MODELS:
        for metric in METRICS:
            subset = transitions[(transitions.model == model) & (transitions.metric == metric)]
            for t1, t2 in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
                tr = subset[(subset.threshold_from == t1) & (subset.threshold_to == t2)]
                valid_tau = tr.kendall_tau.dropna()
                valid_instability = tr.ranking_instability.dropna()
                defined_reversals = tr.worst_group_reversal.dropna()
                reversal_count = int(defined_reversals.sum())
                reversal_trials = int(len(defined_reversals))
                mc_low, mc_high = wilson_interval(reversal_count, reversal_trials)
                summary_rows.append({
                    "model": model, "metric": metric,
                    "threshold_from": t1, "threshold_to": t2,
                    "bootstrap_repetitions": len(tr),
                    "defined_ranking_comparisons": int(tr.kendall_tau.notna().sum()),
                    "undefined_ranking_comparisons": int(tr.kendall_tau.isna().sum()),
                    "defined_worst_group_comparisons": reversal_trials,
                    "undefined_worst_group_comparisons": int(tr.worst_group_reversal.isna().sum()),
                    "worst_group_reversal_count": reversal_count,
                    "worst_group_reversal_rate": defined_reversals.mean(),
                    "reversal_rate_mc_ci95_low": mc_low,
                    "reversal_rate_mc_ci95_high": mc_high,
                    "worst_group_set_change_rate": tr.worst_group_set_changed.mean(),
                    "complete_worst_group_reversal_rate": tr.worst_group_complete_reversal.mean(),
                    "worst_group_jaccard_mean": tr.worst_group_jaccard_similarity.mean(),
                    "kendall_tau_mean": valid_tau.mean(), "kendall_tau_sd": valid_tau.std(),
                    "ranking_instability_mean_0_to_2": valid_instability.mean(),
                    "ranking_instability_normalized_mean_0_to_1": tr.ranking_instability_normalized.mean(),
                    "magnitude_change_mean": tr.magnitude_change.mean(),
                    "magnitude_change_sd": tr.magnitude_change.std(),
                    "signed_gap_change_mean": tr.signed_gap_change.mean(),
                    "magnitude_change_q025": tr.magnitude_change.quantile(0.025),
                    "magnitude_change_median": tr.magnitude_change.quantile(0.50),
                    "magnitude_change_q975": tr.magnitude_change.quantile(0.975),
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nCompleted. Output files:")
    for path in [RAW_PATH, TRANSITION_PATH, SUMMARY_PATH, REFERENCE_PATH]:
        print(path)
    print("\nInterpretation boundary: row-level empirical-table bootstrap; shared-identity dependence is not modelled.")
    print("Ranking instability = 1 - Kendall tau (range 0–2); normalized version = (1 - tau) / 2 (range 0–1).")
    print("Reversal-rate intervals quantify Monte Carlo precision of the bootstrap estimate, not population uncertainty.")


if __name__ == "__main__":
    main()
