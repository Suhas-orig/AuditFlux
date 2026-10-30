import pandas as pd
import numpy as np
from scipy.stats import kendalltau, rankdata
from config import BFW_DATA_PATH, OUTPUT_DIR

# =========================================================
# CONFIG
# =========================================================

MODELS = ["vgg16", "resnet50", "senet50"]
THRESHOLD = 0.50
BOOTSTRAP_REPETITIONS = 1000
GROUP_COLUMN = "a1"
RANDOM_SEED = 42

# This is a paired row bootstrap: the SAME sampled row indices
# are used for all three models in each repetition. This makes
# model-to-model comparisons less confounded by different samples.
# It is still a row-level bootstrap, not an identity/subject-level
# bootstrap; see the methodological note printed at the end.


# =========================================================
# HELPERS
# =========================================================

def calculate_metrics(sampled_codes, number_of_groups):
    """Return per-group FNR and FPR from encoded TN/FP/FN/TP counts."""
    counts = np.bincount(
        sampled_codes,
        minlength=number_of_groups * 4
    ).reshape(number_of_groups, 4)

    tn = counts[:, 0].astype(float)
    fp = counts[:, 1].astype(float)
    fn = counts[:, 2].astype(float)
    tp = counts[:, 3].astype(float)

    positive_denominator = tp + fn
    negative_denominator = fp + tn

    fnr = np.divide(
        fn,
        positive_denominator,
        out=np.full(number_of_groups, np.nan),
        where=positive_denominator > 0
    )
    fpr = np.divide(
        fp,
        negative_denominator,
        out=np.full(number_of_groups, np.nan),
        where=negative_denominator > 0
    )
    return fnr, fpr


def ranking_details(metric_values, group_names):
    """Return worst group, descending ranking and average-tie rank positions."""
    values = np.asarray(metric_values, dtype=float)
    if np.isnan(values).any():
        raise ValueError(
            "A group has an undefined metric in a bootstrap replicate. "
            "Increase sample size or use a suitable stratified design."
        )

    order = np.argsort(-values, kind="stable")
    ordered_groups = [group_names[i] for i in order]

    # rankdata assigns rank 1 to the largest value after negation.
    # Average ranks handle exact ties without arbitrary tie-breaking.
    rank_positions = rankdata(-values, method="average")
    worst_group = group_names[order[0]]
    gap = float(np.max(values) - np.min(values))
    return worst_group, ordered_groups, rank_positions, gap


def compare_rankings(reference_ranks, bootstrap_ranks):
    """Kendall agreement between aligned group rank positions."""
    tau, _ = kendalltau(reference_ranks, bootstrap_ranks)
    if np.isnan(tau):
        return np.nan, np.nan
    return float(tau), float(1.0 - tau)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading BFW dataset...")
required_columns = [GROUP_COLUMN, "label", *MODELS]
df = pd.read_csv(BFW_DATA_PATH, usecols=required_columns)

if df[required_columns].isna().any().any():
    raise ValueError("Missing values found in required BFW columns.")

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Expected labels to be encoded as 0 and 1.")

print(f"Dataset size: {len(df):,}")
print(f"Models: {', '.join(MODELS)}")
print(f"Threshold: {THRESHOLD}")
print(f"Bootstrap repetitions: {BOOTSTRAP_REPETITIONS:,}")

groups = sorted(df[GROUP_COLUMN].unique())
number_of_groups = len(groups)
group_to_code = {group: i for i, group in enumerate(groups)}
group_codes = df[GROUP_COLUMN].map(group_to_code).to_numpy(dtype=np.int16)
labels = df["label"].to_numpy(dtype=np.int8)

# Encode each row's group and confusion-matrix outcome separately per model.
# Outcome code: 0=TN, 1=FP, 2=FN, 3=TP.
combined_codes_by_model = {}

for model in MODELS:
    predictions = (df[model].to_numpy() >= THRESHOLD).astype(np.int8)
    outcome_codes = np.zeros(len(df), dtype=np.int8)
    outcome_codes[(labels == 0) & (predictions == 1)] = 1
    outcome_codes[(labels == 1) & (predictions == 0)] = 2
    outcome_codes[(labels == 1) & (predictions == 1)] = 3
    combined_codes_by_model[model] = group_codes * 4 + outcome_codes


# =========================================================
# FULL-DATA REFERENCE AUDITS
# =========================================================

reference = {}
reference_rows = []

print("\n" + "=" * 72)
print("FULL-DATA REFERENCE AUDITS")
print("=" * 72)

for model in MODELS:
    fnr, fpr = calculate_metrics(
        combined_codes_by_model[model], number_of_groups
    )
    fnr_worst, fnr_order, fnr_ranks, fnr_gap = ranking_details(fnr, groups)
    fpr_worst, fpr_order, fpr_ranks, fpr_gap = ranking_details(fpr, groups)

    reference[model] = {
        "fnr_worst": fnr_worst,
        "fpr_worst": fpr_worst,
        "fnr_ranks": fnr_ranks,
        "fpr_ranks": fpr_ranks,
    }

    print(f"\n{model.upper()}")
    print(f"  FNR worst group: {fnr_worst}; gap: {fnr_gap:.6f}")
    print(f"  FNR ranking (worst to best): {' > '.join(fnr_order)}")
    print(f"  FPR worst group: {fpr_worst}; gap: {fpr_gap:.6f}")
    print(f"  FPR ranking (worst to best): {' > '.join(fpr_order)}")

    reference_rows.append({
        "model": model,
        "threshold": THRESHOLD,
        "fnr_worst_group": fnr_worst,
        "fnr_group_gap": fnr_gap,
        "fnr_ranking_worst_to_best": " > ".join(fnr_order),
        "fpr_worst_group": fpr_worst,
        "fpr_group_gap": fpr_gap,
        "fpr_ranking_worst_to_best": " > ".join(fpr_order),
    })


# =========================================================
# PAIRED ROW BOOTSTRAP
# =========================================================

rng = np.random.default_rng(RANDOM_SEED)
results = []

print("\n" + "=" * 72)
print("PAIRED ROW BOOTSTRAP")
print("=" * 72)

for repetition in range(1, BOOTSTRAP_REPETITIONS + 1):
    # Resample N rows with replacement. The same indices are reused
    # for every model within this repetition.
    sample_indices = rng.integers(0, len(df), size=len(df))

    for model in MODELS:
        sampled_codes = combined_codes_by_model[model][sample_indices]
        fnr, fpr = calculate_metrics(sampled_codes, number_of_groups)

        fnr_worst, _, fnr_ranks, fnr_gap = ranking_details(fnr, groups)
        fpr_worst, _, fpr_ranks, fpr_gap = ranking_details(fpr, groups)

        fnr_tau, fnr_instability = compare_rankings(
            reference[model]["fnr_ranks"], fnr_ranks
        )
        fpr_tau, fpr_instability = compare_rankings(
            reference[model]["fpr_ranks"], fpr_ranks
        )

        results.append({
            "model": model,
            "repetition": repetition,
            "threshold": THRESHOLD,
            "fnr_worst_group": fnr_worst,
            "fnr_worst_group_reversal": int(
                fnr_worst != reference[model]["fnr_worst"]
            ),
            "fnr_gap": fnr_gap,
            "fnr_kendall_tau": fnr_tau,
            "fnr_ranking_instability": fnr_instability,
            "fpr_worst_group": fpr_worst,
            "fpr_worst_group_reversal": int(
                fpr_worst != reference[model]["fpr_worst"]
            ),
            "fpr_gap": fpr_gap,
            "fpr_kendall_tau": fpr_tau,
            "fpr_ranking_instability": fpr_instability,
        })

    if repetition % 100 == 0:
        print(f"Completed {repetition:,}/{BOOTSTRAP_REPETITIONS:,} repetitions")


# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(results)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

combined_output = OUTPUT_DIR / "bootstrap_baseline_all_models_corrected.csv"
results_df.to_csv(combined_output, index=False)

for model in MODELS:
    model_output = OUTPUT_DIR / f"bootstrap_baseline_{model}_corrected.csv"
    results_df[results_df["model"] == model].to_csv(model_output, index=False)

reference_df = pd.DataFrame(reference_rows)
reference_output = OUTPUT_DIR / "bootstrap_reference_audit_all_models.csv"
reference_df.to_csv(reference_output, index=False)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 72)
print("BOOTSTRAP SUMMARY")
print("=" * 72)

for model in MODELS:
    model_results = results_df[results_df["model"] == model]
    print(f"\n{model.upper()}")

    for metric in ["fnr", "fpr"]:
        worst_col = f"{metric}_worst_group"
        reversal_col = f"{metric}_worst_group_reversal"
        instability_col = f"{metric}_ranking_instability"
        gap_col = f"{metric}_gap"

        frequencies = model_results[worst_col].value_counts().to_dict()
        reversal_rate = model_results[reversal_col].mean()
        mean_instability = model_results[instability_col].mean()
        instability_percentiles = model_results[instability_col].quantile(
            [0.025, 0.50, 0.975]
        )

        print(f"  {metric.upper()}:")
        print(f"    Worst-group frequencies: {frequencies}")
        print(f"    Worst-group reversal rate: {reversal_rate:.4f}")
        print(f"    Mean ranking instability (1 - Kendall tau): {mean_instability:.4f}")
        print(
            "    Ranking-instability percentiles "
            f"(2.5%, 50%, 97.5%): {instability_percentiles.to_list()}"
        )
        print(
            f"    Group-gap mean: {model_results[gap_col].mean():.6f}; "
            f"SD: {model_results[gap_col].std():.6f}"
        )

print("\nDONE")
print(f"Combined bootstrap results: {combined_output}")
print(f"Reference audit: {reference_output}")
print("Per-model bootstrap files were also saved.")
print("\nMethodological note: this is a row-level bootstrap. If pair records")
print("share identities, row independence may be violated. An identity-aware")
print("cluster bootstrap would be a stronger follow-up if identity IDs exist.")