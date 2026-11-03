import pandas as pd
import numpy as np
from scipy.stats import kendalltau, rankdata
from config import BFW_DATA_PATH, OUTPUT_DIR

# =========================================================
# CONFIG
# =========================================================

MODELS = ["vgg16", "resnet50", "senet50"]
THRESHOLD = 0.50
BOOTSTRAP_REPETITIONS = 10_000
GROUP_COLUMN = "a1"
RANDOM_SEED = 42

# Paired row bootstrap:
# The same sampled row indices are used for all models within each
# repetition. This supports paired model comparisons under identical
# empirical evaluation samples.
#
# The resampling unit remains the comparison row. This estimates
# resampling variability within the empirical comparison table; it
# does not account for dependence among comparisons sharing identities.

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

    if np.isnan(fnr).any() or np.isnan(fpr).any():
        raise ValueError(
            "A group has an undefined metric in a bootstrap replicate. "
            "The current unstratified row bootstrap produced a group with "
            "no positive or no negative observations."
        )

    return fnr, fpr


def ranking_details(metric_values, group_names):
    """
    Return all worst groups, a readable ranking, average-tie rank positions,
    and the max-minus-min group gap.
    """
    values = np.asarray(metric_values, dtype=float)

    if np.isnan(values).any():
        raise ValueError("Undefined group metric encountered.")

    max_value = np.max(values)
    worst_indices = np.flatnonzero(values == max_value)
    worst_groups = [str(group_names[i]) for i in worst_indices]

    rank_positions = rankdata(-values, method="average")

    # Display tied groups at the same rank rather than arbitrarily ordering
    # them as if one had a worse metric value.
    rank_levels = []
    for rank in sorted(np.unique(rank_positions)):
        tied_indices = np.flatnonzero(rank_positions == rank)
        tied_names = [str(group_names[i]) for i in tied_indices]
        rank_levels.append(" = ".join(tied_names))
    ranking_text = " > ".join(rank_levels)

    gap = float(np.max(values) - np.min(values))
    return worst_groups, ranking_text, rank_positions, gap


def compare_rankings(reference_ranks, bootstrap_ranks):
    """Kendall tau-b agreement and unnormalised 1 - tau disagreement."""
    tau, _ = kendalltau(reference_ranks, bootstrap_ranks)
    if np.isnan(tau):
        return np.nan, np.nan
    tau = float(tau)
    return tau, float(1.0 - tau)


def joined(values):
    """Stable CSV representation of a group set."""
    return " | ".join(sorted(str(value) for value in values))


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q))


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

print("\\n" + "=" * 72)
print("FULL-DATA REFERENCE AUDITS")
print("=" * 72)

for model in MODELS:
    fnr, fpr = calculate_metrics(
        combined_codes_by_model[model], number_of_groups
    )

    fnr_worst, fnr_ranking, fnr_ranks, fnr_gap = ranking_details(fnr, groups)
    fpr_worst, fpr_ranking, fpr_ranks, fpr_gap = ranking_details(fpr, groups)

    reference[model] = {
        "fnr_worst_groups": set(fnr_worst),
        "fpr_worst_groups": set(fpr_worst),
        "fnr_ranks": fnr_ranks,
        "fpr_ranks": fpr_ranks,
        "fnr": fnr.copy(),
        "fpr": fpr.copy(),
    }

    print(f"\\n{model.upper()}")
    print(f"  FNR worst group(s): {joined(fnr_worst)}; gap: {fnr_gap:.6f}")
    print(f"  FNR ranking: {fnr_ranking}")
    print(f"  FPR worst group(s): {joined(fpr_worst)}; gap: {fpr_gap:.6f}")
    print(f"  FPR ranking: {fpr_ranking}")

    row = {
        "model": model,
        "threshold": THRESHOLD,
        "fnr_worst_groups": joined(fnr_worst),
        "fnr_worst_group_count": len(fnr_worst),
        "fnr_group_gap": fnr_gap,
        "fnr_ranking_worst_to_best": fnr_ranking,
        "fpr_worst_groups": joined(fpr_worst),
        "fpr_worst_group_count": len(fpr_worst),
        "fpr_group_gap": fpr_gap,
        "fpr_ranking_worst_to_best": fpr_ranking,
    }

    for i, group in enumerate(groups):
        row[f"fnr_{group}"] = float(fnr[i])
        row[f"fpr_{group}"] = float(fpr[i])

    reference_rows.append(row)


# =========================================================
# PAIRED ROW BOOTSTRAP
# =========================================================

rng = np.random.default_rng(RANDOM_SEED)
bootstrap_rows = []

print("\\n" + "=" * 72)
print("PAIRED ROW BOOTSTRAP")
print("=" * 72)

for repetition in range(1, BOOTSTRAP_REPETITIONS + 1):
    # One sample per repetition, shared by all models.
    sample_indices = rng.integers(0, len(df), size=len(df))

    for model in MODELS:
        sampled_codes = combined_codes_by_model[model][sample_indices]
        fnr, fpr = calculate_metrics(sampled_codes, number_of_groups)

        fnr_worst, fnr_ranking, fnr_ranks, fnr_gap = ranking_details(fnr, groups)
        fpr_worst, fpr_ranking, fpr_ranks, fpr_gap = ranking_details(fpr, groups)

        fnr_tau, fnr_instability = compare_rankings(
            reference[model]["fnr_ranks"], fnr_ranks
        )
        fpr_tau, fpr_instability = compare_rankings(
            reference[model]["fpr_ranks"], fpr_ranks
        )

        reference_fnr_worst = reference[model]["fnr_worst_groups"]
        reference_fpr_worst = reference[model]["fpr_worst_groups"]

        current_fnr_worst = set(fnr_worst)
        current_fpr_worst = set(fpr_worst)

        row = {
            "model": model,
            "repetition": repetition,
            "threshold": THRESHOLD,

            "fnr_worst_groups": joined(fnr_worst),
            "fnr_worst_group_count": len(fnr_worst),
            "fnr_worst_group_membership_changed": int(
                current_fnr_worst != reference_fnr_worst
            ),
            "fnr_worst_group_complete_reversal": int(
                current_fnr_worst.isdisjoint(reference_fnr_worst)
            ),
            "fnr_gap": fnr_gap,
            "fnr_kendall_tau_b": fnr_tau,
            "fnr_ranking_disagreement_1_minus_tau": fnr_instability,
            "fnr_ranking_worst_to_best": fnr_ranking,

            "fpr_worst_groups": joined(fpr_worst),
            "fpr_worst_group_count": len(fpr_worst),
            "fpr_worst_group_membership_changed": int(
                current_fpr_worst != reference_fpr_worst
            ),
            "fpr_worst_group_complete_reversal": int(
                current_fpr_worst.isdisjoint(reference_fpr_worst)
            ),
            "fpr_gap": fpr_gap,
            "fpr_kendall_tau_b": fpr_tau,
            "fpr_ranking_disagreement_1_minus_tau": fpr_instability,
            "fpr_ranking_worst_to_best": fpr_ranking,
        }

        # Retain every subgroup metric for every bootstrap repetition.
        for i, group in enumerate(groups):
            row[f"fnr_{group}"] = float(fnr[i])
            row[f"fpr_{group}"] = float(fpr[i])

        bootstrap_rows.append(row)

    if repetition % 500 == 0:
        print(
            f"Completed {repetition:,}/{BOOTSTRAP_REPETITIONS:,} "
            f"repetitions"
        )


# =========================================================
# SAVE RAW BOOTSTRAP RESULTS
# =========================================================

results_df = pd.DataFrame(bootstrap_rows)
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
# BOOTSTRAP SUMMARY
# =========================================================

summary_rows = []
group_summary_rows = []
worst_frequency_rows = []

for model in MODELS:
    model_results = results_df[results_df["model"] == model]

    for metric in ["fnr", "fpr"]:
        gap_values = model_results[f"{metric}_gap"].to_numpy()
        tau_values = model_results[f"{metric}_kendall_tau_b"].to_numpy()
        disagreement_values = model_results[
            f"{metric}_ranking_disagreement_1_minus_tau"
        ].to_numpy()

        reversal_col = f"{metric}_worst_group_membership_changed"
        complete_col = f"{metric}_worst_group_complete_reversal"

        summary_rows.append({
            "model": model,
            "metric": metric.upper(),
            "repetitions": len(model_results),

            "worst_group_membership_change_rate": float(
                model_results[reversal_col].mean()
            ),
            "complete_worst_group_reversal_rate": float(
                model_results[complete_col].mean()
            ),

            "group_gap_mean": float(np.mean(gap_values)),
            "group_gap_sd": float(np.std(gap_values, ddof=1)),
            "group_gap_p2_5": percentile(gap_values, 0.025),
            "group_gap_median": percentile(gap_values, 0.50),
            "group_gap_p97_5": percentile(gap_values, 0.975),

            "kendall_tau_b_mean": float(np.mean(tau_values)),
            "kendall_tau_b_sd": float(np.std(tau_values, ddof=1)),
            "kendall_tau_b_p2_5": percentile(tau_values, 0.025),
            "kendall_tau_b_median": percentile(tau_values, 0.50),
            "kendall_tau_b_p97_5": percentile(tau_values, 0.975),

            "ranking_disagreement_mean": float(np.mean(disagreement_values)),
            "ranking_disagreement_p2_5": percentile(disagreement_values, 0.025),
            "ranking_disagreement_median": percentile(disagreement_values, 0.50),
            "ranking_disagreement_p97_5": percentile(disagreement_values, 0.975),
        })

        # Per-group bootstrap distributions and percentile intervals.
        for group in groups:
            values = model_results[f"{metric}_{group}"].to_numpy()
            group_summary_rows.append({
                "model": model,
                "metric": metric.upper(),
                "group": group,
                "repetitions": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "p2_5": percentile(values, 0.025),
                "median": percentile(values, 0.50),
                "p97_5": percentile(values, 0.975),
            })

            # Membership frequency is not mutually exclusive if a replicate
            # contains tied worst groups. Frequencies can therefore sum > 1.
            worst_col = f"{metric}_worst_groups"
            membership_count = model_results[worst_col].apply(
                lambda value: group in str(value).split(" | ")
            ).sum()

            worst_frequency_rows.append({
                "model": model,
                "metric": metric.upper(),
                "group": group,
                "worst_group_membership_count": int(membership_count),
                "worst_group_membership_frequency": float(
                    membership_count / len(model_results)
                ),
            })

summary_df = pd.DataFrame(summary_rows)
group_summary_df = pd.DataFrame(group_summary_rows)
worst_frequency_df = pd.DataFrame(worst_frequency_rows)

summary_output = OUTPUT_DIR / "bootstrap_baseline_summary_all_models.csv"
group_summary_output = OUTPUT_DIR / "bootstrap_group_metric_summary_all_models.csv"
worst_frequency_output = OUTPUT_DIR / "bootstrap_worst_group_frequency_all_models.csv"

summary_df.to_csv(summary_output, index=False)
group_summary_df.to_csv(group_summary_output, index=False)
worst_frequency_df.to_csv(worst_frequency_output, index=False)


# =========================================================
# SUMMARY PRINT
# =========================================================

print("\\n" + "=" * 72)
print("BOOTSTRAP SUMMARY")
print("=" * 72)

for _, row in summary_df.iterrows():
    print(f"\\n{row['model'].upper()} — {row['metric']}")
    print(
        "  Worst-group membership-change rate: "
        f"{row['worst_group_membership_change_rate']:.4f}"
    )
    print(
        "  Complete worst-group reversal rate: "
        f"{row['complete_worst_group_reversal_rate']:.4f}"
    )
    print(
        "  Group-gap mean (SD): "
        f"{row['group_gap_mean']:.6f} "
        f"({row['group_gap_sd']:.6f})"
    )
    print(
        "  Group-gap 95% percentile interval: "
        f"[{row['group_gap_p2_5']:.6f}, "
        f"{row['group_gap_p97_5']:.6f}]"
    )
    print(
        "  Mean Kendall tau-b: "
        f"{row['kendall_tau_b_mean']:.4f}"
    )
    print(
        "  Kendall tau-b 95% percentile interval: "
        f"[{row['kendall_tau_b_p2_5']:.4f}, "
        f"{row['kendall_tau_b_p97_5']:.4f}]"
    )

print("\\nDONE")
print(f"Combined bootstrap results: {combined_output}")
print(f"Reference audit: {reference_output}")
print(f"Bootstrap summary: {summary_output}")
print(f"Group metric summary: {group_summary_output}")
print(f"Worst-group frequencies: {worst_frequency_output}")
print("Per-model bootstrap files were also saved.")

print("\\nMethodological note:")
print("- This is a paired row-level bootstrap of the empirical BFW comparison table.")
print("- It does not account for dependence among comparisons sharing identities.")
print("- Percentile intervals describe row-bootstrap variability, not identity-population uncertainty.")
print("- The full-data reference is calculated from the same empirical dataset, not an independent test set.")
print("- Ranking disagreement is 1 - Kendall tau-b and ranges from 0 to 2.")
