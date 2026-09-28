import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from config import BFW_DATA_PATH, OUTPUT_DIR


# =========================================================
# CONFIG
# =========================================================

MODEL = "resnet50"

THRESHOLD = 0.50

BOOTSTRAP_REPETITIONS = 1000

GROUP_COLUMN = "a1"

RANDOM_SEED = 42


# =========================================================
# LOAD DATA
# =========================================================

print("Loading BFW dataset...")

df = pd.read_csv(
    BFW_DATA_PATH,
    usecols=[GROUP_COLUMN, "label", MODEL]
)

print(f"Dataset size: {len(df):,}")


# =========================================================
# PREPARE GROUPS
# =========================================================

groups = sorted(df[GROUP_COLUMN].unique())

group_to_code = {
    group: i
    for i, group in enumerate(groups)
}

group_codes = df[GROUP_COLUMN].map(group_to_code).to_numpy()

labels = df["label"].to_numpy()

predictions = (
    df[MODEL].to_numpy() >= THRESHOLD
).astype(np.int8)


# =========================================================
# CREATE FOUR OUTCOME CATEGORIES
#
# 0 = TN
# 1 = FP
# 2 = FN
# 3 = TP
# =========================================================

outcome_codes = np.zeros(
    len(df),
    dtype=np.int8
)

# True negative
outcome_codes[
    (labels == 0) & (predictions == 0)
] = 0

# False positive
outcome_codes[
    (labels == 0) & (predictions == 1)
] = 1

# False negative
outcome_codes[
    (labels == 1) & (predictions == 0)
] = 2

# True positive
outcome_codes[
    (labels == 1) & (predictions == 1)
] = 3


# Combine group + outcome into one category.
# There are 8 groups × 4 outcomes = 32 categories.

combined_codes = (
    group_codes * 4
    + outcome_codes
)


# =========================================================
# REFERENCE AUDIT
#
# This is the audit performed on the complete dataset.
# Bootstrap rankings will be compared against this.
# =========================================================

counts = np.bincount(
    combined_codes,
    minlength=len(groups) * 4
).reshape(len(groups), 4)

TN = counts[:, 0]
FP = counts[:, 1]
FN = counts[:, 2]
TP = counts[:, 3]

reference_TPR = TP / (TP + FN)
reference_FNR = FN / (TP + FN)
reference_FPR = FP / (FP + TN)

reference_fnr_ranking = np.argsort(
    -reference_FNR
)

reference_fpr_ranking = np.argsort(
    -reference_FPR
)

reference_fnr_worst = groups[
    reference_fnr_ranking[0]
]

reference_fpr_worst = groups[
    reference_fpr_ranking[0]
]


# =========================================================
# DISPLAY REFERENCE AUDIT
# =========================================================

print("\n" + "=" * 60)
print("REFERENCE AUDIT")
print("=" * 60)

print(f"Threshold: {THRESHOLD}")

print(
    f"FNR worst group: "
    f"{reference_fnr_worst}"
)

print(
    f"FPR worst group: "
    f"{reference_fpr_worst}"
)

print("\nFNR ranking:")

print(
    " > ".join(
        groups[i]
        for i in reference_fnr_ranking
    )
)

print("\nFPR ranking:")

print(
    " > ".join(
        groups[i]
        for i in reference_fpr_ranking
    )
)


# =========================================================
# BOOTSTRAP
# =========================================================

rng = np.random.default_rng(RANDOM_SEED)

results = []

print("\n" + "=" * 60)
print("BOOTSTRAP")
print("=" * 60)

for repetition in range(
    1,
    BOOTSTRAP_REPETITIONS + 1
):

    # -----------------------------------------------------
    # Sample N rows WITH replacement.
    # N = original dataset size.
    # -----------------------------------------------------

    sample_indices = rng.integers(
        0,
        len(df),
        size=len(df)
    )

    sampled_codes = combined_codes[
        sample_indices
    ]


    # -----------------------------------------------------
    # Count TN / FP / FN / TP for every group
    # -----------------------------------------------------

    bootstrap_counts = np.bincount(
        sampled_codes,
        minlength=len(groups) * 4
    ).reshape(len(groups), 4)

    TN = bootstrap_counts[:, 0]
    FP = bootstrap_counts[:, 1]
    FN = bootstrap_counts[:, 2]
    TP = bootstrap_counts[:, 3]


    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    TPR = TP / (TP + FN)

    FNR = FN / (TP + FN)

    FPR = FP / (FP + TN)


    # -----------------------------------------------------
    # FNR ranking
    # -----------------------------------------------------

    fnr_ranking = np.argsort(-FNR)

    fnr_worst = groups[
        fnr_ranking[0]
    ]

    fnr_gap = (
        FNR.max() - FNR.min()
    )

    # Compare bootstrap ranking to reference ranking

    fnr_tau, _ = kendalltau(
        reference_fnr_ranking,
        fnr_ranking
    )

    fnr_ranking_instability = (
        1 - fnr_tau
    )


    # -----------------------------------------------------
    # FPR ranking
    # -----------------------------------------------------

    fpr_ranking = np.argsort(-FPR)

    fpr_worst = groups[
        fpr_ranking[0]
    ]

    fpr_gap = (
        FPR.max() - FPR.min()
    )

    fpr_tau, _ = kendalltau(
        reference_fpr_ranking,
        fpr_ranking
    )

    fpr_ranking_instability = (
        1 - fpr_tau
    )


    # -----------------------------------------------------
    # Store one row per bootstrap repetition
    # -----------------------------------------------------

    results.append({

        "repetition": repetition,

        # FNR

        "fnr_worst_group": fnr_worst,

        "fnr_worst_group_reversal": int(
            fnr_worst != reference_fnr_worst
        ),

        "fnr_gap": fnr_gap,

        "fnr_kendall_tau": fnr_tau,

        "fnr_ranking_instability":
            fnr_ranking_instability,

        # FPR

        "fpr_worst_group": fpr_worst,

        "fpr_worst_group_reversal": int(
            fpr_worst != reference_fpr_worst
        ),

        "fpr_gap": fpr_gap,

        "fpr_kendall_tau": fpr_tau,

        "fpr_ranking_instability":
            fpr_ranking_instability
    })


    if repetition % 100 == 0:

        print(
            f"Completed "
            f"{repetition}/{BOOTSTRAP_REPETITIONS}"
        )


# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(results)

OUTPUT_PATH = (
    OUTPUT_DIR / "bootstrap_baseline_resnet50.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("BOOTSTRAP SUMMARY")
print("=" * 60)


# ---------------------------------------------------------
# Worst-group frequencies
# ---------------------------------------------------------

print("\nFNR worst-group frequency:")

print(
    results_df[
        "fnr_worst_group"
    ].value_counts()
)


print("\nFPR worst-group frequency:")

print(
    results_df[
        "fpr_worst_group"
    ].value_counts()
)


# ---------------------------------------------------------
# Reversal rates
# ---------------------------------------------------------

fnr_reversal_rate = (
    results_df[
        "fnr_worst_group_reversal"
    ].mean()
)

fpr_reversal_rate = (
    results_df[
        "fpr_worst_group_reversal"
    ].mean()
)


print(
    f"\nFNR worst-group reversal rate: "
    f"{fnr_reversal_rate:.4f}"
)

print(
    f"FPR worst-group reversal rate: "
    f"{fpr_reversal_rate:.4f}"
)


# ---------------------------------------------------------
# Ranking instability
# ---------------------------------------------------------

print("\nAverage ranking instability:")

print(
    f"FNR: "
    f"{results_df['fnr_ranking_instability'].mean():.4f}"
)

print(
    f"FPR: "
    f"{results_df['fpr_ranking_instability'].mean():.4f}"
)


# ---------------------------------------------------------
# Gap variability
# ---------------------------------------------------------

print("\nGap statistics:")

print(
    f"FNR gap mean: "
    f"{results_df['fnr_gap'].mean():.6f}"
)

print(
    f"FNR gap SD: "
    f"{results_df['fnr_gap'].std():.6f}"
)

print(
    f"FPR gap mean: "
    f"{results_df['fpr_gap'].mean():.6f}"
)

print(
    f"FPR gap SD: "
    f"{results_df['fpr_gap'].std():.6f}"
)


# ---------------------------------------------------------
# Percentiles
# ---------------------------------------------------------

print("\nFNR ranking instability percentiles:")

print(
    results_df[
        "fnr_ranking_instability"
    ].quantile(
        [0.025, 0.50, 0.975]
    )
)


print("\nFPR ranking instability percentiles:")

print(
    results_df[
        "fpr_ranking_instability"
    ].quantile(
        [0.025, 0.50, 0.975]
    )
)


print("\nDONE")
print(f"Saved to: {OUTPUT_PATH}")