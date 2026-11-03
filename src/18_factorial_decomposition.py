from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Keep the same project configuration convention as Programs 16 and 17.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import OUTPUT_DIR

INPUT_NAME = "16_architecture_transfer_dataset.csv"
PREFIX = "18_factorial_decomposition"
SEED = 20261109
BOOTSTRAP_REPS = 500

EXPECTED_MODELS = {"vgg16", "resnet50", "senet50"}
EXPECTED_METRICS = {"FNR", "FPR"}
EXPECTED_FOLDS = {1, 2, 3, 4, 5}
EXPECTED_TRANSITIONS = {(0.90, 0.95), (0.95, 0.975), (0.975, 0.99)}
CONTINUOUS_TARGETS = [
    "target_magnitude_change",
    "target_normalized_ranking_instability",
]
BINARY_TARGET = "target_worst_group_set_changed"


def transition_name(row: pd.Series) -> str:
    return f"{row['fpr_quantile_from']:.3f}->{row['fpr_quantile_to']:.3f}"


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "fold", "split_id", "model", "metric", "fpr_quantile_from",
        "fpr_quantile_to", *CONTINUOUS_TARGETS, BINARY_TARGET,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if len(df) != 1800:
        raise ValueError(f"Expected 1,800 Program 16 rows; found {len(df)}")
    if set(df["model"].unique()) != EXPECTED_MODELS:
        raise ValueError(f"Unexpected model levels: {sorted(df['model'].unique())}")
    if set(df["metric"].unique()) != EXPECTED_METRICS:
        raise ValueError(f"Unexpected metric levels: {sorted(df['metric'].unique())}")
    if set(df["fold"].unique()) != EXPECTED_FOLDS:
        raise ValueError(f"Unexpected fold levels: {sorted(df['fold'].unique())}")
    transitions = set(zip(df.fpr_quantile_from, df.fpr_quantile_to))
    if transitions != EXPECTED_TRANSITIONS:
        raise ValueError(f"Unexpected threshold transitions: {transitions}")
    if df[CONTINUOUS_TARGETS + [BINARY_TARGET]].isna().any().any():
        raise ValueError("Missing target values detected; refusing silent row deletion.")
    if not set(df[BINARY_TARGET].unique()).issubset({0, 1}):
        raise ValueError("Worst-group-change target must be binary (0/1).")

    # The intended complete crossed/repeated design has one observation per cell.
    cell = df.groupby(["fold", "split_id", "model", "metric", "fpr_quantile_from", "fpr_quantile_to"]).size()
    if not (cell == 1).all() or len(cell) != len(df):
        raise ValueError("Duplicate or missing fold/split/model/metric/transition records.")
    panel_counts = df.groupby(["fold", "split_id"]).size()
    if len(panel_counts) != 100 or not (panel_counts == 18).all():
        raise ValueError("Expected 100 identity-panel blocks with 18 repeated records each.")


def type3_effect_sizes(ols_fit) -> pd.DataFrame:
    """Descriptive Type-III effect sizes; deliberately no inferential p-values.

    The repeated identity-panel splits overlap in their identities. Therefore,
    conventional row-independent ANOVA p-values and cluster-robust p-values
    would overstate the amount of independent replication. We retain only
    descriptive sums of squares and partial eta-squared effect sizes.
    """
    table = anova_lm(ols_fit, typ=3).reset_index().rename(columns={"index": "term"})
    residual = table.loc[table.term == "Residual", "sum_sq"].iloc[0]
    rows = []
    for _, r in table.iterrows():
        if r["term"] in {"Residual", "Intercept"}:
            continue
        ss = float(r["sum_sq"])
        rows.append({
            "term": r["term"],
            "df_term": float(r["df"]),
            "type3_sum_squares": ss,
            "partial_eta_squared_descriptive": ss / (ss + residual) if ss + residual > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def fit_continuous(sub: pd.DataFrame, target: str, metric: str):
    formula = f"{target} ~ C(fold) * C(model) * C(transition)"
    fit = smf.ols(formula, data=sub).fit()
    effects = type3_effect_sizes(fit)
    effects.insert(0, "metric", metric)
    effects.insert(1, "target", target)
    effects["n"] = len(sub)
    effects["identity_panel_splits"] = sub.panel_id.nunique()
    model_row = {
        "metric": metric,
        "target": target,
        "n": len(sub),
        "identity_panel_splits": sub.panel_id.nunique(),
        "r_squared_descriptive": fit.rsquared,
        "adjusted_r_squared_descriptive": fit.rsquared_adj,
        "residual_sd_descriptive": float(np.sqrt(fit.mse_resid)),
    }
    return fit, effects, model_row


def fit_binary(sub: pd.DataFrame, metric: str):
    # Linear probability model with pairwise interactions. It is used only for
    # descriptive decomposition of event-rate variation, not inferential testing.
    formula = (
        f"{BINARY_TARGET} ~ C(fold) + C(model) + C(transition)"
        "+ C(fold):C(model) + C(fold):C(transition) + C(model):C(transition)"
    )
    fit = smf.ols(formula, data=sub).fit()
    effects = type3_effect_sizes(fit)
    effects.insert(0, "metric", metric)
    effects.insert(1, "target", BINARY_TARGET)
    effects["n"] = len(sub)
    effects["identity_panel_splits"] = sub.panel_id.nunique()
    summary = {
        "metric": metric,
        "target": BINARY_TARGET,
        "model_type": "descriptive_linear_probability_factorial_model",
        "n": len(sub),
        "identity_panel_splits": sub.panel_id.nunique(),
        "event_rate": float(sub[BINARY_TARGET].mean()),
        "r_squared_descriptive": fit.rsquared,
        "adjusted_r_squared_descriptive": fit.rsquared_adj,
        "residual_sd_descriptive": float(np.sqrt(fit.mse_resid)),
    }
    return fit, effects, summary


def bootstrap_marginal_effects(df: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """Fast stratified cluster bootstrap over the 100 identity-panel blocks.

    A sampled panel carries its complete architecture x transition repeated
    measurements. Multiplicity weights avoid repeatedly copying/grouping rows.
    """
    rng = np.random.default_rng(seed)
    folds = ["1", "2", "3", "4", "5"]
    models = ["vgg16", "resnet50", "senet50"]
    transitions = ["0.900->0.950", "0.950->0.975", "0.975->0.990"]
    panel_meta = (
        df[["panel_id", "fold"]].drop_duplicates().sort_values(["fold", "panel_id"])
        .reset_index(drop=True)
    )
    panel_ids = panel_meta.panel_id.tolist()
    fold_values = panel_meta.fold.astype(str).to_numpy()
    fold_positions = {f: np.flatnonzero(fold_values == f) for f in folds}

    # Each target matrix is 100 panels x 9 within-panel repeated cells.
    matrices = {}
    for metric in ["FNR", "FPR"]:
        part = df[df.metric == metric]
        for target in CONTINUOUS_TARGETS + [BINARY_TARGET]:
            pivot = part.pivot(index="panel_id", columns=["model", "transition"], values=target)
            pivot = pivot.reindex(index=panel_ids)
            pivot = pivot.reindex(columns=pd.MultiIndex.from_product([models, transitions]))
            matrices[(metric, target)] = pivot.to_numpy(dtype=float)

    records = []
    for b in range(reps):
        weights = np.zeros(len(panel_ids), dtype=float)
        for fold in folds:
            pos = fold_positions[fold]
            draws = rng.choice(pos, size=len(pos), replace=True)
            weights += np.bincount(draws, minlength=len(panel_ids))

        for (metric, target), mat in matrices.items():
            weighted_cell_sums = weights @ mat
            # Model marginal means: average 3 transitions per sampled panel.
            for mi, model in enumerate(models):
                cols = slice(mi * 3, (mi + 1) * 3)
                estimate = weighted_cell_sums[cols].sum() / (weights.sum() * 3)
                records.append({"bootstrap_rep": b, "metric": metric, "target": target,
                                "effect_type": "marginal_mean:model", "level": model,
                                "estimate": float(estimate)})
            # Transition marginal means: average 3 architectures per panel.
            for ti, transition in enumerate(transitions):
                cols = [mi * 3 + ti for mi in range(3)]
                estimate = weighted_cell_sums[cols].sum() / (weights.sum() * 3)
                records.append({"bootstrap_rep": b, "metric": metric, "target": target,
                                "effect_type": "marginal_mean:transition", "level": transition,
                                "estimate": float(estimate)})
            # Fold marginal means: resampling is within fold, 20 panels x 9 cells.
            for fold in folds:
                pos = fold_positions[fold]
                estimate = (weights[pos] @ mat[pos, :].sum(axis=1)) / (weights[pos].sum() * 9)
                records.append({"bootstrap_rep": b, "metric": metric, "target": target,
                                "effect_type": "marginal_mean:fold", "level": fold,
                                "estimate": float(estimate)})
            # Architecture x transition cell means.
            for mi, model in enumerate(models):
                for ti, transition in enumerate(transitions):
                    col = mi * 3 + ti
                    estimate = weighted_cell_sums[col] / weights.sum()
                    records.append({"bootstrap_rep": b, "metric": metric, "target": target,
                                    "effect_type": "cell_mean:model_x_transition",
                                    "level": f"{model}|{transition}", "estimate": float(estimate)})

    boot_df = pd.DataFrame(records)
    summary = (
        boot_df.groupby(["metric", "target", "effect_type", "level"], as_index=False)
        .agg(bootstrap_mean=("estimate", "mean"),
             split_resampling_p2_5=("estimate", lambda x: np.quantile(x, 0.025)),
             split_resampling_p97_5=("estimate", lambda x: np.quantile(x, 0.975)),
             bootstrap_sd=("estimate", "std"),
             bootstrap_repetitions=("bootstrap_rep", "nunique"))
    )
    return summary


def main():
    input_path = OUTPUT_DIR / INPUT_NAME
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find {input_path}. Run Program 16 first and keep its output in OUTPUT_DIR."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    validate_input(df)

    df = df.copy()
    df["fold"] = df["fold"].astype(int).astype(str)
    df["split_id"] = df["split_id"].astype(int)
    df["transition"] = df.apply(transition_name, axis=1)
    df["panel_id"] = df["fold"].astype(str) + "_split_" + df["split_id"].astype(str)
    df["transition"] = pd.Categorical(
        df["transition"],
        categories=["0.900->0.950", "0.950->0.975", "0.975->0.990"],
        ordered=True,
    )
    df["model"] = pd.Categorical(df["model"], categories=["vgg16", "resnet50", "senet50"], ordered=True)
    df["fold"] = pd.Categorical(df["fold"], categories=["1", "2", "3", "4", "5"], ordered=True)

    effect_tables, fit_summaries = [], []
    cell_rows = []
    for metric, sub in df.groupby("metric", observed=True):
        for target in CONTINUOUS_TARGETS:
            fit, term_tests, model_summary = fit_continuous(sub, target, metric)
            effect_tables.append(term_tests)
            fit_summaries.append(model_summary)
        fit, term_tests, model_summary = fit_binary(sub, metric)
        effect_tables.append(term_tests)
        fit_summaries.append(model_summary)

        cell = (
            sub.groupby(["fold", "model", "transition"], observed=True)
            .agg(n=(BINARY_TARGET, "size"),
                 magnitude_mean=("target_magnitude_change", "mean"),
                 magnitude_sd=("target_magnitude_change", "std"),
                 ranking_mean=("target_normalized_ranking_instability", "mean"),
                 ranking_sd=("target_normalized_ranking_instability", "std"),
                 worst_group_change_rate=(BINARY_TARGET, "mean"))
            .reset_index()
        )
        cell.insert(0, "metric", metric)
        cell_rows.append(cell)

    effects_df = pd.concat(effect_tables, ignore_index=True)
    fit_df = pd.DataFrame(fit_summaries)
    cells_df = pd.concat(cell_rows, ignore_index=True)

    # Marginal means and interaction-cell estimates with split-panel bootstrap CIs.
    boot_summary = bootstrap_marginal_effects(df, BOOTSTRAP_REPS, SEED)

    effects_path = OUTPUT_DIR / f"{PREFIX}_factor_effect_sizes.csv"
    fit_path = OUTPUT_DIR / f"{PREFIX}_model_fit_summary.csv"
    cell_path = OUTPUT_DIR / f"{PREFIX}_cell_summary.csv"
    boot_path = OUTPUT_DIR / f"{PREFIX}_split_resampling_sensitivity.csv"
    effects_df.to_csv(effects_path, index=False)
    fit_df.to_csv(fit_path, index=False)
    cells_df.to_csv(cell_path, index=False)
    boot_summary.to_csv(boot_path, index=False)

    manifest = {
        "experiment": "Program 18 - Factorial Decomposition of Audit Instability",
        "input": str(input_path),
        "input_rows": int(len(df)),
        "input_source": "Program 16 pilot/target identity-disjoint panel dataset; no BFW score recalculation",
        "design": {
            "architectures": sorted(df.model.astype(str).unique().tolist()),
            "folds": sorted(df.fold.astype(str).unique().tolist()),
            "threshold_transitions": sorted(df.transition.astype(str).unique().tolist()),
            "metrics_analysed_separately": sorted(df.metric.unique().tolist()),
            "identity_panel_blocks": int(df.panel_id.nunique()),
            "repeated_records_per_panel": 18,
            "observations_per_metric": 900,
            "observations_per_factorial_cell": 20,
            "factorial_model": "full fold x architecture x threshold-transition factorial, separately by metric and target",
            "continuous_analysis": "full factorial OLS; Type-III sums of squares and partial eta-squared reported descriptively",
            "binary_analysis": "linear probability factorial model with main effects and pairwise interactions; descriptive only",
            "resampling_sensitivity": f"{BOOTSTRAP_REPS} fold-stratified split-resampling repetitions; resample 20 split panels within each fixed fold and retain all architecture x transition records",
        },
        "targets": CONTINUOUS_TARGETS + [BINARY_TARGET],
        "interpretation": {
            "main_effects": "differences across the observed levels of architecture, fold, and threshold transition",
            "interactions": "whether threshold-transition effects differ by architecture and/or fold",
            "partial_eta_squared": "conditional descriptive effect size; values are not additive independent variance shares",
            "split_resampling_percentiles": "sensitivity ranges across resampled split panels, not confidence intervals or population-level inferential intervals",
        },
        "limitations": [
            "Only three architectures, five predefined BFW folds, and three pilot-calibrated threshold transitions are represented.",
            "The 20 identity-panel splits within each fold overlap in their identities and are not independent clusters. Consequently, no inferential p-values are reported; split-resampling percentiles are sensitivity summaries only.",
            "Fold and architecture are treated as fixed observed levels; results do not establish general effects over all possible architectures or datasets.",
            "FNR and FPR targets are analysed separately because their scales and operating characteristics differ.",
            "The binary worst-group-change model omits the 3-way interaction because the event is sparse; all binary effect sizes are descriptive.",
            "This analysis decomposes variation in Program 16 target outcomes; it does not establish causal effects of architecture, fold, or threshold choice.",
            "Underlying face-recognition scores and pilot-calibrated threshold construction are inherited from Program 16.",
        ],
        "random_seed": SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "outputs": {
            "factor_effect_sizes": effects_path.name,
            "model_fit_summary": fit_path.name,
            "cell_summary": cell_path.name,
            "split_resampling_sensitivity": boot_path.name,
        },
    }
    manifest_path = OUTPUT_DIR / f"{PREFIX}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Program 18 completed successfully.")
    print(f"Input rows: {len(df):,}")
    print(f"Panel clusters: {df.panel_id.nunique():,}")
    print(f"Factor-effect rows: {len(effects_df):,}")
    print(f"Bootstrap summary rows: {len(boot_summary):,}")
    print("Outputs:")
    for p in [effects_path, fit_path, cell_path, boot_path, manifest_path]:
        print(f" - {p}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
