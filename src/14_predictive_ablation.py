from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import OUTPUT_DIR

INPUT_PATH = OUTPUT_DIR / "11_predictive_pilot_dataset.csv"
PREFIX = "14_predictive_ablation"
RANDOM_SEED = 20261105

REGRESSION_TARGETS = [
    "target_magnitude_change",
    "target_normalized_ranking_instability",
]
CLASSIFICATION_TARGET = "target_worst_group_set_changed"
AUXILIARY_TARGET = "target_decision_flip_rate"

CATEGORICAL = ["model", "metric"]
CONTEXT = CATEGORICAL + [
    "threshold_from", "threshold_to", "threshold_midpoint", "threshold_width"
]


def build_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    # Pilot-only predictors. Fold and split identifiers are never predictors.
    count_cols = [
        c for c in columns
        if c in {"pilot_rows", "pilot_positive_rows", "pilot_negative_rows"}
        or "_n_metric_class" in c
        or "_n_opposite_class" in c
        or "group_count" in c
        or c.endswith("_count_imbalance")
    ]
    distribution_tokens = (
        "_metric_score_mean", "_metric_score_sd", "_metric_score_q10",
        "_metric_score_q50", "_metric_score_q90", "_opposite_score_mean",
        "_opposite_score_sd", "_metric_score_mean_", "_metric_score_sd_",
        "_metric_score_q10_", "_metric_score_q50_", "_metric_score_q90_",
        "_opposite_score_mean_", "_opposite_score_sd_",
    )
    distribution_cols = [
        c for c in columns
        if any(token in c for token in distribution_tokens)
    ]
    proximity_cols = [
        c for c in columns
        if "fraction_between_thresholds" in c
        or "fraction_near_threshold" in c
    ]

    # De-duplicate while preserving column order.
    def unique(items):
        return list(dict.fromkeys(c for c in items if c in columns))

    context = unique(CONTEXT)
    counts = unique(context + count_cols)
    distribution = unique(counts + distribution_cols)
    proximity = unique(counts + proximity_cols)
    full = unique(distribution + proximity_cols)

    return {
        "context_only": context,
        "counts_only": counts,
        "score_distribution_no_proximity": distribution,
        "threshold_proximity_only": proximity,
        "full_pilot_features": full,
    }


def make_pipeline(estimator, numeric_cols, categorical_cols, scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def regression_scores(y, p):
    rho = np.nan
    if len(y) > 2 and np.unique(y).size > 1 and np.unique(p).size > 1:
        value = spearmanr(y, p).statistic
        rho = float(value) if np.isfinite(value) else np.nan
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "spearman": rho,
    }


def classification_scores(y, probability):
    pred = (probability >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "brier": float(brier_score_loss(y, probability)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
        "precision_at_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else np.nan,
        "average_precision": float(average_precision_score(y, probability)),
    }


def run_regression(data, feature_sets):
    logo = LeaveOneGroupOut()
    prediction_rows, metric_rows = [], []
    model_specs = {
        "dummy_mean": (lambda: DummyRegressor(strategy="mean"), False),
        "ridge": (lambda: Ridge(alpha=10.0), True),
        "extra_trees": (lambda: ExtraTreesRegressor(
            n_estimators=25, min_samples_leaf=20, max_features=0.35,
            max_depth=8, n_jobs=4, random_state=RANDOM_SEED
        ), False),
    }

    for target in REGRESSION_TARGETS:
        subset = data.loc[data[target].notna()].reset_index(drop=True)
        y = subset[target].to_numpy(float)
        groups = subset["fold"].to_numpy()
        for set_name, features in feature_sets.items():
            for model_name, (factory, scale) in model_specs.items():
                oof = np.full(len(subset), np.nan)
                for train_idx, test_idx in logo.split(subset[features], y, groups):
                    cat = [c for c in CATEGORICAL if c in features]
                    num = [c for c in features if c not in cat]
                    pipe = make_pipeline(factory(), num, cat, scale)
                    pipe.fit(subset.iloc[train_idx][features], y[train_idx])
                    oof[test_idx] = pipe.predict(subset.iloc[test_idx][features])
                if not np.isfinite(oof).all():
                    raise RuntimeError(f"Missing OOF regression predictions: {target}/{set_name}/{model_name}")

                metric_rows.append({
                    "target": target, "feature_set": set_name,
                    "predictor": model_name, "evaluation": "pooled_leave_one_fold_out",
                    **regression_scores(y, oof),
                })
                for fold in sorted(np.unique(groups)):
                    mask = groups == fold
                    metric_rows.append({
                        "target": target, "feature_set": set_name,
                        "predictor": model_name, "evaluation": f"held_out_fold_{fold}",
                        **regression_scores(y[mask], oof[mask]),
                    })
                for i, row in subset.iterrows():
                    prediction_rows.append({
                        "target": target, "feature_set": set_name,
                        "predictor": model_name, "fold": int(row["fold"]),
                        "split_id": int(row["split_id"]), "model": row["model"],
                        "metric": row["metric"], "threshold_from": row["threshold_from"],
                        "threshold_to": row["threshold_to"],
                        "observed": float(y[i]), "predicted": float(oof[i]),
                    })
                print(f"Regression: {target} | {set_name} | {model_name}", flush=True)
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def run_classification(data, feature_sets):
    logo = LeaveOneGroupOut()
    y = data[CLASSIFICATION_TARGET].to_numpy(int)
    groups = data["fold"].to_numpy()
    prediction_rows, metric_rows = [], []
    model_specs = {
        "dummy_prior": (lambda: DummyClassifier(strategy="prior"), False),
        "logistic_balanced": (lambda: LogisticRegression(
            max_iter=1500, class_weight="balanced", solver="liblinear",
            random_state=RANDOM_SEED
        ), True),
        "extra_trees_balanced": (lambda: ExtraTreesClassifier(
            n_estimators=25, min_samples_leaf=20, max_features=0.35,
            max_depth=8, class_weight="balanced", n_jobs=4,
            random_state=RANDOM_SEED
        ), False),
    }

    for set_name, features in feature_sets.items():
        for model_name, (factory, scale) in model_specs.items():
            probability = np.full(len(data), np.nan)
            for train_idx, test_idx in logo.split(data[features], y, groups):
                cat = [c for c in CATEGORICAL if c in features]
                num = [c for c in features if c not in cat]
                pipe = make_pipeline(factory(), num, cat, scale)
                pipe.fit(data.iloc[train_idx][features], y[train_idx])
                probability[test_idx] = pipe.predict_proba(data.iloc[test_idx][features])[:, 1]
            if not np.isfinite(probability).all():
                raise RuntimeError(f"Missing OOF classification predictions: {set_name}/{model_name}")

            metric_rows.append({
                "target": CLASSIFICATION_TARGET, "feature_set": set_name,
                "predictor": model_name, "evaluation": "pooled_leave_one_fold_out",
                **classification_scores(y, probability),
            })
            for fold in sorted(np.unique(groups)):
                mask = groups == fold
                metric_rows.append({
                    "target": CLASSIFICATION_TARGET, "feature_set": set_name,
                    "predictor": model_name, "evaluation": f"held_out_fold_{fold}",
                    **classification_scores(y[mask], probability[mask]),
                })
            for i, row in data.iterrows():
                prediction_rows.append({
                    "target": CLASSIFICATION_TARGET, "feature_set": set_name,
                    "predictor": model_name, "fold": int(row["fold"]),
                    "split_id": int(row["split_id"]), "model": row["model"],
                    "metric": row["metric"], "threshold_from": row["threshold_from"],
                    "threshold_to": row["threshold_to"], "observed": int(y[i]),
                    "probability": float(probability[i]),
                })
            print(f"Classification: {set_name} | {model_name}", flush=True)
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run Program 11 first and retain its predictive dataset."
        )
    data = pd.read_csv(INPUT_PATH)
    required = {
        "fold", "split_id", "model", "metric", "threshold_from", "threshold_to",
        *REGRESSION_TARGETS, CLASSIFICATION_TARGET, AUXILIARY_TARGET,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Program 11 dataset is missing required columns: {missing}")

    # Explicitly exclude all target-derived fields and all fold/split identifiers.
    excluded = {
        "fold", "split_id", "target_magnitude_change",
        "target_normalized_ranking_instability", "target_decision_flip_rate",
        "target_worst_group_set_changed", "target_worst_group_from",
        "target_worst_group_to", "target_kendall_tau_b",
        "target_ranking_instability", "target_gap_from", "target_gap_to",
        "target_signed_gap_change",
    }
    candidate_columns = [c for c in data.columns if c not in excluded]
    feature_sets = build_feature_sets(candidate_columns)

    # The direct decision-flip target is analysed separately. Its total
    # threshold-band feature is an almost direct pilot analogue, so it is
    # excluded from this confirmatory ablation rather than headlining it.
    reg_pred, reg_metrics = run_regression(data, feature_sets)
    cls_pred, cls_metrics = run_classification(data, feature_sets)

    reg_pred.to_csv(OUTPUT_DIR / f"{PREFIX}_regression_oof_predictions.csv", index=False)
    reg_metrics.to_csv(OUTPUT_DIR / f"{PREFIX}_regression_metrics.csv", index=False)
    cls_pred.to_csv(OUTPUT_DIR / f"{PREFIX}_classification_oof_predictions.csv", index=False)
    cls_metrics.to_csv(OUTPUT_DIR / f"{PREFIX}_classification_metrics.csv", index=False)

    # Decision-flip diagnostic: compare the direct proxy against a model
    # without any threshold-proximity features, to expose the proxy effect.
    flip_rows = []
    flip_target = data[AUXILIARY_TARGET].to_numpy(float)
    for set_name in ["score_distribution_no_proximity", "full_pilot_features"]:
        features = feature_sets[set_name]
        for model_name, factory, scale in [
            ("ridge", lambda: Ridge(alpha=10.0), True),
            ("extra_trees", lambda: ExtraTreesRegressor(
                n_estimators=25, min_samples_leaf=20, max_features=0.35,
                max_depth=8, n_jobs=4, random_state=RANDOM_SEED
            ), False),
        ]:
            oof = np.full(len(data), np.nan)
            groups = data["fold"].to_numpy()
            for train_idx, test_idx in LeaveOneGroupOut().split(data[features], flip_target, groups):
                cat = [c for c in CATEGORICAL if c in features]
                num = [c for c in features if c not in cat]
                pipe = make_pipeline(factory(), num, cat, scale)
                pipe.fit(data.iloc[train_idx][features], flip_target[train_idx])
                oof[test_idx] = pipe.predict(data.iloc[test_idx][features])
            flip_rows.append({
                "target": AUXILIARY_TARGET, "feature_set": set_name,
                "predictor": model_name, "evaluation": "pooled_leave_one_fold_out",
                **regression_scores(flip_target, oof),
            })
    pd.DataFrame(flip_rows).to_csv(OUTPUT_DIR / f"{PREFIX}_decision_flip_diagnostic.csv", index=False)

    manifest = {
        "experiment": "Program 14 - Predictive Pilot Feature Ablation and Leakage-Sensitivity Analysis",
        "input": str(INPUT_PATH),
        "input_rows": int(len(data)),
        "outer_validation": "Leave-one-original-BFW-fold-out; all repeated splits within a fold remain together",
        "outer_groups": sorted(int(x) for x in data["fold"].unique()),
        "feature_sets": {k: v for k, v in feature_sets.items()},
        "regression_targets": REGRESSION_TARGETS,
        "classification_target": CLASSIFICATION_TARGET,
        "auxiliary_target": AUXILIARY_TARGET,
        "models": ["dummy_mean", "ridge", "extra_trees", "dummy_prior", "logistic_balanced", "extra_trees_balanced"],
        "random_seed": RANDOM_SEED,
        "interpretation_notes": [
            "The primary question is whether pilot-only score and count summaries add predictive value beyond context metadata.",
            "Fold and split identifiers are never predictors; all splits from an original fold are held out together.",
            "Only five original folds are available, so pooled OOF metrics are descriptive and do not justify narrow inferential confidence intervals.",
            "Repeated splits within a fold are dependent and are not treated as independent validation units.",
            "Decision-flip rate is auxiliary because threshold-band fractions are a direct pilot proxy for the same event; it is not the headline predictive contribution.",
            "This is within-BFW, held-out-identity feasibility, not external dataset or unseen-architecture validation.",
            "No hyperparameter tuning is performed on the held-out folds; estimator settings are fixed in advance.",
        ],
        "outputs": [
            f"{PREFIX}_regression_oof_predictions.csv",
            f"{PREFIX}_regression_metrics.csv",
            f"{PREFIX}_classification_oof_predictions.csv",
            f"{PREFIX}_classification_metrics.csv",
            f"{PREFIX}_decision_flip_diagnostic.csv",
            f"{PREFIX}_manifest.json",
        ],
    }
    with (OUTPUT_DIR / f"{PREFIX}_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nProgram 14 completed.")
    print(f"Rows: {len(data):,}; outer folds: {len(manifest['outer_groups'])}")
    print("Feature sets:", ", ".join(feature_sets))
    print("Saved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
