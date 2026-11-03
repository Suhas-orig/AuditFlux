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
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import OUTPUT_DIR

INPUT_PATH = OUTPUT_DIR / "11_predictive_pilot_dataset.csv"
PREFIX = "15_cross_context_generalization"
RANDOM_SEED = 20261106

REGRESSION_TARGETS = [
    "target_magnitude_change",
    "target_normalized_ranking_instability",
]
CLASSIFICATION_TARGET = "target_worst_group_set_changed"
CATEGORICAL = ["model", "metric"]
CONTEXT = CATEGORICAL + [
    "threshold_from", "threshold_to", "threshold_midpoint", "threshold_width"
]


def build_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    count_cols = [
        c for c in columns
        if c in {"pilot_rows", "pilot_positive_rows", "pilot_negative_rows"}
        or "_n_metric_class" in c or "_n_opposite_class" in c
        or "group_count" in c or c.endswith("_count_imbalance")
    ]
    distribution_tokens = (
        "_metric_score_mean", "_metric_score_sd", "_metric_score_q10",
        "_metric_score_q50", "_metric_score_q90", "_opposite_score_mean",
        "_opposite_score_sd", "_metric_score_mean_", "_metric_score_sd_",
        "_metric_score_q10_", "_metric_score_q50_", "_metric_score_q90_",
        "_opposite_score_mean_", "_opposite_score_sd_",
    )
    distribution_cols = [c for c in columns if any(t in c for t in distribution_tokens)]
    proximity_cols = [
        c for c in columns
        if "fraction_between_thresholds" in c or "fraction_near_threshold" in c
    ]

    def unique(items):
        return list(dict.fromkeys(c for c in items if c in columns))

    context = unique(CONTEXT)
    counts = unique(context + count_cols)
    distribution = unique(counts + distribution_cols)
    proximity = unique(counts + proximity_cols)
    full = unique(distribution + proximity_cols)
    # Program 14 already compared all five feature families. Program 15
    # focuses on the three most informative contrasts to keep the stricter
    # crossed validation computationally manageable:
    # context baseline, threshold-proximity features, and the full pilot set.
    return {
        "context_only": context,
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


def reg_scores(y, p):
    rho = np.nan
    if len(y) > 2 and np.unique(y).size > 1 and np.unique(p).size > 1:
        value = spearmanr(y, p).statistic
        rho = float(value) if np.isfinite(value) else np.nan
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.unique(y).size > 1 else np.nan,
        "spearman": rho,
    }


def cls_scores(y, prob):
    hard = (prob >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "brier": float(brier_score_loss(y, prob)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, hard)),
        "precision_at_0_5": float(precision_score(y, hard, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, hard, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)) if np.unique(y).size == 2 else np.nan,
        "average_precision": float(average_precision_score(y, prob)),
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing {INPUT_PATH}. Run Program 11 first.")
    data = pd.read_csv(INPUT_PATH)
    required = {
        "fold", "split_id", "model", "metric", "threshold_from", "threshold_to",
        *REGRESSION_TARGETS, CLASSIFICATION_TARGET,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Program 11 dataset is missing required columns: {missing}")

    data = data.copy()
    data["fold"] = data["fold"].astype(int)
    data["transition_id"] = (
        data["threshold_from"].map(lambda x: f"{float(x):.2f}")
        + "_to_"
        + data["threshold_to"].map(lambda x: f"{float(x):.2f}")
    )
    transitions = sorted(data["transition_id"].unique())
    folds = sorted(data["fold"].unique())
    if len(folds) != 5 or len(transitions) != 8:
        raise ValueError(
            f"Expected 5 original folds and 8 adjacent threshold transitions; "
            f"found {len(folds)} folds and {len(transitions)} transitions."
        )

    feature_sets = build_feature_sets(data.columns.tolist())
    for name, cols in feature_sets.items():
        if not cols:
            raise ValueError(f"Feature set {name} is empty.")

    # Crossed blocked validation: each test block is one original BFW fold x
    # one threshold transition. Training excludes the entire held-out fold AND
    # the entire held-out transition. Thus neither the target identity fold nor
    # that threshold interval appears in training.
    blocks = []
    for fold in folds:
        for transition in transitions:
            test_mask = (data["fold"] == fold) & (data["transition_id"] == transition)
            train_mask = (data["fold"] != fold) & (data["transition_id"] != transition)
            train_idx = np.flatnonzero(train_mask.to_numpy())
            test_idx = np.flatnonzero(test_mask.to_numpy())
            if len(test_idx) == 0 or len(train_idx) == 0:
                raise RuntimeError(f"Empty train/test block for fold={fold}, transition={transition}")
            # Explicit leakage assertions.
            assert fold not in set(data.iloc[train_idx]["fold"].unique())
            assert transition not in set(data.iloc[train_idx]["transition_id"].unique())
            blocks.append((fold, transition, train_idx, test_idx))

    reg_pred_rows, reg_metric_rows = [], []
    reg_specs = {
        "dummy_mean": (lambda: DummyRegressor(strategy="mean"), False),
        "ridge": (lambda: Ridge(alpha=10.0), True),
        "extra_trees": (lambda: ExtraTreesRegressor(
            n_estimators=25, min_samples_leaf=20, max_features=0.35,
            max_depth=8, n_jobs=4, random_state=RANDOM_SEED
        ), False),
    }

    for target in REGRESSION_TARGETS:
        valid = data[target].notna().to_numpy()
        for set_name, features in feature_sets.items():
            for model_name, (factory, scale) in reg_specs.items():
                oof = np.full(len(data), np.nan, dtype=float)
                for fold, transition, train_idx_all, test_idx_all in blocks:
                    train_idx = train_idx_all[valid[train_idx_all]]
                    test_idx = test_idx_all[valid[test_idx_all]]
                    if len(test_idx) == 0:
                        continue
                    y_train = data.iloc[train_idx][target].to_numpy(float)
                    cat = [c for c in CATEGORICAL if c in features]
                    num = [c for c in features if c not in cat]
                    pipe = make_pipeline(factory(), num, cat, scale)
                    pipe.fit(data.iloc[train_idx][features], y_train)
                    oof[test_idx] = pipe.predict(data.iloc[test_idx][features])
                if not np.isfinite(oof[valid]).all():
                    raise RuntimeError(f"Missing OOF predictions: {target}/{set_name}/{model_name}")

                y_all = data.loc[valid, target].to_numpy(float)
                p_all = oof[valid]
                reg_metric_rows.append({
                    "target": target, "feature_set": set_name, "predictor": model_name,
                    "evaluation": "pooled_crossed_block_oof", **reg_scores(y_all, p_all),
                })
                for fold, transition, _, test_idx_all in blocks:
                    idx = test_idx_all[valid[test_idx_all]]
                    if len(idx):
                        reg_metric_rows.append({
                            "target": target, "feature_set": set_name, "predictor": model_name,
                            "evaluation": "held_out_fold_transition_block",
                            "fold": fold, "transition_id": transition,
                            **reg_scores(data.iloc[idx][target].to_numpy(float), oof[idx]),
                        })
                for i in np.flatnonzero(valid):
                    row = data.iloc[i]
                    reg_pred_rows.append({
                        "target": target, "feature_set": set_name, "predictor": model_name,
                        "fold": int(row["fold"]), "split_id": int(row["split_id"]),
                        "transition_id": row["transition_id"], "model": row["model"],
                        "metric": row["metric"], "threshold_from": row["threshold_from"],
                        "threshold_to": row["threshold_to"], "observed": float(row[target]),
                        "predicted": float(oof[i]),
                    })
                print(f"Regression: {target} | {set_name} | {model_name}", flush=True)

    cls_pred_rows, cls_metric_rows = [], []
    y_all = data[CLASSIFICATION_TARGET].to_numpy(int)
    cls_specs = {
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
        for model_name, (factory, scale) in cls_specs.items():
            prob = np.full(len(data), np.nan, dtype=float)
            for fold, transition, train_idx, test_idx in blocks:
                y_train = y_all[train_idx]
                cat = [c for c in CATEGORICAL if c in features]
                num = [c for c in features if c not in cat]
                pipe = make_pipeline(factory(), num, cat, scale)
                pipe.fit(data.iloc[train_idx][features], y_train)
                prob[test_idx] = pipe.predict_proba(data.iloc[test_idx][features])[:, 1]
            if not np.isfinite(prob).all():
                raise RuntimeError(f"Missing classification predictions: {set_name}/{model_name}")

            cls_metric_rows.append({
                "target": CLASSIFICATION_TARGET, "feature_set": set_name,
                "predictor": model_name, "evaluation": "pooled_crossed_block_oof",
                **cls_scores(y_all, prob),
            })
            for fold, transition, _, test_idx in blocks:
                cls_metric_rows.append({
                    "target": CLASSIFICATION_TARGET, "feature_set": set_name,
                    "predictor": model_name, "evaluation": "held_out_fold_transition_block",
                    "fold": fold, "transition_id": transition,
                    **cls_scores(y_all[test_idx], prob[test_idx]),
                })
            for i, row in data.iterrows():
                cls_pred_rows.append({
                    "target": CLASSIFICATION_TARGET, "feature_set": set_name,
                    "predictor": model_name, "fold": int(row["fold"]),
                    "split_id": int(row["split_id"]), "transition_id": row["transition_id"],
                    "model": row["model"], "metric": row["metric"],
                    "threshold_from": row["threshold_from"], "threshold_to": row["threshold_to"],
                    "observed": int(y_all[i]), "probability": float(prob[i]),
                })
            print(f"Classification: {set_name} | {model_name}", flush=True)

    reg_pred = pd.DataFrame(reg_pred_rows)
    reg_metrics = pd.DataFrame(reg_metric_rows)
    cls_pred = pd.DataFrame(cls_pred_rows)
    cls_metrics = pd.DataFrame(cls_metric_rows)

    # Directly quantify incremental pilot-feature value against context-only
    # using identical held-out observations and fixed predictors.
    comparison_rows = []
    for (target, predictor, evaluation), group in reg_metrics.groupby(["target", "predictor", "evaluation"], dropna=False):
        if evaluation == "pooled_crossed_block_oof":
            base = group[group.feature_set == "context_only"]
            for _, row in group.iterrows():
                b = base.iloc[0]
                comparison_rows.append({
                    "task": target, "predictor": predictor, "evaluation": evaluation,
                    "fold": np.nan, "transition_id": "ALL", "feature_set": row.feature_set,
                    "delta_mae_vs_context": row.mae - b.mae,
                    "delta_rmse_vs_context": row.rmse - b.rmse,
                    "delta_r2_vs_context": row.r2 - b.r2,
                })
        else:
            for _, row in group.iterrows():
                b = group[group.feature_set == "context_only"].iloc[0]
                comparison_rows.append({
                    "task": target, "predictor": predictor, "evaluation": evaluation,
                    "fold": row.get("fold", np.nan), "transition_id": row.get("transition_id", ""),
                    "feature_set": row.feature_set,
                    "delta_mae_vs_context": row.mae - b.mae,
                    "delta_rmse_vs_context": row.rmse - b.rmse,
                    "delta_r2_vs_context": row.r2 - b.r2,
                })
    for (predictor, evaluation), group in cls_metrics.groupby(["predictor", "evaluation"], dropna=False):
        if evaluation == "pooled_crossed_block_oof":
            base = group[group.feature_set == "context_only"].iloc[0]
        for _, row in group.iterrows():
            b = base if evaluation == "pooled_crossed_block_oof" else group[group.feature_set == "context_only"].iloc[0]
            comparison_rows.append({
                "task": CLASSIFICATION_TARGET, "predictor": predictor,
                "evaluation": evaluation, "fold": row.get("fold", np.nan),
                "transition_id": row.get("transition_id", "ALL"),
                "feature_set": row.feature_set,
                "delta_brier_vs_context": row.brier - b.brier,
                "delta_roc_auc_vs_context": row.roc_auc - b.roc_auc if pd.notna(row.roc_auc) and pd.notna(b.roc_auc) else np.nan,
                "delta_average_precision_vs_context": row.average_precision - b.average_precision,
            })
    comparison = pd.DataFrame(comparison_rows)

    outputs = {
        "regression_oof_predictions": OUTPUT_DIR / f"{PREFIX}_regression_oof_predictions.csv",
        "regression_metrics": OUTPUT_DIR / f"{PREFIX}_regression_metrics.csv",
        "classification_oof_predictions": OUTPUT_DIR / f"{PREFIX}_classification_oof_predictions.csv",
        "classification_metrics": OUTPUT_DIR / f"{PREFIX}_classification_metrics.csv",
        "feature_lift_comparison": OUTPUT_DIR / f"{PREFIX}_feature_lift_comparison.csv",
        "manifest": OUTPUT_DIR / f"{PREFIX}_manifest.json",
    }
    reg_pred.to_csv(outputs["regression_oof_predictions"], index=False)
    reg_metrics.to_csv(outputs["regression_metrics"], index=False)
    cls_pred.to_csv(outputs["classification_oof_predictions"], index=False)
    cls_metrics.to_csv(outputs["classification_metrics"], index=False)
    comparison.to_csv(outputs["feature_lift_comparison"], index=False)

    manifest = {
        "experiment": "Program 15 - Cross-Context Predictive Generalization",
        "input": str(INPUT_PATH), "input_rows": int(len(data)),
        "outer_validation": "crossed blocked validation; each test block is one original BFW fold x one adjacent threshold transition",
        "training_exclusion": "training excludes all rows from the held-out fold AND all rows from the held-out threshold transition",
        "folds": [int(x) for x in folds], "threshold_transitions": transitions,
        "outer_blocks": int(len(blocks)),
        "test_block_size": "typically 120 rows (20 identity splits x 3 models x 2 metrics)",
        "feature_sets": {k: v for k, v in feature_sets.items()},
        "regression_targets": REGRESSION_TARGETS,
        "classification_target": CLASSIFICATION_TARGET,
        "predictors": {
            "regression": list(reg_specs.keys()),
            "classification": list(cls_specs.keys()),
        },
        "random_seed": RANDOM_SEED,
        "hyperparameter_tuning": "none; settings fixed to Program 14 for comparability",
        "leakage_controls": [
            "fold and split identifiers are never predictors",
            "no row from held-out fold appears in training",
            "no row from held-out threshold transition appears in training",
            "all rows from a fold-transition test block are predicted by a model trained without either axis",
            "pilot and target identities remain disjoint as established by Program 11",
        ],
        "interpretation_notes": [
            "This is a stricter joint generalization test than Program 14's leave-one-fold-out validation.",
            "The test asks whether pilot information predicts outcomes for a held-out identity fold at a threshold interval absent from training.",
            "Only five BFW folds and eight threshold transitions are available; block-level metrics are descriptive and do not support narrow inferential confidence intervals.",
            "The 40 crossed test blocks are not assumed independent; repeated split structure and common source dataset induce dependence.",
            "A positive result supports within-BFW cross-context feasibility, not external dataset or unseen-architecture generalization.",
            "Decision-flip rate is omitted as a primary target because it is directly approximated by threshold-band score fractions.",
        ],
        "outputs": {k: str(v.name) for k, v in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nProgram 15 completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"Outer blocks: {len(blocks)} | input rows: {len(data)}")
    print("Pooled regression metrics:")
    print(reg_metrics[reg_metrics.evaluation == "pooled_crossed_block_oof"].to_string(index=False))
    print("Pooled classification metrics:")
    print(cls_metrics[cls_metrics.evaluation == "pooled_crossed_block_oof"].to_string(index=False))


if __name__ == "__main__":
    main()
