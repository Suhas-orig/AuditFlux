from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# AuditFlux Program 17
# Factorial decomposition of architecture and identity-fold generalization
# Reads the already validated Program 16 dataset. It does not recompute BFW
# pair scores, pilot splits, thresholds, or target outcomes.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import OUTPUT_DIR  # noqa: E402

INPUT_PATH = OUTPUT_DIR / "16_architecture_transfer_dataset.csv"
PREFIX = "17_generalization_decomposition"
MODELS = ["vgg16", "resnet50", "senet50"]
FOLDS = [1, 2, 3, 4, 5]
TARGET_REGRESSION = [
    "target_magnitude_change",
    "target_normalized_ranking_instability",
]
TARGET_CLASSIFICATION = "target_worst_group_set_changed"
RANDOM_SEED = 20261108

CONTEXT_COLS = [
    "metric",
    "fpr_quantile_from",
    "fpr_quantile_to",
    "fpr_quantile_midpoint",
    "fpr_quantile_width",
]


def make_pipeline(estimator, numeric_cols, categorical_cols, scale):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def regression_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 and np.unique(y).size > 1 else np.nan,
        "spearman": float(spearmanr(y, pred).statistic)
        if len(y) > 2 and np.unique(y).size > 1 and np.unique(pred).size > 1
        else np.nan,
    }


def classification_metrics(y, prob):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    hard = (prob >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "brier": float(brier_score_loss(y, prob)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, hard)),
        "roc_auc": float(roc_auc_score(y, prob)) if np.unique(y).size == 2 else np.nan,
        "average_precision": float(average_precision_score(y, prob)),
    }


def predictor_specs():
    # Settings intentionally match Program 16; no tuning is conducted here.
    return [
        ("dummy_mean", lambda: DummyRegressor(strategy="mean"), False),
        ("ridge", lambda: Ridge(alpha=10.0), True),
        (
            "extra_trees",
            lambda: ExtraTreesRegressor(
                n_estimators=15,
                min_samples_leaf=25,
                max_features=0.25,
                max_depth=6,
                n_jobs=1,
                random_state=RANDOM_SEED,
            ),
            False,
        ),
    ]


def classifier_specs():
    return [
        ("dummy_prior", lambda: DummyClassifier(strategy="prior"), False),
        (
            "logistic_balanced",
            lambda: LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_SEED,
            ),
            True,
        ),
        (
            "extra_trees_balanced",
            lambda: ExtraTreesClassifier(
                n_estimators=15,
                min_samples_leaf=25,
                max_features=0.25,
                max_depth=6,
                class_weight="balanced",
                n_jobs=1,
                random_state=RANDOM_SEED,
            ),
            False,
        ),
    ]


def build_blocks(data):
    blocks = []
    # A. Architecture-only: architecture absent from training; folds represented.
    for model in MODELS:
        blocks.append({
            "condition": "architecture_only",
            "block_id": f"architecture_{model}",
            "held_out_model": model,
            "held_out_fold": "ALL",
            "test_mask": data["model"].eq(model),
            "train_mask": data["model"].ne(model),
            "interpretation": "unseen architecture, familiar BFW fold distribution",
        })
    # B. Fold-only: identity fold absent from training; architectures represented.
    for fold in FOLDS:
        blocks.append({
            "condition": "fold_only",
            "block_id": f"fold_{fold}",
            "held_out_model": "ALL",
            "held_out_fold": fold,
            "test_mask": data["fold"].eq(fold),
            "train_mask": data["fold"].ne(fold),
            "interpretation": "unseen BFW identity fold, familiar architectures",
        })
    # C. Joint: both architecture and identity fold absent from training.
    for model in MODELS:
        for fold in FOLDS:
            blocks.append({
                "condition": "joint_architecture_and_fold",
                "block_id": f"{model}_fold_{fold}",
                "held_out_model": model,
                "held_out_fold": fold,
                "test_mask": data["model"].eq(model) & data["fold"].eq(fold),
                "train_mask": data["model"].ne(model) & data["fold"].ne(fold),
                "interpretation": "both architecture and BFW identity fold unseen",
            })
    return blocks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Required Program 16 dataset not found: {INPUT_PATH}\n"
            "Run Program 16 first; Program 17 reuses its calibrated pilot/target data."
        )

    print(f"Loading Program 16 dataset: {INPUT_PATH}", flush=True)
    data = pd.read_csv(INPUT_PATH)
    required = {
        "fold", "split_id", "model", "metric", "fpr_quantile_from",
        "fpr_quantile_to", *TARGET_REGRESSION, TARGET_CLASSIFICATION,
        "threshold_from_raw", "threshold_to_raw",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Program 16 dataset is missing required columns: {missing}")

    data["fold"] = data["fold"].astype(int)
    if set(data["model"].unique()) != set(MODELS):
        raise ValueError(f"Unexpected architecture set: {sorted(data['model'].unique())}")
    if set(data["fold"].unique()) != set(FOLDS):
        raise ValueError(f"Unexpected fold set: {sorted(data['fold'].unique())}")

    key_cols = ["fold", "split_id", "model", "metric", "fpr_quantile_from", "fpr_quantile_to"]
    if data.duplicated(key_cols).any():
        raise ValueError("Duplicate Program 16 predictive-example keys detected.")
    expected_rows = len(FOLDS) * 20 * len(MODELS) * 2 * 3
    if len(data) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows from Program 16, found {len(data)}.")

    # Program 16's full feature set, excluding architecture ID, raw thresholds,
    # and all target-derived fields. Absolute pooled score mean/SD are omitted
    # exactly as in Program 16 to reduce architecture-scale dependence.
    target_and_metadata = {
        "fold", "split_id", "model", "threshold_from_raw", "threshold_to_raw",
        "target_gap_from", "target_gap_to", *TARGET_REGRESSION,
        TARGET_CLASSIFICATION, "target_worst_group_from", "target_worst_group_to",
        "pilot_neg_score_mean", "pilot_neg_score_sd",
    }
    full_cols = [c for c in data.columns if c not in target_and_metadata]
    feature_sets = {
        "context_only": CONTEXT_COLS,
        "pilot_calibrated_features": full_cols,
    }
    categorical_cols = ["metric"]

    blocks = build_blocks(data)
    reg_predictions, cls_predictions = [], []
    reg_block_metrics, cls_block_metrics, block_audit = [], [], []

    # Each condition is a separate complete partition of the 1,800 examples.
    # Every example must be tested exactly once within each condition.
    for block in blocks:
        train = data.loc[block["train_mask"]].reset_index(drop=True)
        test = data.loc[block["test_mask"]].reset_index(drop=True)
        if train.empty or test.empty:
            raise RuntimeError(f"Empty partition for {block['block_id']}")

        condition = block["condition"]
        held_model = block["held_out_model"]
        held_fold = block["held_out_fold"]

        if condition in {"architecture_only", "joint_architecture_and_fold"}:
            if held_model in set(train["model"]):
                raise RuntimeError(f"Architecture leakage in {block['block_id']}")
        if condition in {"fold_only", "joint_architecture_and_fold"}:
            if held_fold != "ALL" and int(held_fold) in set(train["fold"]):
                raise RuntimeError(f"Fold leakage in {block['block_id']}")

        block_audit.append({
            "condition": condition,
            "block_id": block["block_id"],
            "held_out_model": held_model,
            "held_out_fold": held_fold,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_architectures": "|".join(sorted(train["model"].unique())),
            "test_architectures": "|".join(sorted(test["model"].unique())),
            "train_folds": "|".join(str(x) for x in sorted(train["fold"].unique())),
            "test_folds": "|".join(str(x) for x in sorted(test["fold"].unique())),
            "same_fold_identity_overlap_allowed": condition == "architecture_only",
            "same_architecture_overlap_allowed": condition == "fold_only",
        })

        print(
            f"{condition}: {block['block_id']} | train={len(train)} test={len(test)}",
            flush=True,
        )

        for target_name in TARGET_REGRESSION:
            train_target = train.loc[train[target_name].notna()].reset_index(drop=True)
            test_target = test.loc[test[target_name].notna()].reset_index(drop=True)
            y_test = test_target[target_name].to_numpy(dtype=float)

            for feature_name, cols_used in feature_sets.items():
                numeric_cols = [c for c in cols_used if c not in categorical_cols]
                for predictor, factory, scale in predictor_specs():
                    pipe = make_pipeline(factory(), numeric_cols, categorical_cols, scale)
                    pipe.fit(train_target[cols_used], train_target[target_name].to_numpy(dtype=float))
                    pred = pipe.predict(test_target[cols_used])
                    for rec, obs, pr in zip(test_target.to_dict("records"), y_test, pred):
                        reg_predictions.append({
                            "condition": condition,
                            "block_id": block["block_id"],
                            "held_out_model": held_model,
                            "held_out_fold": held_fold,
                            "target": target_name,
                            "feature_set": feature_name,
                            "predictor": predictor,
                            "fold": int(rec["fold"]),
                            "split_id": int(rec["split_id"]),
                            "model": rec["model"],
                            "metric": rec["metric"],
                            "fpr_quantile_from": float(rec["fpr_quantile_from"]),
                            "fpr_quantile_to": float(rec["fpr_quantile_to"]),
                            "observed": float(obs),
                            "predicted": float(pr),
                        })
                    reg_block_metrics.append({
                        "condition": condition,
                        "block_id": block["block_id"],
                        "held_out_model": held_model,
                        "held_out_fold": held_fold,
                        "target": target_name,
                        "feature_set": feature_name,
                        "predictor": predictor,
                        **regression_metrics(y_test, pred),
                    })

        for feature_name, cols_used in feature_sets.items():
            numeric_cols = [c for c in cols_used if c not in categorical_cols]
            y_test = test[TARGET_CLASSIFICATION].to_numpy(dtype=int)
            for predictor, factory, scale in classifier_specs():
                pipe = make_pipeline(factory(), numeric_cols, categorical_cols, scale)
                pipe.fit(train[cols_used], train[TARGET_CLASSIFICATION].to_numpy(dtype=int))
                prob = pipe.predict_proba(test[cols_used])[:, 1]
                for rec, obs, pr in zip(test.to_dict("records"), y_test, prob):
                    cls_predictions.append({
                        "condition": condition,
                        "block_id": block["block_id"],
                        "held_out_model": held_model,
                        "held_out_fold": held_fold,
                        "feature_set": feature_name,
                        "predictor": predictor,
                        "fold": int(rec["fold"]),
                        "split_id": int(rec["split_id"]),
                        "model": rec["model"],
                        "metric": rec["metric"],
                        "fpr_quantile_from": float(rec["fpr_quantile_from"]),
                        "fpr_quantile_to": float(rec["fpr_quantile_to"]),
                        "observed": int(obs),
                        "predicted_probability": float(pr),
                    })
                cls_block_metrics.append({
                    "condition": condition,
                    "block_id": block["block_id"],
                    "held_out_model": held_model,
                    "held_out_fold": held_fold,
                    "feature_set": feature_name,
                    "predictor": predictor,
                    **classification_metrics(y_test, prob),
                })

    reg_df = pd.DataFrame(reg_predictions)
    cls_df = pd.DataFrame(cls_predictions)
    reg_block_df = pd.DataFrame(reg_block_metrics)
    cls_block_df = pd.DataFrame(cls_block_metrics)
    audit_df = pd.DataFrame(block_audit)

    # Check that every original predictive example is tested exactly once per
    # validation condition. This makes pooled comparisons use equal test rows.
    expected_keys = set(map(tuple, data[key_cols].itertuples(index=False, name=None)))
    for condition in ["architecture_only", "fold_only", "joint_architecture_and_fold"]:
        part = reg_df[(reg_df.condition == condition) &
                      (reg_df.target == TARGET_REGRESSION[0]) &
                      (reg_df.feature_set == "context_only") &
                      (reg_df.predictor == "dummy_mean")]
        observed_keys = list(map(tuple, part[key_cols].itertuples(index=False, name=None)))
        if len(observed_keys) != len(expected_keys) or set(observed_keys) != expected_keys:
            raise RuntimeError(f"OOF partition for {condition} does not test each row exactly once.")

    pooled_reg = []
    for keys, part in reg_df.groupby(["condition", "target", "feature_set", "predictor"]):
        pooled_reg.append({
            "condition": keys[0], "block_id": "POOLED_OOF", "held_out_model": "ALL",
            "held_out_fold": "ALL", "target": keys[1], "feature_set": keys[2],
            "predictor": keys[3], **regression_metrics(part.observed, part.predicted),
        })
    pooled_cls = []
    for keys, part in cls_df.groupby(["condition", "feature_set", "predictor"]):
        pooled_cls.append({
            "condition": keys[0], "block_id": "POOLED_OOF", "held_out_model": "ALL",
            "held_out_fold": "ALL", "feature_set": keys[1], "predictor": keys[2],
            **classification_metrics(part.observed, part.predicted_probability),
        })
    pooled_reg_df = pd.DataFrame(pooled_reg)
    pooled_cls_df = pd.DataFrame(pooled_cls)

    # Paired block-level feature lift: positive MAE improvement means pilot
    # features beat context; positive ROC-AUC improvement means better ranking.
    lift_rows = []
    for condition in ["architecture_only", "fold_only", "joint_architecture_and_fold"]:
        for target_name in TARGET_REGRESSION:
            for predictor in ["ridge", "extra_trees"]:
                sub = reg_block_df[
                    (reg_block_df.condition == condition) &
                    (reg_block_df.target == target_name) &
                    (reg_block_df.predictor == predictor)
                ]
                wide = sub.pivot(index="block_id", columns="feature_set", values="mae")
                for block_id, row in wide.iterrows():
                    lift_rows.append({
                        "condition": condition, "task": target_name,
                        "predictor": predictor, "block_id": block_id,
                        "context_metric": "MAE", "context_value": row.get("context_only", np.nan),
                        "pilot_value": row.get("pilot_calibrated_features", np.nan),
                        "improvement": row.get("context_only", np.nan) - row.get("pilot_calibrated_features", np.nan),
                        "improvement_direction": "positive_is_better",
                    })
        for predictor in ["logistic_balanced", "extra_trees_balanced"]:
            sub = cls_block_df[
                (cls_block_df.condition == condition) &
                (cls_block_df.predictor == predictor)
            ]
            wide = sub.pivot(index="block_id", columns="feature_set", values="roc_auc")
            for block_id, row in wide.iterrows():
                lift_rows.append({
                    "condition": condition, "task": TARGET_CLASSIFICATION,
                    "predictor": predictor, "block_id": block_id,
                    "context_metric": "ROC_AUC", "context_value": row.get("context_only", np.nan),
                    "pilot_value": row.get("pilot_calibrated_features", np.nan),
                    "improvement": row.get("pilot_calibrated_features", np.nan) - row.get("context_only", np.nan),
                    "improvement_direction": "positive_is_better",
                })
    lift_df = pd.DataFrame(lift_rows)

    # Save outputs.
    outputs = {
        "regression_oof_predictions": f"{PREFIX}_regression_oof_predictions.csv",
        "classification_oof_predictions": f"{PREFIX}_classification_oof_predictions.csv",
        "regression_block_metrics": f"{PREFIX}_regression_block_metrics.csv",
        "classification_block_metrics": f"{PREFIX}_classification_block_metrics.csv",
        "regression_pooled_metrics": f"{PREFIX}_regression_pooled_metrics.csv",
        "classification_pooled_metrics": f"{PREFIX}_classification_pooled_metrics.csv",
        "feature_lift": f"{PREFIX}_feature_lift.csv",
        "block_audit": f"{PREFIX}_block_audit.csv",
    }
    reg_df.to_csv(OUTPUT_DIR / outputs["regression_oof_predictions"], index=False)
    cls_df.to_csv(OUTPUT_DIR / outputs["classification_oof_predictions"], index=False)
    reg_block_df.to_csv(OUTPUT_DIR / outputs["regression_block_metrics"], index=False)
    cls_block_df.to_csv(OUTPUT_DIR / outputs["classification_block_metrics"], index=False)
    pooled_reg_df.to_csv(OUTPUT_DIR / outputs["regression_pooled_metrics"], index=False)
    pooled_cls_df.to_csv(OUTPUT_DIR / outputs["classification_pooled_metrics"], index=False)
    lift_df.to_csv(OUTPUT_DIR / outputs["feature_lift"], index=False)
    audit_df.to_csv(OUTPUT_DIR / outputs["block_audit"], index=False)

    manifest = {
        "experiment": "Program 17 - Factorial Decomposition of Architecture and Identity-Fold Generalization",
        "input": str(INPUT_PATH),
        "input_rows": int(len(data)),
        "input_source": "Program 16 calibrated pilot/target dataset; no BFW score recalculation",
        "architectures": MODELS,
        "identity_folds": FOLDS,
        "validation_conditions": {
            "architecture_only": {
                "blocks": len(MODELS),
                "training_excludes": "held-out architecture",
                "test_contains": "all folds for held-out architecture",
                "interpretation": "architecture transfer conditional on familiar BFW identity-fold distribution",
                "qualification": "same BFW fold/split compositions can appear in source architectures; this is not identity-disjoint architecture transfer",
            },
            "fold_only": {
                "blocks": len(FOLDS),
                "training_excludes": "held-out BFW identity fold",
                "test_contains": "all architectures for held-out fold",
                "interpretation": "identity-fold generalization with familiar architectures",
            },
            "joint_architecture_and_fold": {
                "blocks": len(MODELS) * len(FOLDS),
                "training_excludes": "held-out architecture AND held-out BFW identity fold",
                "test_contains": "held-out architecture/fold intersection",
                "interpretation": "joint architecture and identity-fold generalization; reproduces Program 16 split logic",
            },
        },
        "feature_sets": {
            "context_only": CONTEXT_COLS,
            "pilot_calibrated_features": "same Program 16 pilot features; architecture ID and raw score thresholds excluded; target-derived columns excluded",
        },
        "regression_targets": TARGET_REGRESSION,
        "classification_target": TARGET_CLASSIFICATION,
        "predictor_settings": "fixed to Program 16; no hyperparameter tuning",
        "pooled_comparison": "each validation condition tests each of the same 1,800 predictive examples exactly once; block sizes differ by condition",
        "dependence_and_limitations": [
            "The 3 architecture domains and 5 BFW folds are few; results are descriptive and exploratory.",
            "The 40 repeated identity-panel splits are dependent; no narrow inferential confidence intervals are reported.",
            "Architecture-only condition intentionally permits familiar folds and split compositions in other architectures; it measures conditional architecture transfer, not unseen-identity transfer.",
            "Fold-only condition retains all architectures in training; it measures fold generalization conditional on known architectures.",
            "Joint condition excludes both factors and is the strictest test.",
            "All conditions reuse the same pilot-calibrated thresholds and target outcomes from Program 16.",
            "The underlying face-recognition scores are fixed; this tests transfer of the instability predictor, not transfer of face-recognition models.",
            "The BFW impostor graph is incomplete and comparisons share identity endpoints; targets remain conditional on the observed comparison design.",
        ],
        "random_seed": RANDOM_SEED,
        "outputs": outputs,
    }
    with (OUTPUT_DIR / f"{PREFIX}_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nPROGRAM 17 COMPLETE", flush=True)
    print("Pooled regression metrics:")
    print(pooled_reg_df.to_string(index=False))
    print("Pooled classification metrics:")
    print(pooled_cls_df.to_string(index=False))
    print("Feature-lift block counts:")
    print(lift_df.groupby(["condition", "task", "predictor", "improvement_direction"])["improvement"].agg(["count", "mean", lambda x: int((x > 0).sum())]).to_string())
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
