from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import OUTPUT_DIR

COUNT_FILE = Path(OUTPUT_DIR) / "false_positive_count_repetition_summary.csv"
TRANSITION_FILE = Path(OUTPUT_DIR) / "tie_aware_transition_diagnostics.csv"

BASE_FEATURES = ["sample_size", "threshold_from"]
DIAGNOSTIC_FEATURES = [
    "total_false_positives",
    "total_negatives",
    "all_groups_zero_fp",
    "groups_with_zero_fp",
    "groups_tied_at_max_fpr",
    "unique_worst_group",
    "max_fpr",
    "second_highest_fpr_gap",
]
TARGET_BINARY = "worst_set_changed"
TARGET_CONTINUOUS = "tie_aware_ranking_instability"


def build_dataset():
    counts = pd.read_csv(COUNT_FILE)
    transitions = pd.read_csv(TRANSITION_FILE)

    transitions = transitions[
        (transitions["metric"] == "FPR")
        & (transitions["threshold_from"].isin([0.60, 0.65]))
        & (transitions["threshold_to"].isin([0.65, 0.70]))
    ].copy()

    data = transitions.merge(
        counts,
        left_on=["sample_size", "repetition", "threshold_from"],
        right_on=["sample_size", "repetition", "threshold"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_count"),
    )
    required = BASE_FEATURES + DIAGNOSTIC_FEATURES + [TARGET_BINARY]
    data = data.dropna(subset=required).copy()
    data[TARGET_BINARY] = data[TARGET_BINARY].astype(int)
    data["cv_group"] = (
        data["sample_size"].astype(str) + "_" + data["repetition"].astype(str)
    )
    return data


def make_preprocessor(feature_names):
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), feature_names)
        ],
        remainder="drop",
    )


def main():
    if not COUNT_FILE.exists() or not TRANSITION_FILE.exists():
        raise FileNotFoundError(
            "Experiment 13 needs Experiment 12's false_positive_count_repetition_summary.csv "
            "and Experiment 11's tie_aware_transition_diagnostics.csv in OUTPUT_DIR."
        )

    out_dir = Path(OUTPUT_DIR)
    data = build_dataset()
    if len(data) < 100:
        raise ValueError(f"Too few matched rows for CV: {len(data)}")

    baseline_features = BASE_FEATURES
    full_features = BASE_FEATURES + DIAGNOSTIC_FEATURES
    groups = data["cv_group"]
    cv = GroupKFold(n_splits=5)

    results = []
    prediction_frames = []

    # Binary outcome: did the set of worst-FPR groups change?
    y_binary = data[TARGET_BINARY].to_numpy()
    for model_name, features in [
        ("baseline_size_threshold", baseline_features),
        ("diagnostics_augmented", full_features),
    ]:
        model = Pipeline([
            ("preprocess", make_preprocessor(features)),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ])
        prob = cross_val_predict(
            model, data[features], y_binary, groups=groups, cv=cv,
            method="predict_proba"
        )[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(y_binary, prob) if len(np.unique(y_binary)) > 1 else np.nan
        results.append({
            "target": TARGET_BINARY,
            "model": model_name,
            "n_rows": len(data),
            "positive_rate": float(y_binary.mean()),
            "roc_auc": float(auc),
            "brier_score": float(brier_score_loss(y_binary, prob)),
            "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_binary, pred)),
        })
        prediction_frames.append(pd.DataFrame({
            "sample_size": data["sample_size"],
            "repetition": data["repetition"],
            "threshold_from": data["threshold_from"],
            "threshold_to": data["threshold_to"],
            "target": TARGET_BINARY,
            "model": model_name,
            "observed": y_binary,
            "predicted": prob,
        }))

    # Continuous outcome: tie-aware ranking instability (1 - Kendall tau-b).
    regression_data = data.dropna(subset=[TARGET_CONTINUOUS]).copy()
    y_cont = regression_data[TARGET_CONTINUOUS].to_numpy(dtype=float)
    regression_groups = regression_data["cv_group"]
    regression_cv = GroupKFold(n_splits=5)
    for model_name, features, estimator in [
        ("baseline_size_threshold", baseline_features, DummyRegressor(strategy="mean")),
        ("diagnostics_augmented", full_features, Ridge(alpha=10.0)),
    ]:
        model = Pipeline([
            ("preprocess", make_preprocessor(features)),
            ("model", estimator),
        ])
        pred = cross_val_predict(
            model, regression_data[features], y_cont, groups=regression_groups, cv=regression_cv
        )
        results.append({
            "target": TARGET_CONTINUOUS,
            "model": model_name,
            "n_rows": len(regression_data),
            "positive_rate": np.nan,
            "roc_auc": np.nan,
            "brier_score": np.nan,
            "balanced_accuracy_at_0_5": np.nan,
            "mae": float(mean_absolute_error(y_cont, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_cont, pred))),
            "r2": float(r2_score(y_cont, pred)),
        })
        prediction_frames.append(pd.DataFrame({
            "sample_size": regression_data["sample_size"],
            "repetition": regression_data["repetition"],
            "threshold_from": regression_data["threshold_from"],
            "threshold_to": regression_data["threshold_to"],
            "target": TARGET_CONTINUOUS,
            "model": model_name,
            "observed": y_cont,
            "predicted": pred,
        }))

    comparison = pd.DataFrame(results)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    dataset_path = out_dir / "experiment13_prediction_dataset.csv"
    comparison_path = out_dir / "experiment13_model_comparison.csv"
    predictions_path = out_dir / "experiment13_cv_predictions.csv"

    data.to_csv(dataset_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(f"Matched rows: {len(data)}")
    print(f"Unique sample-repetition groups: {data['cv_group'].nunique()}")
    print("\nOut-of-fold model comparison:")
    print(comparison.to_string(index=False))
    print("\nSaved:")
    print(dataset_path)
    print(comparison_path)
    print(predictions_path)
    print("\nInterpretation caution: compare augmented performance with baseline; "
          "a high score alone is not proof of generalization beyond this dataset/design.")


if __name__ == "__main__":
    main()