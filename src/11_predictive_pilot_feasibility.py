from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
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
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import BFW_DATA_PATH, OUTPUT_DIR


# ---------------------------------------------------------------------
# Experimental design
# ---------------------------------------------------------------------
MODELS = ["vgg16", "resnet50", "senet50"]
GROUP_COLUMN = "a1"
SECOND_GROUP_COLUMN = "a2"
LABEL_COLUMN = "label"
ID1_COLUMN = "id1"
ID2_COLUMN = "id2"
FOLD_COLUMN = "fold"

GROUPS = ["AF", "AM", "BF", "BM", "IF", "IM", "WF", "WM"]
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
METRICS = ["FNR", "FPR"]

# Each fold has 20 identities per demographic group. Split 10/10,
# producing identity-disjoint pilot and target panels with 80 identities each.
IDENTITIES_PER_GROUP_PER_PANEL = 10
SPLITS_PER_FOLD = 20
RANDOM_SEED = 20261102

PREFIX = "11_predictive_pilot"
TARGETS_REGRESSION = [
    "target_magnitude_change",
    "target_normalized_ranking_instability",
    "target_decision_flip_rate",
]
TARGET_CLASSIFICATION = "target_worst_group_set_changed"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def calculate_group_metrics(labels, scores, group_codes, group_count, threshold):
    predictions = (scores >= threshold).astype(np.int8)
    outcomes = labels * 2 + predictions  # 0=TN, 1=FP, 2=FN, 3=TP
    combined = group_codes * 4 + outcomes
    counts = np.bincount(
        combined, minlength=group_count * 4
    ).reshape(group_count, 4)

    tn, fp, fn, tp = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
    positives = tp + fn
    negatives = tn + fp

    fnr = np.divide(
        fn, positives,
        out=np.full(group_count, np.nan, dtype=float),
        where=positives > 0,
    )
    fpr = np.divide(
        fp, negatives,
        out=np.full(group_count, np.nan, dtype=float),
        where=negatives > 0,
    )
    return {"FNR": fnr, "FPR": fpr, "counts": counts}


def summarize_group_vector(values, prefix, output):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        output[f"{prefix}_mean"] = np.nan
        output[f"{prefix}_sd"] = np.nan
        output[f"{prefix}_min"] = np.nan
        output[f"{prefix}_max"] = np.nan
        return
    output[f"{prefix}_mean"] = float(np.mean(finite))
    output[f"{prefix}_sd"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    output[f"{prefix}_min"] = float(np.min(finite))
    output[f"{prefix}_max"] = float(np.max(finite))


def pilot_features(
    labels, scores, group_codes, group_names, model, metric, threshold_from, threshold_to
):
    """Features available from a pilot panel, before auditing the target panel."""
    features = {
        "model": model,
        "metric": metric,
        "threshold_from": float(threshold_from),
        "threshold_to": float(threshold_to),
        "threshold_midpoint": float((threshold_from + threshold_to) / 2),
        "threshold_width": float(threshold_to - threshold_from),
        "pilot_rows": int(len(labels)),
        "pilot_positive_rows": int(np.sum(labels == 1)),
        "pilot_negative_rows": int(np.sum(labels == 0)),
    }

    metric_label = 1 if metric == "FNR" else 0
    opposite_label = 1 - metric_label

    group_counts = []
    group_means = []
    group_sds = []
    group_q10 = []
    group_q50 = []
    group_q90 = []
    group_between = []
    group_near_from = []
    group_near_to = []
    group_opposite_counts = []
    group_opposite_means = []
    group_opposite_sds = []

    for group_index, group_name in enumerate(group_names):
        group_mask = group_codes == group_index
        metric_mask = group_mask & (labels == metric_label)
        opposite_mask = group_mask & (labels == opposite_label)

        metric_scores = scores[metric_mask]
        opposite_scores = scores[opposite_mask]

        n_metric = len(metric_scores)
        n_opposite = len(opposite_scores)

        if n_metric == 0 or n_opposite == 0:
            raise ValueError(
                f"Pilot panel has an empty class for group {group_name}; "
                "the identity split is unsuitable."
            )

        mean_score = float(np.mean(metric_scores))
        sd_score = float(np.std(metric_scores, ddof=1))
        q10, q50, q90 = np.quantile(metric_scores, [0.10, 0.50, 0.90])
        between = float(
            np.mean((metric_scores >= threshold_from) & (metric_scores < threshold_to))
        )
        near_from = float(
            np.mean(np.abs(metric_scores - threshold_from) <= 0.025)
        )
        near_to = float(
            np.mean(np.abs(metric_scores - threshold_to) <= 0.025)
        )

        opposite_mean = float(np.mean(opposite_scores))
        opposite_sd = float(np.std(opposite_scores, ddof=1))

        features[f"pilot_{group_name}_n_metric_class"] = int(n_metric)
        features[f"pilot_{group_name}_n_opposite_class"] = int(n_opposite)
        features[f"pilot_{group_name}_metric_score_mean"] = mean_score
        features[f"pilot_{group_name}_metric_score_sd"] = sd_score
        features[f"pilot_{group_name}_metric_score_q10"] = float(q10)
        features[f"pilot_{group_name}_metric_score_q50"] = float(q50)
        features[f"pilot_{group_name}_metric_score_q90"] = float(q90)
        features[f"pilot_{group_name}_fraction_between_thresholds"] = between
        features[f"pilot_{group_name}_fraction_near_threshold_from"] = near_from
        features[f"pilot_{group_name}_fraction_near_threshold_to"] = near_to
        features[f"pilot_{group_name}_opposite_score_mean"] = opposite_mean
        features[f"pilot_{group_name}_opposite_score_sd"] = opposite_sd

        group_counts.append(n_metric)
        group_means.append(mean_score)
        group_sds.append(sd_score)
        group_q10.append(float(q10))
        group_q50.append(float(q50))
        group_q90.append(float(q90))
        group_between.append(between)
        group_near_from.append(near_from)
        group_near_to.append(near_to)
        group_opposite_counts.append(n_opposite)
        group_opposite_means.append(opposite_mean)
        group_opposite_sds.append(opposite_sd)

    for name, values in [
        ("pilot_metric_class_group_count", group_counts),
        ("pilot_metric_score_mean", group_means),
        ("pilot_metric_score_sd", group_sds),
        ("pilot_metric_score_q10", group_q10),
        ("pilot_metric_score_q50", group_q50),
        ("pilot_metric_score_q90", group_q90),
        ("pilot_fraction_between_thresholds", group_between),
        ("pilot_fraction_near_threshold_from", group_near_from),
        ("pilot_fraction_near_threshold_to", group_near_to),
        ("pilot_opposite_class_group_count", group_opposite_counts),
        ("pilot_opposite_score_mean", group_opposite_means),
        ("pilot_opposite_score_sd", group_opposite_sds),
    ]:
        summarize_group_vector(values, name, features)

    features["pilot_metric_class_count_imbalance"] = (
        max(group_counts) / min(group_counts)
    )
    features["pilot_opposite_class_count_imbalance"] = (
        max(group_opposite_counts) / min(group_opposite_counts)
    )
    features["pilot_total_fraction_between_thresholds"] = float(
        np.mean((scores >= threshold_from) & (scores < threshold_to))
    )
    return features


def target_transition(
    labels, scores, group_codes, group_names, metric, threshold_from, threshold_to
):
    metrics_from = calculate_group_metrics(
        labels, scores, group_codes, len(group_names), threshold_from
    )[metric]
    metrics_to = calculate_group_metrics(
        labels, scores, group_codes, len(group_names), threshold_to
    )[metric]

    if not np.isfinite(metrics_from).all() or not np.isfinite(metrics_to).all():
        raise ValueError(
            "Target identity panel contains a demographic group without "
            "the required positive/negative denominator."
        )

    rounded_from = np.round(metrics_from, 12)
    rounded_to = np.round(metrics_to, 12)
    ranks_from = rankdata(-rounded_from, method="average")
    ranks_to = rankdata(-rounded_to, method="average")

    tau_result = kendalltau(ranks_from, ranks_to, variant="b")
    tau = float(tau_result.statistic) if np.isfinite(tau_result.statistic) else np.nan

    worst_from = set(np.flatnonzero(rounded_from == rounded_from.max()).tolist())
    worst_to = set(np.flatnonzero(rounded_to == rounded_to.max()).tolist())

    gap_from = float(metrics_from.max() - metrics_from.min())
    gap_to = float(metrics_to.max() - metrics_to.min())

    decision_flip_rate = float(
        np.mean(
            (scores >= threshold_from).astype(np.int8)
            != (scores >= threshold_to).astype(np.int8)
        )
    )

    return {
        "target_worst_group_from": "|".join(str(group_names[i]) for i in sorted(worst_from)),
        "target_worst_group_to": "|".join(str(group_names[i]) for i in sorted(worst_to)),
        "target_worst_group_set_changed": int(worst_from != worst_to),
        "target_kendall_tau_b": tau,
        "target_ranking_instability": 1.0 - tau if np.isfinite(tau) else np.nan,
        "target_normalized_ranking_instability": (
            (1.0 - tau) / 2.0 if np.isfinite(tau) else np.nan
        ),
        "target_gap_from": gap_from,
        "target_gap_to": gap_to,
        "target_signed_gap_change": gap_to - gap_from,
        "target_magnitude_change": abs(gap_to - gap_from),
        "target_decision_flip_rate": decision_flip_rate,
    }


def make_pipeline(estimator, numeric_columns, categorical_columns, scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_columns),
        ("categorical", categorical_pipe, categorical_columns),
    ])

    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def regression_metrics(y_true, y_pred):
    result = {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }
    if len(y_true) > 2 and np.unique(y_true).size > 1 and np.unique(y_pred).size > 1:
        corr = spearmanr(y_true, y_pred).statistic
    else:
        corr = np.nan
    result["spearman"] = float(corr) if np.isfinite(corr) else np.nan
    return result


def run_regression_cv(data, feature_columns, numeric_columns, categorical_columns):
    logo = LeaveOneGroupOut()
    prediction_rows = []
    metric_rows = []

    model_specs = {
        "dummy_mean": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: Ridge(alpha=10.0),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=25,
            min_samples_leaf=20,
            max_features=0.35,
            max_depth=8,
            n_jobs=4,
            random_state=RANDOM_SEED,
        ),
    }

    X = data[feature_columns]
    groups = data["fold"].to_numpy()

    for target in TARGETS_REGRESSION:
        valid = data[target].notna().to_numpy()
        X_target = X.loc[valid].reset_index(drop=True)
        y_target = data.loc[valid, target].to_numpy(dtype=float)
        fold_target = groups[valid]
        metadata = data.loc[valid, [
            "fold", "split_id", "model", "metric",
            "threshold_from", "threshold_to", target,
        ]].reset_index(drop=True)

        for model_name, estimator_factory in model_specs.items():
            oof = np.full(len(y_target), np.nan, dtype=float)

            for train_index, test_index in logo.split(X_target, y_target, fold_target):
                estimator = estimator_factory()
                pipeline = make_pipeline(
                    estimator, numeric_columns, categorical_columns,
                    scale_numeric=(model_name == "ridge"),
                )
                pipeline.fit(X_target.iloc[train_index], y_target[train_index])
                oof[test_index] = pipeline.predict(X_target.iloc[test_index])

            if not np.isfinite(oof).all():
                raise RuntimeError(f"Missing out-of-fold predictions for {target}/{model_name}.")

            overall = regression_metrics(y_target, oof)
            metric_rows.append({
                "target": target,
                "predictor": model_name,
                "evaluation": "leave_one_fold_out",
                **overall,
            })

            for fold in sorted(np.unique(fold_target)):
                mask = fold_target == fold
                fold_metrics = regression_metrics(y_target[mask], oof[mask])
                metric_rows.append({
                    "target": target,
                    "predictor": model_name,
                    "evaluation": f"held_out_fold_{fold}",
                    **fold_metrics,
                })

            print(
                f"  Regression CV complete: target={target}, predictor={model_name}",
                flush=True,
            )
            for meta, observed_value, prediction in zip(
                metadata.to_dict("records"), y_target, oof
            ):
                prediction_rows.append({
                    "target": target,
                    "predictor": model_name,
                    "fold": int(meta["fold"]),
                    "split_id": int(meta["split_id"]),
                    "model": meta["model"],
                    "metric": meta["metric"],
                    "threshold_from": float(meta["threshold_from"]),
                    "threshold_to": float(meta["threshold_to"]),
                    "observed": float(observed_value),
                    "predicted": float(prediction),
                })

    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def run_classification_cv(data, feature_columns, numeric_columns, categorical_columns):
    target = TARGET_CLASSIFICATION
    X = data[feature_columns]
    y = data[target].to_numpy(dtype=int)
    groups = data["fold"].to_numpy()
    logo = LeaveOneGroupOut()

    model_specs = {
        "dummy_prior": lambda: DummyClassifier(strategy="prior"),
        "logistic_balanced": lambda: LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_SEED,
        ),
    }

    prediction_rows = []
    metric_rows = []

    for model_name, estimator_factory in model_specs.items():
        probabilities = np.full(len(y), np.nan, dtype=float)
        for train_index, test_index in logo.split(X, y, groups):
            estimator = estimator_factory()
            pipeline = make_pipeline(
                estimator, numeric_columns, categorical_columns,
                scale_numeric=(model_name == "logistic_balanced"),
            )
            pipeline.fit(X.iloc[train_index], y[train_index])
            probabilities[test_index] = pipeline.predict_proba(X.iloc[test_index])[:, 1]

        predicted = (probabilities >= 0.5).astype(int)
        metrics = {
            "n": int(len(y)),
            "positive_rate": float(np.mean(y)),
            "brier": float(brier_score_loss(y, probabilities)),
            "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, predicted)),
            "roc_auc": (
                float(roc_auc_score(y, probabilities))
                if len(np.unique(y)) == 2 else np.nan
            ),
            "average_precision": float(average_precision_score(y, probabilities)),
        }
        metric_rows.append({
            "target": target,
            "predictor": model_name,
            "evaluation": "leave_one_fold_out",
            **metrics,
        })

        for fold in sorted(np.unique(groups)):
            mask = groups == fold
            fold_y = y[mask]
            fold_p = probabilities[mask]
            fold_pred = (fold_p >= 0.5).astype(int)
            metric_rows.append({
                "target": target,
                "predictor": model_name,
                "evaluation": f"held_out_fold_{fold}",
                "n": int(mask.sum()),
                "positive_rate": float(np.mean(fold_y)),
                "brier": float(brier_score_loss(fold_y, fold_p)),
                "balanced_accuracy_at_0_5": float(
                    balanced_accuracy_score(fold_y, fold_pred)
                ),
                "roc_auc": (
                    float(roc_auc_score(fold_y, fold_p))
                    if len(np.unique(fold_y)) == 2 else np.nan
                ),
                "average_precision": float(average_precision_score(fold_y, fold_p)),
            })

        print(
            f"  Classification CV complete: predictor={model_name}",
            flush=True,
        )
        metadata_records = data[
            ["fold", "split_id", "model", "metric", "threshold_from", "threshold_to"]
        ].to_dict("records")
        for meta, observed_value, probability in zip(metadata_records, y, probabilities):
            prediction_rows.append({
                "predictor": model_name,
                "fold": int(meta["fold"]),
                "split_id": int(meta["split_id"]),
                "model": meta["model"],
                "metric": meta["metric"],
                "threshold_from": float(meta["threshold_from"]),
                "threshold_to": float(meta["threshold_to"]),
                "observed": int(observed_value),
                "predicted_probability": float(probability),
            })

    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = Path(BFW_DATA_PATH)

    required = [
        FOLD_COLUMN, ID1_COLUMN, ID2_COLUMN, LABEL_COLUMN,
        GROUP_COLUMN, SECOND_GROUP_COLUMN, *MODELS,
    ]
    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path, usecols=required)

    if df[required].isna().any().any():
        missing = df[required].isna().sum()
        raise ValueError(f"Missing values found:\n{missing[missing > 0]}")

    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int8)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must be encoded as 0=impostor and 1=genuine.")

    # Build an endpoint-level identity table from both pair endpoints.
    endpoint_1 = df[[ID1_COLUMN, FOLD_COLUMN, GROUP_COLUMN]].rename(
        columns={ID1_COLUMN: "identity", GROUP_COLUMN: "group"}
    )
    endpoint_2 = df[[ID2_COLUMN, FOLD_COLUMN, SECOND_GROUP_COLUMN]].rename(
        columns={ID2_COLUMN: "identity", SECOND_GROUP_COLUMN: "group"}
    )
    identity_records = pd.concat([endpoint_1, endpoint_2], ignore_index=True)
    identity_check = identity_records.groupby("identity").agg(
        fold_count=(FOLD_COLUMN, "nunique"),
        group_count=("group", "nunique"),
        fold=(FOLD_COLUMN, "first"),
        group=("group", "first"),
    )
    if (identity_check["fold_count"] != 1).any() or (identity_check["group_count"] != 1).any():
        raise ValueError("Identity-to-fold or identity-to-demographic mapping is inconsistent.")

    identity_table = identity_check.reset_index().sort_values("identity").reset_index(drop=True)
    if identity_table["identity"].nunique() != 800:
        raise ValueError(f"Expected 800 identities; found {identity_table['identity'].nunique()}.")

    if set(identity_table["group"].unique()) != set(GROUPS):
        raise ValueError("Unexpected demographic group set.")

    identity_counts = pd.crosstab(identity_table["fold"], identity_table["group"])
    identity_counts = identity_counts.reindex(columns=GROUPS, fill_value=0)
    if not (identity_counts == 20).all().all():
        raise ValueError(
            "Expected exactly 20 identities per group per fold. "
            f"Observed:\n{identity_counts}"
        )

    identity_to_position = {
        identity: position
        for position, identity in enumerate(identity_table["identity"].to_numpy())
    }
    row_id1 = df[ID1_COLUMN].map(identity_to_position).to_numpy(dtype=np.int32)
    row_id2 = df[ID2_COLUMN].map(identity_to_position).to_numpy(dtype=np.int32)

    group_to_code = {group: i for i, group in enumerate(GROUPS)}
    row_group_codes = df[GROUP_COLUMN].map(group_to_code).to_numpy(dtype=np.int8)
    fold_values = df[FOLD_COLUMN].to_numpy()
    scores_by_model = {
        model: df[model].to_numpy(dtype=np.float64) for model in MODELS
    }

    folds = np.sort(identity_table["fold"].unique())
    fold_identity_positions = {
        fold: np.flatnonzero(identity_table["fold"].to_numpy() == fold)
        for fold in folds
    }
    identity_groups = identity_table["group"].to_numpy()

    # Confirm each row's endpoint identities belong to the same predefined fold.
    identity_folds = identity_table["fold"].to_numpy()
    if not np.array_equal(identity_folds[row_id1], identity_folds[row_id2]):
        raise ValueError("A comparison connects identities from different folds.")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    split_audit = []

    print(f"Rows: {len(df):,}")
    print(f"Identity counts per fold/group:\n{identity_counts}")
    print(f"Splits per fold: {SPLITS_PER_FOLD}")
    print("Design: identity-disjoint pilot and target panels; 10 identities/group/panel.")
    print("Validation: leave-one-original-fold-out; all splits from a fold stay together.")

    for fold in folds:
        fold_row_positions = np.flatnonzero(fold_values == fold)
        fold_ids = fold_identity_positions[fold]

        for split_id in range(1, SPLITS_PER_FOLD + 1):
            pilot_ids = []
            target_ids = []

            for group in GROUPS:
                group_ids = identity_table.loc[
                    (identity_table["fold"] == fold)
                    & (identity_table["group"] == group),
                    "identity",
                ].to_numpy()

                shuffled = rng.permutation(group_ids)
                pilot_ids.extend(shuffled[:IDENTITIES_PER_GROUP_PER_PANEL].tolist())
                target_ids.extend(shuffled[IDENTITIES_PER_GROUP_PER_PANEL:].tolist())

            pilot_positions = np.array(
                [identity_to_position[x] for x in pilot_ids], dtype=np.int32
            )
            target_positions = np.array(
                [identity_to_position[x] for x in target_ids], dtype=np.int32
            )

            pilot_mask = (
                np.isin(row_id1[fold_row_positions], pilot_positions)
                & np.isin(row_id2[fold_row_positions], pilot_positions)
            )
            target_mask = (
                np.isin(row_id1[fold_row_positions], target_positions)
                & np.isin(row_id2[fold_row_positions], target_positions)
            )

            pilot_rows = fold_row_positions[pilot_mask]
            target_rows = fold_row_positions[target_mask]

            if len(pilot_rows) == 0 or len(target_rows) == 0:
                raise ValueError(f"Empty pilot or target panel in fold {fold}, split {split_id}.")

            # Identity-disjointness check.
            if set(pilot_ids) & set(target_ids):
                raise RuntimeError("Pilot and target identity sets overlap.")

            split_audit.append({
                "fold": int(fold),
                "split_id": split_id,
                "pilot_identity_count": len(pilot_ids),
                "target_identity_count": len(target_ids),
                "pilot_rows": len(pilot_rows),
                "target_rows": len(target_rows),
                "pilot_positive_rows": int(np.sum(labels[pilot_rows] == 1)),
                "pilot_negative_rows": int(np.sum(labels[pilot_rows] == 0)),
                "target_positive_rows": int(np.sum(labels[target_rows] == 1)),
                "target_negative_rows": int(np.sum(labels[target_rows] == 0)),
            })

            pilot_labels = labels[pilot_rows]
            target_labels = labels[target_rows]
            pilot_groups = row_group_codes[pilot_rows]
            target_groups = row_group_codes[target_rows]

            for model in MODELS:
                pilot_scores = scores_by_model[model][pilot_rows]
                target_scores = scores_by_model[model][target_rows]

                target_by_threshold = {
                    threshold: calculate_group_metrics(
                        target_labels, target_scores, target_groups,
                        len(GROUPS), threshold
                    )
                    for threshold in THRESHOLDS
                }

                for metric in METRICS:
                    for threshold_from, threshold_to in zip(
                        THRESHOLDS[:-1], THRESHOLDS[1:]
                    ):
                        features = pilot_features(
                            pilot_labels,
                            pilot_scores,
                            pilot_groups,
                            GROUPS,
                            model,
                            metric,
                            threshold_from,
                            threshold_to,
                        )

                        target = target_transition(
                            target_labels,
                            target_scores,
                            target_groups,
                            GROUPS,
                            metric,
                            threshold_from,
                            threshold_to,
                        )

                        rows.append({
                            "fold": int(fold),
                            "split_id": split_id,
                            **features,
                            **target,
                        })

        print(f"Completed fold {fold}: {SPLITS_PER_FOLD} identity-disjoint splits.")
        gc.collect()

    dataset = pd.DataFrame(rows)
    audit_df = pd.DataFrame(split_audit)

    expected_rows = (
        len(folds) * SPLITS_PER_FOLD * len(MODELS)
        * len(METRICS) * (len(THRESHOLDS) - 1)
    )
    if len(dataset) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} examples; created {len(dataset)}.")

    # Verify no missing primary predictors/targets and no duplicated conditions.
    key_columns = [
        "fold", "split_id", "model", "metric", "threshold_from", "threshold_to"
    ]
    if dataset.duplicated(key_columns).any():
        raise RuntimeError("Duplicate predictive examples detected.")

    # Kendall tau-b is legitimately undefined when a target ranking is
    # completely tied. Preserve those cases as NaN and exclude them only from
    # the corresponding ranking-regression fit.
    all_target_columns = [
        "target_magnitude_change",
        "target_normalized_ranking_instability",
        "target_decision_flip_rate",
        TARGET_CLASSIFICATION,
    ]
    required_defined_targets = [
        "target_magnitude_change",
        "target_decision_flip_rate",
        TARGET_CLASSIFICATION,
    ]
    if dataset[required_defined_targets].isna().any().any():
        raise RuntimeError("Missing non-ranking predictive targets detected.")
    undefined_ranking_targets = int(
        dataset["target_normalized_ranking_instability"].isna().sum()
    )

    dataset_path = OUTPUT_DIR / f"{PREFIX}_dataset.csv"
    audit_path = OUTPUT_DIR / f"{PREFIX}_split_audit.csv"
    manifest_path = OUTPUT_DIR / f"{PREFIX}_manifest.json"

    dataset.to_csv(dataset_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    # Features are pilot-only; fold/split identifiers are grouping metadata,
    # not predictors. Target columns are never included in X.
    excluded = set(["fold", "split_id"] + [
        "target_magnitude_change",
        "target_normalized_ranking_instability",
        "target_decision_flip_rate",
        TARGET_CLASSIFICATION,
        "target_worst_group_from",
        "target_worst_group_to",
        "target_kendall_tau_b",
        "target_ranking_instability",
        "target_gap_from",
        "target_gap_to",
        "target_signed_gap_change",
    ])
    feature_columns = [column for column in dataset.columns if column not in excluded]

    categorical_columns = ["model", "metric"]
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]

    regression_predictions, regression_metrics_df = run_regression_cv(
        dataset, feature_columns, numeric_columns, categorical_columns
    )
    classification_predictions, classification_metrics_df = run_classification_cv(
        dataset, feature_columns, numeric_columns, categorical_columns
    )

    regression_predictions.to_csv(
        OUTPUT_DIR / f"{PREFIX}_regression_oof_predictions.csv", index=False
    )
    regression_metrics_df.to_csv(
        OUTPUT_DIR / f"{PREFIX}_regression_metrics.csv", index=False
    )
    classification_predictions.to_csv(
        OUTPUT_DIR / f"{PREFIX}_classification_oof_predictions.csv", index=False
    )
    classification_metrics_df.to_csv(
        OUTPUT_DIR / f"{PREFIX}_classification_metrics.csv", index=False
    )

    manifest = {
        "dataset_path": str(data_path),
        "dataset_sha256": sha256_file(data_path),
        "rows_in_source": int(len(df)),
        "identities": int(len(identity_table)),
        "folds": [int(x) for x in folds],
        "groups": GROUPS,
        "models": MODELS,
        "thresholds": THRESHOLDS,
        "metrics": METRICS,
        "splits_per_fold": SPLITS_PER_FOLD,
        "identities_per_group_per_panel": IDENTITIES_PER_GROUP_PER_PANEL,
        "predictive_examples": int(len(dataset)),
        "random_seed": RANDOM_SEED,
        "validation": "LeaveOneGroupOut by original BFW fold",
        "pilot_target_design": (
            "Within each fold, split 20 identities/group into disjoint "
            "10-identity pilot and 10-identity target panels. Retain only "
            "comparison rows whose id1 and id2 are both inside the same panel."
        ),
        "targets": all_target_columns,
        "features": feature_columns,
        "limitations": [
            "Only five original folds are available as outer generalization groups.",
            "Repeated identity splits within a fold are correlated; all are held out together.",
            "This predicts instability on an identity-disjoint target panel from pilot-panel summaries, not a universal population quantity.",
            "Only three precomputed model score columns are available.",
            "This does not establish generalization to unseen model architectures or datasets.",
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\n" + "=" * 72)
    print("AUDITFLUX — PILOT-TO-HELD-OUT-IDENTITY PREDICTIVE FEASIBILITY")
    print("=" * 72)
    print(f"Predictive dataset rows: {len(dataset):,} (expected {expected_rows:,})")
    print(f"Predictor columns: {len(feature_columns)}")
    print(f"Splits: {len(audit_df):,}")
    print(f"Undefined ranking targets (retained as NaN): {undefined_ranking_targets:,}")
    print(f"Observed worst-group change rate: {dataset[TARGET_CLASSIFICATION].mean():.4f}")
    print("\nRegression metrics (pooled out-of-fold):")
    print(
        regression_metrics_df[
            regression_metrics_df["evaluation"] == "leave_one_fold_out"
        ].to_string(index=False)
    )
    print("\nClassification metrics (pooled out-of-fold):")
    print(
        classification_metrics_df[
            classification_metrics_df["evaluation"] == "leave_one_fold_out"
        ].to_string(index=False)
    )

    print("\nSaved outputs:")
    print(dataset_path)
    print(audit_path)
    print(OUTPUT_DIR / f"{PREFIX}_regression_oof_predictions.csv")
    print(OUTPUT_DIR / f"{PREFIX}_regression_metrics.csv")
    print(OUTPUT_DIR / f"{PREFIX}_classification_oof_predictions.csv")
    print(OUTPUT_DIR / f"{PREFIX}_classification_metrics.csv")
    print(manifest_path)

    print("\nInterpretation:")
    print(
        "This is a predictive feasibility experiment. Pilot features are computed "
        "from one identity panel; instability targets are measured on a disjoint "
        "identity panel. All splits from an original fold are held out together."
    )
    print(
        "The five-fold outer validation is the primary leakage control. Repeated "
        "splits are not independent observations and must not be randomly split "
        "between training and test sets."
    )


if __name__ == "__main__":
    main()
