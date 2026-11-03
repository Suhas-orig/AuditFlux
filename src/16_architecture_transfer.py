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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import BFW_DATA_PATH, OUTPUT_DIR

# ---------------------------------------------------------------------
# Study design: architecture-held-out AND identity-fold-held-out transfer
# ---------------------------------------------------------------------
MODELS = ["vgg16", "resnet50", "senet50"]
GROUPS = ["AF", "AM", "BF", "BM", "IF", "IM", "WF", "WM"]
METRICS = ["FNR", "FPR"]
FPR_QUANTILES = [0.90, 0.95, 0.975, 0.99]
IDENTITIES_PER_GROUP_PER_PANEL = 10
SPLITS_PER_FOLD = 20
RANDOM_SEED = 20261107
PREFIX = "16_architecture_transfer"
TARGET_REGRESSION = ["target_magnitude_change", "target_normalized_ranking_instability"]
TARGET_CLASSIFICATION = "target_worst_group_set_changed"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def calculate_group_rates(labels, scores, group_codes, threshold):
    pred = scores >= threshold
    rates = {}
    for metric in METRICS:
        values = np.full(len(GROUPS), np.nan, dtype=float)
        for g in range(len(GROUPS)):
            mask = group_codes == g
            if metric == "FNR":
                denom = np.sum(mask & (labels == 1))
                numer = np.sum(mask & (labels == 1) & (~pred))
            else:
                denom = np.sum(mask & (labels == 0))
                numer = np.sum(mask & (labels == 0) & pred)
            if denom:
                values[g] = numer / denom
        if not np.isfinite(values).all():
            raise ValueError(f"Missing group denominator for {metric} at threshold {threshold}.")
        rates[metric] = values
    return rates


def ranking_instability(a, b):
    ra = rankdata(-np.round(a, 12), method="average")
    rb = rankdata(-np.round(b, 12), method="average")
    tau = kendalltau(ra, rb, variant="b").statistic
    return float((1.0 - tau) / 2.0) if np.isfinite(tau) else np.nan


def worst_set(values):
    v = np.round(np.asarray(values, dtype=float), 12)
    return tuple(np.flatnonzero(v == np.max(v)).tolist())


def build_pilot_features(labels, scores, group_codes, q_from, q_to, t_from, t_to):
    """Model-agnostic features; thresholds are calibrated on pilot impostors."""
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    neg_mean = float(np.mean(neg))
    neg_sd = float(np.std(neg, ddof=1))
    pos_mean = float(np.mean(pos))
    pos_sd = float(np.std(pos, ddof=1))
    if neg_sd <= 0 or pos_sd <= 0:
        raise ValueError("Degenerate pilot score distribution.")

    features = {
        "metric": None,  # set by caller
        "fpr_quantile_from": float(q_from),
        "fpr_quantile_to": float(q_to),
        "fpr_quantile_midpoint": float((q_from + q_to) / 2),
        "fpr_quantile_width": float(q_to - q_from),
        "pilot_rows": int(len(labels)),
        "pilot_positive_rows": int(np.sum(labels == 1)),
        "pilot_negative_rows": int(np.sum(labels == 0)),
        "pilot_neg_score_mean": neg_mean,
        "pilot_neg_score_sd": neg_sd,
        "pilot_pos_score_mean_z": (pos_mean - neg_mean) / neg_sd,
        "pilot_pos_score_sd_ratio": pos_sd / neg_sd,
        "pilot_threshold_from_z": (t_from - neg_mean) / neg_sd,
        "pilot_threshold_to_z": (t_to - neg_mean) / neg_sd,
        "pilot_threshold_width_z": (t_to - t_from) / neg_sd,
    }

    rates_from = calculate_group_rates(labels, scores, group_codes, t_from)
    rates_to = calculate_group_rates(labels, scores, group_codes, t_to)

    for metric in METRICS:
        for g, group in enumerate(GROUPS):
            mask = group_codes == g
            gneg = scores[mask & (labels == 0)]
            gpos = scores[mask & (labels == 1)]
            if len(gneg) == 0 or len(gpos) == 0:
                raise ValueError(f"Empty pilot class for group {group}.")
            features[f"pilot_{group}_n_neg"] = int(len(gneg))
            features[f"pilot_{group}_n_pos"] = int(len(gpos))
            features[f"pilot_{group}_neg_mean_z"] = (float(np.mean(gneg)) - neg_mean) / neg_sd
            features[f"pilot_{group}_neg_sd_ratio"] = float(np.std(gneg, ddof=1)) / neg_sd
            features[f"pilot_{group}_pos_mean_z"] = (float(np.mean(gpos)) - neg_mean) / neg_sd
            features[f"pilot_{group}_pos_sd_ratio"] = float(np.std(gpos, ddof=1)) / neg_sd
            features[f"pilot_{group}_neg_q10_z"] = (float(np.quantile(gneg, .10)) - neg_mean) / neg_sd
            features[f"pilot_{group}_neg_q50_z"] = (float(np.quantile(gneg, .50)) - neg_mean) / neg_sd
            features[f"pilot_{group}_neg_q90_z"] = (float(np.quantile(gneg, .90)) - neg_mean) / neg_sd
            features[f"pilot_{group}_pos_q10_z"] = (float(np.quantile(gpos, .10)) - neg_mean) / neg_sd
            features[f"pilot_{group}_pos_q50_z"] = (float(np.quantile(gpos, .50)) - neg_mean) / neg_sd
            features[f"pilot_{group}_pos_q90_z"] = (float(np.quantile(gpos, .90)) - neg_mean) / neg_sd
            features[f"pilot_{group}_{metric}_from"] = float(rates_from[metric][g])
            features[f"pilot_{group}_{metric}_to"] = float(rates_to[metric][g])
            features[f"pilot_{group}_{metric}_change"] = float(rates_to[metric][g] - rates_from[metric][g])

        v_from = rates_from[metric]
        v_to = rates_to[metric]
        features[f"pilot_{metric}_gap_from"] = float(v_from.max() - v_from.min())
        features[f"pilot_{metric}_gap_to"] = float(v_to.max() - v_to.min())
        features[f"pilot_{metric}_absolute_gap_change"] = float(abs((v_to.max()-v_to.min()) - (v_from.max()-v_from.min())))
        features[f"pilot_{metric}_ranking_instability"] = ranking_instability(v_from, v_to)
        features[f"pilot_{metric}_worst_group_changed"] = int(worst_set(v_from) != worst_set(v_to))

    return features


def build_target(labels, scores, group_codes, metric, t_from, t_to):
    a = calculate_group_rates(labels, scores, group_codes, t_from)[metric]
    b = calculate_group_rates(labels, scores, group_codes, t_to)[metric]
    gap_a = float(a.max() - a.min())
    gap_b = float(b.max() - b.min())
    return {
        "target_gap_from": gap_a,
        "target_gap_to": gap_b,
        "target_magnitude_change": abs(gap_b - gap_a),
        "target_normalized_ranking_instability": ranking_instability(a, b),
        "target_worst_group_set_changed": int(worst_set(a) != worst_set(b)),
        "target_worst_group_from": "|".join(GROUPS[i] for i in worst_set(a)),
        "target_worst_group_to": "|".join(GROUPS[i] for i in worst_set(b)),
    }


def make_pipeline(estimator, numeric_cols, categorical_cols, scale):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", num_pipe, numeric_cols),
        ("categorical", cat_pipe, categorical_cols),
    ])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def regression_metrics(y, p):
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.unique(y).size > 1 else np.nan,
        "spearman": float(spearmanr(y, p).statistic) if len(y) > 2 and np.unique(y).size > 1 and np.unique(p).size > 1 else np.nan,
    }


def classification_metrics(y, p):
    hard = (p >= .5).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "brier": float(brier_score_loss(y, p)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, hard)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "average_precision": float(average_precision_score(y, p)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(BFW_DATA_PATH)
    cols = ["fold", "id1", "id2", "label", "a1", "a2", *MODELS]
    print(f"Loading BFW table: {source}", flush=True)
    df = pd.read_csv(source, usecols=cols)
    if df[cols].isna().any().any():
        raise ValueError("Missing values detected in required BFW columns.")
    labels = df["label"].to_numpy(dtype=np.int8)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Expected label 0=impostor and 1=genuine.")

    ep1 = df[["id1", "fold", "a1"]].rename(columns={"id1": "identity", "a1": "group"})
    ep2 = df[["id2", "fold", "a2"]].rename(columns={"id2": "identity", "a2": "group"})
    identity_records = pd.concat([ep1, ep2], ignore_index=True)
    identity_check = identity_records.groupby("identity").agg(
        fold_count=("fold", "nunique"), group_count=("group", "nunique"),
        fold=("fold", "first"), group=("group", "first"),
    )
    if (identity_check["fold_count"] != 1).any() or (identity_check["group_count"] != 1).any():
        raise ValueError("Identity-to-fold/group mapping is inconsistent.")
    identity_table = identity_check.reset_index().sort_values("identity").reset_index(drop=True)
    if len(identity_table) != 800:
        raise ValueError(f"Expected 800 identities, found {len(identity_table)}.")
    counts = pd.crosstab(identity_table["fold"], identity_table["group"]).reindex(columns=GROUPS, fill_value=0)
    if not (counts == 20).all().all():
        raise ValueError(f"Expected 20 identities per fold/group.\n{counts}")

    identity_to_pos = {x: i for i, x in enumerate(identity_table["identity"].to_numpy())}
    row_i = df["id1"].map(identity_to_pos).to_numpy(dtype=np.int32)
    row_j = df["id2"].map(identity_to_pos).to_numpy(dtype=np.int32)
    identity_fold = identity_table["fold"].to_numpy()
    identity_group = identity_table["group"].to_numpy()
    if not np.array_equal(identity_fold[row_i], identity_fold[row_j]):
        raise ValueError("Cross-fold identity comparisons found.")
    group_to_code = {g: i for i, g in enumerate(GROUPS)}
    row_groups = df["a1"].map(group_to_code).to_numpy(dtype=np.int8)
    row_folds = df["fold"].to_numpy()
    score_arrays = {m: df[m].to_numpy(dtype=float) for m in MODELS}
    fold_ids = sorted(identity_table["fold"].unique())
    rng = np.random.default_rng(RANDOM_SEED)

    examples = []
    split_audit = []
    for fold in fold_ids:
        fold_rows = np.flatnonzero(row_folds == fold)
        for split_id in range(1, SPLITS_PER_FOLD + 1):
            pilot_ids, target_ids = [], []
            for group in GROUPS:
                ids = identity_table.loc[(identity_table.fold == fold) & (identity_table.group == group), "identity"].to_numpy()
                shuffled = rng.permutation(ids)
                pilot_ids.extend(shuffled[:IDENTITIES_PER_GROUP_PER_PANEL].tolist())
                target_ids.extend(shuffled[IDENTITIES_PER_GROUP_PER_PANEL:].tolist())
            ppos = np.array([identity_to_pos[x] for x in pilot_ids], dtype=np.int32)
            tpos = np.array([identity_to_pos[x] for x in target_ids], dtype=np.int32)
            p_mask = np.isin(row_i[fold_rows], ppos) & np.isin(row_j[fold_rows], ppos)
            t_mask = np.isin(row_i[fold_rows], tpos) & np.isin(row_j[fold_rows], tpos)
            p_rows, t_rows = fold_rows[p_mask], fold_rows[t_mask]
            if len(p_rows) == 0 or len(t_rows) == 0 or set(pilot_ids) & set(target_ids):
                raise RuntimeError(f"Invalid pilot/target split: fold={fold}, split={split_id}")
            split_audit.append({
                "fold": int(fold), "split_id": split_id,
                "pilot_identity_count": len(pilot_ids), "target_identity_count": len(target_ids),
                "pilot_rows": len(p_rows), "target_rows": len(t_rows),
            })

            p_labels, t_labels = labels[p_rows], labels[t_rows]
            p_groups, t_groups = row_groups[p_rows], row_groups[t_rows]
            for model in MODELS:
                p_scores, t_scores = score_arrays[model][p_rows], score_arrays[model][t_rows]
                impostor_scores = p_scores[p_labels == 0]
                thresholds = np.quantile(impostor_scores, FPR_QUANTILES)
                if np.any(np.diff(thresholds) <= 0):
                    raise ValueError(f"Non-increasing calibrated thresholds for {model}, fold {fold}, split {split_id}.")
                for metric in METRICS:
                    for k in range(len(FPR_QUANTILES) - 1):
                        q0, q1 = FPR_QUANTILES[k], FPR_QUANTILES[k + 1]
                        t0, t1 = float(thresholds[k]), float(thresholds[k + 1])
                        feat = build_pilot_features(p_labels, p_scores, p_groups, q0, q1, t0, t1)
                        feat["metric"] = metric
                        target = build_target(t_labels, t_scores, t_groups, metric, t0, t1)
                        examples.append({
                            "fold": int(fold), "split_id": split_id, "model": model,
                            "metric": metric, "fpr_quantile_from": q0, "fpr_quantile_to": q1,
                            "threshold_from_raw": t0, "threshold_to_raw": t1,
                            **{k: v for k, v in feat.items() if k != "metric"},
                            **target,
                        })
        print(f"Built fold {fold}", flush=True)
        gc.collect()

    data = pd.DataFrame(examples)
    expected = len(fold_ids) * SPLITS_PER_FOLD * len(MODELS) * len(METRICS) * (len(FPR_QUANTILES)-1)
    if len(data) != expected:
        raise RuntimeError(f"Expected {expected} rows, built {len(data)}")
    if data.duplicated(["fold", "split_id", "model", "metric", "fpr_quantile_from", "fpr_quantile_to"]).any():
        raise RuntimeError("Duplicate predictive examples.")

    # Model identity and raw thresholds are metadata only, never predictors.
    excluded = {
        "fold", "split_id", "model", "threshold_from_raw", "threshold_to_raw",
        "target_gap_from", "target_gap_to", "target_magnitude_change",
        "target_normalized_ranking_instability", "target_worst_group_set_changed",
        "target_worst_group_from", "target_worst_group_to",
    }
    context_cols = ["metric", "fpr_quantile_from", "fpr_quantile_to", "fpr_quantile_midpoint", "fpr_quantile_width"]
    full_cols = [c for c in data.columns if c not in excluded and c not in {"pilot_neg_score_mean", "pilot_neg_score_sd"}]
    feature_sets = {"context_only": context_cols, "pilot_calibrated_features": full_cols}
    cat_cols = ["metric"]
    results_reg, results_cls, reg_metrics, cls_metrics, lift_rows = [], [], [], [], []

    # Crossed outer blocks: unseen architecture AND unseen BFW fold.
    blocks = [(m, f) for m in MODELS for f in fold_ids]
    for held_model, held_fold in blocks:
        test_mask = (data.model == held_model) & (data.fold == held_fold)
        train_mask = (data.model != held_model) & (data.fold != held_fold)
        train = data.loc[train_mask].reset_index(drop=True)
        test = data.loc[test_mask].reset_index(drop=True)
        if len(train) == 0 or len(test) == 0:
            raise RuntimeError(f"Empty train/test block {held_model}/{held_fold}")
        if held_model in set(train.model) or held_fold in set(train.fold):
            raise RuntimeError("Architecture/fold leakage in training partition.")
        block_id = f"{held_model}_fold_{held_fold}"
        print(f"Training for held-out block {block_id}: train={len(train)}, test={len(test)}", flush=True)

        for target_name in TARGET_REGRESSION:
            train_target = train[train[target_name].notna()].reset_index(drop=True)
            test_target = test[test[target_name].notna()].reset_index(drop=True)
            for feature_name, cols_used in feature_sets.items():
                numeric_cols = [c for c in cols_used if c not in cat_cols]
                for predictor, factory, scale in [
                    ("dummy_mean", lambda: DummyRegressor(strategy="mean"), False),
                    ("ridge", lambda: Ridge(alpha=10.0), True),
                    ("extra_trees", lambda: ExtraTreesRegressor(n_estimators=15, min_samples_leaf=25, max_features=.25, max_depth=6, n_jobs=1, random_state=RANDOM_SEED), False),
                ]:
                    pipe = make_pipeline(factory(), numeric_cols, cat_cols, scale)
                    pipe.fit(train_target[cols_used], train_target[target_name].to_numpy(float))
                    pred = pipe.predict(test_target[cols_used])
                    y = test_target[target_name].to_numpy(float)
                    for rec, obs, pr in zip(test_target.to_dict("records"), y, pred):
                        results_reg.append({
                            "held_out_model": held_model, "held_out_fold": int(held_fold),
                            "block_id": block_id, "target": target_name, "feature_set": feature_name,
                            "predictor": predictor, "fold": int(rec["fold"]), "split_id": int(rec["split_id"]),
                            "model": rec["model"], "metric": rec["metric"],
                            "fpr_quantile_from": rec["fpr_quantile_from"], "fpr_quantile_to": rec["fpr_quantile_to"],
                            "observed": float(obs), "predicted": float(pr),
                        })
                    met = regression_metrics(y, pred)
                    reg_metrics.append({
                        "held_out_model": held_model, "held_out_fold": int(held_fold),
                        "block_id": block_id, "target": target_name, "feature_set": feature_name,
                        "predictor": predictor, **met,
                    })

        # Classification target
        for feature_name, cols_used in feature_sets.items():
            numeric_cols = [c for c in cols_used if c not in cat_cols]
            for predictor, factory, scale in [
                ("dummy_prior", lambda: DummyClassifier(strategy="prior"), False),
                ("logistic_balanced", lambda: LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear", random_state=RANDOM_SEED), True),
                ("extra_trees_balanced", lambda: ExtraTreesClassifier(n_estimators=15, min_samples_leaf=25, max_features=.25, max_depth=6, class_weight="balanced", n_jobs=1, random_state=RANDOM_SEED), False),
            ]:
                pipe = make_pipeline(factory(), numeric_cols, cat_cols, scale)
                pipe.fit(train[cols_used], train[TARGET_CLASSIFICATION].to_numpy(int))
                prob = pipe.predict_proba(test[cols_used])[:, 1]
                y = test[TARGET_CLASSIFICATION].to_numpy(int)
                for rec, obs, pr in zip(test.to_dict("records"), y, prob):
                    results_cls.append({
                        "held_out_model": held_model, "held_out_fold": int(held_fold),
                        "block_id": block_id, "feature_set": feature_name, "predictor": predictor,
                        "fold": int(rec["fold"]), "split_id": int(rec["split_id"]),
                        "model": rec["model"], "metric": rec["metric"],
                        "fpr_quantile_from": rec["fpr_quantile_from"], "fpr_quantile_to": rec["fpr_quantile_to"],
                        "observed": int(obs), "predicted_probability": float(pr),
                    })
                cls_metrics.append({
                    "held_out_model": held_model, "held_out_fold": int(held_fold),
                    "block_id": block_id, "feature_set": feature_name, "predictor": predictor,
                    **classification_metrics(y, prob),
                })

    reg_df = pd.DataFrame(results_reg)
    cls_df = pd.DataFrame(results_cls)
    reg_met_df = pd.DataFrame(reg_metrics)
    cls_met_df = pd.DataFrame(cls_metrics)

    # Pooled summaries aggregate all OOF blocks, but block metrics remain primary diagnostics.
    pooled_reg = []
    for keys, part in reg_df.groupby(["target", "feature_set", "predictor"]):
        pooled_reg.append({"held_out_model": "ALL", "held_out_fold": -1, "block_id": "POOLED_OOF", "target": keys[0], "feature_set": keys[1], "predictor": keys[2], **regression_metrics(part.observed.to_numpy(float), part.predicted.to_numpy(float))})
    reg_met_df = pd.concat([reg_met_df, pd.DataFrame(pooled_reg)], ignore_index=True)
    pooled_cls = []
    for keys, part in cls_df.groupby(["feature_set", "predictor"]):
        pooled_cls.append({"held_out_model": "ALL", "held_out_fold": -1, "block_id": "POOLED_OOF", "feature_set": keys[0], "predictor": keys[1], **classification_metrics(part.observed.to_numpy(int), part.predicted_probability.to_numpy(float))})
    cls_met_df = pd.concat([cls_met_df, pd.DataFrame(pooled_cls)], ignore_index=True)

    # Feature lift: paired block-level difference in MAE / ROC-AUC versus context baseline.
    for target_name in TARGET_REGRESSION:
        for predictor in ["ridge", "extra_trees"]:
            sub = reg_met_df[(reg_met_df.target == target_name) & (reg_met_df.predictor == predictor) & (reg_met_df.block_id != "POOLED_OOF")]
            wide = sub.pivot(index="block_id", columns="feature_set", values="mae")
            for block_id, row in wide.iterrows():
                lift_rows.append({"task": target_name, "predictor": predictor, "block_id": block_id,
                                  "context_mae": row.get("context_only", np.nan),
                                  "pilot_mae": row.get("pilot_calibrated_features", np.nan),
                                  "mae_improvement": row.get("context_only", np.nan) - row.get("pilot_calibrated_features", np.nan)})
    sub = cls_met_df[(cls_met_df.predictor.isin(["logistic_balanced", "extra_trees_balanced"])) & (cls_met_df.block_id != "POOLED_OOF")]
    for predictor in sub.predictor.unique():
        part = sub[sub.predictor == predictor].pivot(index="block_id", columns="feature_set", values="roc_auc")
        for block_id, row in part.iterrows():
            lift_rows.append({"task": TARGET_CLASSIFICATION, "predictor": predictor, "block_id": block_id,
                              "context_roc_auc": row.get("context_only", np.nan),
                              "pilot_roc_auc": row.get("pilot_calibrated_features", np.nan),
                              "roc_auc_improvement": row.get("pilot_calibrated_features", np.nan) - row.get("context_only", np.nan)})
    lift_df = pd.DataFrame(lift_rows)

    # Save the constructed dataset and audit alongside model outputs.
    dataset_path = OUTPUT_DIR / f"{PREFIX}_dataset.csv"
    audit_path = OUTPUT_DIR / f"{PREFIX}_split_audit.csv"
    data.to_csv(dataset_path, index=False)
    pd.DataFrame(split_audit).to_csv(audit_path, index=False)
    reg_df.to_csv(OUTPUT_DIR / f"{PREFIX}_regression_oof_predictions.csv", index=False)
    cls_df.to_csv(OUTPUT_DIR / f"{PREFIX}_classification_oof_predictions.csv", index=False)
    reg_met_df.to_csv(OUTPUT_DIR / f"{PREFIX}_regression_metrics.csv", index=False)
    cls_met_df.to_csv(OUTPUT_DIR / f"{PREFIX}_classification_metrics.csv", index=False)
    lift_df.to_csv(OUTPUT_DIR / f"{PREFIX}_feature_lift_comparison.csv", index=False)

    manifest = {
        "experiment": "Program 16 - Leave-One-Architecture-Out Transfer with Held-Out Identity Fold",
        "source_path": str(source), "source_sha256": sha256_file(source),
        "source_rows": int(len(df)), "identity_count": int(len(identity_table)),
        "architectures": MODELS, "architecture_domains": 3,
        "folds": [int(x) for x in fold_ids], "folds_per_architecture": 5,
        "splits_per_fold": SPLITS_PER_FOLD, "identities_per_group_per_panel": IDENTITIES_PER_GROUP_PER_PANEL,
        "pilot_target": "identity-disjoint induced subgraphs; both endpoints must belong to the same panel",
        "operating_point_calibration": {
            "method": "model-specific global impostor-score quantiles estimated from pilot panel only",
            "quantiles": FPR_QUANTILES,
            "interpretation": "thresholds are calibrated to comparable nominal pooled pilot FPR operating points; realized target FPR may differ",
            "demographic_policy": "one global threshold per architecture/split; no group-specific thresholding",
        },
        "predictor_features": "architecture identifier and raw score thresholds excluded; score summaries standardized relative to pooled pilot impostor distribution; includes pilot group FNR/FPR at calibrated operating points",
        "outer_validation": "15 crossed blocks: held-out architecture x held-out original BFW fold",
        "training_exclusion": "training excludes all rows from held-out architecture AND all rows from held-out fold",
        "block_count": len(blocks), "target_metrics": METRICS,
        "regression_targets": TARGET_REGRESSION, "classification_target": TARGET_CLASSIFICATION,
        "predictors": ["dummy", "ridge", "extra_trees", "logistic", "extra_trees_classifier"],
        "random_seed": RANDOM_SEED,
        "limitations": [
            "Only three architecture domains are available; leave-one-architecture-out results are exploratory and cannot support broad architecture-population inference.",
            "Each outer block has one unseen architecture and one unseen BFW fold; this is deliberately stringent and may understate performance relative to architecture-only transfer.",
            "Pilot impostor quantiles calibrate a single global threshold; this does not guarantee identical realized target FPR across architectures or groups.",
            "The target graph is incomplete and comparisons share identity endpoints; results are conditional on the observed BFW comparison design.",
            "The same fixed pretrained model scores are used; this evaluates transfer of an instability predictor, not transfer of face-recognition embeddings or recognition models.",
            "The 15 outer blocks and repeated splits are dependent; block metrics are descriptive, not independent-sample confidence intervals.",
        ],
        "outputs": [
            f"{PREFIX}_dataset.csv", f"{PREFIX}_split_audit.csv",
            f"{PREFIX}_regression_oof_predictions.csv", f"{PREFIX}_regression_metrics.csv",
            f"{PREFIX}_classification_oof_predictions.csv", f"{PREFIX}_classification_metrics.csv",
            f"{PREFIX}_feature_lift_comparison.csv", f"{PREFIX}_manifest.json",
        ],
    }
    with (OUTPUT_DIR / f"{PREFIX}_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nPROGRAM 16 COMPLETE")
    print(f"Predictive examples: {len(data):,} (expected {expected:,})")
    print(f"Outer blocks: {len(blocks)}")
    print("Pooled regression metrics:")
    print(reg_met_df[reg_met_df.block_id == "POOLED_OOF"].to_string(index=False))
    print("Pooled classification metrics:")
    print(cls_met_df[cls_met_df.block_id == "POOLED_OOF"].to_string(index=False))
    print("Saved outputs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
