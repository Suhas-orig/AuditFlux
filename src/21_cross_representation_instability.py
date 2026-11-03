from __future__ import annotations

"""AuditFlux Program 21: Cross-representation geometry and verification-error analysis.

Question:
Can FaceNet-CASIA identity geometry help explain or predict identity-level
false-rejection behaviour in the original VGG16, ResNet50 and SENet50 scores?

Unit of analysis: identity (800 rows per model x threshold), never individual pairs.
Primary outcomes: identity-level FNR on genuine comparisons for the three
original architectures. FaceNet-derived scores/errors are deliberately NOT
used as outcomes or predictors in the cross-representation tests.

Validation: leave-one-original-BFW-fold-out. Geometry features are computed
from Program 20's independent FaceNet representation; group-only and
 group+geometry models are compared using out-of-fold identity predictions.

This is observational/exploratory. It does not establish causality.
"""

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import BFW_DATA_PATH, OUTPUT_DIR  # noqa: E402

PREFIX = "21_cross_representation"
MODELS = ["vgg16", "resnet50", "senet50"]
GROUPS = ["AF", "AM", "BF", "BM", "IF", "IM", "WF", "WM"]
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.05), 2).tolist()
SEED = 20261121


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def prepare_unique_genuine_pairs(df: pd.DataFrame):
    """Return one observation per distinct, non-self, unordered genuine image pair.

    BFW stores some genuine pairs in both orientations and also contains self-pairs.
    For 25 images per identity, the intended distinct genuine comparisons are
    C(25, 2) = 300. Self-pairs are not genuine verification trials and must be
    excluded; reversed duplicates must be canonicalised and counted once.
    """
    cols = ["fold", "p1", "p2", "label", "id1", "id2", "a1", *MODELS]
    genuine_rows = df.loc[df.label.eq(1), cols].copy()
    if not (genuine_rows.id1 == genuine_rows.id2).all():
        raise ValueError("A label=1 row has different endpoint identities.")

    original_rows = len(genuine_rows)
    self_mask = genuine_rows.p1.astype(str).eq(genuine_rows.p2.astype(str))
    self_rows = int(self_mask.sum())
    genuine_rows = genuine_rows.loc[~self_mask].copy()

    p1 = genuine_rows.p1.astype(str)
    p2 = genuine_rows.p2.astype(str)
    genuine_rows["pair_path_a"] = np.where(p1 <= p2, p1, p2)
    genuine_rows["pair_path_b"] = np.where(p1 <= p2, p2, p1)

    # Confirm reversed/duplicate rows refer to the same identity, fold and group,
    # and that the precomputed model scores agree before dropping duplicates.
    key = ["pair_path_a", "pair_path_b"]
    grouped = genuine_rows.groupby(key, sort=False)
    consistency_cols = ["id1", "fold", "a1", *MODELS]
    inconsistent = {}
    for col in consistency_cols:
        counts = grouped[col].nunique(dropna=False)
        inconsistent[col] = int((counts > 1).sum())
    if any(inconsistent.values()):
        raise ValueError(f"Canonical genuine-pair duplicates are inconsistent: {inconsistent}")

    deduplicated = genuine_rows.drop_duplicates(subset=key, keep="first").copy()
    duplicate_rows_removed = len(genuine_rows) - len(deduplicated)
    deduplicated = deduplicated.rename(columns={"id1": "identity", "a1": "group"})

    # Verify exact combinatorial coverage: 25 images imply 300 distinct
    # non-self unordered image pairs per identity.
    pair_counts = deduplicated.groupby("identity").size()
    image_counts = deduplicated.groupby("identity").apply(
        lambda part: len(set(part.p1.astype(str)) | set(part.p2.astype(str))),
        include_groups=False,
    )
    if len(pair_counts) != 800:
        raise ValueError(f"Expected 800 identities after deduplication; found {len(pair_counts)}.")
    if not (image_counts == 25).all():
        raise ValueError("Not every identity has exactly 25 unique image endpoints.")
    expected_pairs = 25 * 24 // 2
    if not (pair_counts == expected_pairs).all():
        bad = pair_counts[pair_counts != expected_pairs]
        raise ValueError(f"Expected exactly {expected_pairs} unique genuine pairs per identity; exceptions: {bad.to_dict()}")

    audit = {
        "genuine_rows_before_cleaning": int(original_rows),
        "self_comparison_rows_removed": self_rows,
        "nonself_rows_before_deduplication": int(len(genuine_rows)),
        "reversed_or_repeated_rows_removed": int(duplicate_rows_removed),
        "unique_nonself_genuine_pairs": int(len(deduplicated)),
        "identities_verified": int(len(pair_counts)),
        "images_per_identity_min": int(image_counts.min()),
        "images_per_identity_max": int(image_counts.max()),
        "unique_genuine_pairs_per_identity_min": int(pair_counts.min()),
        "unique_genuine_pairs_per_identity_max": int(pair_counts.max()),
        "duplicate_score_or_metadata_inconsistencies": inconsistent,
    }
    return deduplicated, audit


def metric_rows(df: pd.DataFrame, features: pd.DataFrame):
    genuine, pair_audit = prepare_unique_genuine_pairs(df)
    if not (genuine.groupby("identity").group.nunique().max() == 1):
        raise ValueError("An identity maps to multiple a1 groups among genuine pairs.")
    if not (genuine.groupby("identity").fold.nunique().max() == 1):
        raise ValueError("An identity maps to multiple folds among genuine pairs.")

    records = []
    for model in MODELS:
        score = genuine[model].to_numpy(dtype=float)
        for threshold in THRESHOLDS:
            part = genuine[["fold", "identity", "group"]].copy()
            part["false_reject"] = score < threshold
            agg = part.groupby(["fold", "identity", "group"], as_index=False).agg(
                identity_FNR=("false_reject", "mean"),
                genuine_pair_count=("false_reject", "size"),
                false_reject_count=("false_reject", "sum"),
            )
            agg["model"] = model
            agg["threshold"] = threshold
            records.append(agg)
    outcomes = pd.concat(records, ignore_index=True)
    outcomes = outcomes.merge(features, on=["identity", "fold", "group"], how="left", validate="many_to_one")
    if outcomes["within_identity_cosine_mean"].isna().any():
        missing = outcomes.loc[outcomes.within_identity_cosine_mean.isna(), "identity"].nunique()
        raise ValueError(f"Missing Program 20 compactness for {missing} identities.")
    if not outcomes.genuine_pair_count.eq(300).all():
        raise ValueError("Identity FNR denominators are not uniformly 300 after pair cleaning.")
    return outcomes, pair_audit


def within_group_associations(outcomes: pd.DataFrame):
    rows = []
    geometry_cols = [
        "within_identity_cosine_mean", "within_identity_cosine_median",
        "within_identity_cosine_sd", "within_identity_cosine_q10",
        "within_identity_cosine_q90", "source_aspect_ratio_mean",
        "source_aspect_ratio_sd",
    ]
    for (model, threshold), part in outcomes.groupby(["model", "threshold"]):
        # Pooled correlation is reported only as descriptive context.
        for feature in geometry_cols:
            valid = part[[feature, "identity_FNR"]].dropna()
            rho = spearmanr(valid[feature], valid.identity_FNR).statistic if len(valid) > 10 else np.nan
            rows.append({"model": model, "threshold": threshold, "scope": "pooled",
                         "feature": feature, "n_identities": len(valid), "spearman_rho": rho})
        # Remove group means from both variables: this asks whether geometry
        # tracks error variation among identities within the same demographic group.
        for feature in geometry_cols:
            x = part[feature] - part.groupby("group")[feature].transform("mean")
            y = part.identity_FNR - part.groupby("group").identity_FNR.transform("mean")
            valid = pd.DataFrame({"x": x, "y": y}).dropna()
            rho = spearmanr(valid.x, valid.y).statistic if len(valid) > 10 else np.nan
            rows.append({"model": model, "threshold": threshold, "scope": "within_group_demeaned",
                         "feature": feature, "n_identities": len(valid), "spearman_rho": rho})
    return pd.DataFrame(rows)


def fixed_effect_regression(outcomes: pd.DataFrame):
    # Descriptive OLS with group and fold fixed effects, HC3 robust standard errors.
    # One row per identity x model x threshold; identity is the observational unit.
    import statsmodels.formula.api as smf
    rows = []
    for (model, threshold), part in outcomes.groupby(["model", "threshold"]):
        for feature in ["within_identity_cosine_mean", "within_identity_cosine_sd",
                        "within_identity_cosine_q10", "source_aspect_ratio_sd"]:
            clean = part[["identity_FNR", feature, "group", "fold"]].dropna().copy()
            if clean[feature].nunique() < 2:
                continue
            fit = smf.ols(f"identity_FNR ~ {feature} + C(group) + C(fold)", data=clean).fit(cov_type="HC3")
            term = fit.params.get(feature, np.nan)
            rows.append({"model": model, "threshold": threshold, "feature": feature,
                         "n_identities": len(clean), "coefficient": term,
                         "robust_se_HC3": fit.bse.get(feature, np.nan),
                         "p_value_descriptive": fit.pvalues.get(feature, np.nan),
                         "r_squared": fit.rsquared})
    result = pd.DataFrame(rows)
    if not result.empty:
        # Control family-wise error across all prespecified model x threshold x
        # feature coefficient tests. These remain exploratory fixed-effect tests.
        from statsmodels.stats.multitest import multipletests
        valid = result.p_value_descriptive.notna()
        result["p_value_holm_global"] = np.nan
        if valid.any():
            result.loc[valid, "p_value_holm_global"] = multipletests(
                result.loc[valid, "p_value_descriptive"].to_numpy(),
                method="holm",
            )[1]
    return result


def build_oof_predictions(outcomes: pd.DataFrame):
    # Compare a demographic-only baseline to demographic + FaceNet geometry.
    # Each original fold is fully held out. All transformations fit on training folds.
    geometry = [
        "within_identity_cosine_mean", "within_identity_cosine_median",
        "within_identity_cosine_sd", "within_identity_cosine_q10",
        "within_identity_cosine_q90", "source_aspect_ratio_mean",
        "source_aspect_ratio_sd",
    ]
    pred_rows, metric_rows = [], []
    for (model, threshold), part in outcomes.groupby(["model", "threshold"]):
        part = part.sort_values(["fold", "identity"]).reset_index(drop=True)
        oof = {"group_only": np.full(len(part), np.nan), "group_plus_geometry": np.full(len(part), np.nan)}
        for heldout in sorted(part.fold.unique()):
            train_idx = np.flatnonzero(part.fold.to_numpy() != heldout)
            test_idx = np.flatnonzero(part.fold.to_numpy() == heldout)
            y_train = part.loc[train_idx, "identity_FNR"].to_numpy(float)
            for config in oof:
                numeric = [] if config == "group_only" else geometry
                transformers = [("group", OneHotEncoder(handle_unknown="ignore"), ["group"])]
                if numeric:
                    transformers.append(("numeric", Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]), numeric))
                prep = ColumnTransformer(transformers, remainder="drop")
                pipe = Pipeline([("prep", prep), ("ridge", Ridge(alpha=10.0))])
                pipe.fit(part.iloc[train_idx], y_train)
                oof[config][test_idx] = pipe.predict(part.iloc[test_idx])
        y = part.identity_FNR.to_numpy(float)
        for config, p in oof.items():
            residual = y - p
            metric_rows.append({
                "model": model, "threshold": threshold, "feature_set": config,
                "n_identities": len(y), "mae": mean_absolute_error(y, p),
                "rmse": mean_squared_error(y, p) ** 0.5,
                "r2": r2_score(y, p),
                "spearman_rho": spearmanr(y, p).statistic,
            })
            for i, row in part.iterrows():
                pred_rows.append({"model": model, "threshold": threshold,
                                  "feature_set": config, "identity": int(row.identity),
                                  "fold": int(row.fold), "group": row.group,
                                  "actual_identity_FNR": float(y[i]),
                                  "predicted_identity_FNR": float(p[i]),
                                  "absolute_error": float(abs(residual[i]))})
    return pd.DataFrame(pred_rows), pd.DataFrame(metric_rows)


def main():
    start = time.time()
    data_path = Path(BFW_DATA_PATH)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    compact_path = out_dir / "20_representation_space_identity_compactness.csv"
    image_path = out_dir / "20_representation_space_image_metadata.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"BFW table not found: {data_path}")
    if not compact_path.exists() or not image_path.exists():
        raise FileNotFoundError("Program 20 identity_compactness and image_metadata outputs are required in OUTPUT_DIR.")

    print(f"Loading BFW table: {data_path}")
    df = pd.read_csv(data_path)
    required = {"fold", "label", "id1", "id2", "a1", *MODELS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"BFW table is missing columns: {sorted(missing)}")
    if len(df) != 923898:
        warnings.warn(f"Expected the previously audited 923,898 rows; found {len(df):,}.")

    compact = pd.read_csv(compact_path)
    images = pd.read_csv(image_path)
    compact = compact.rename(columns={"group": "group"})
    aspect = images.groupby(["identity", "fold", "group"], as_index=False).agg(
        source_aspect_ratio_mean=("source_aspect_ratio", "mean"),
        source_aspect_ratio_sd=("source_aspect_ratio", "std"),
        source_width_mean=("source_width", "mean"),
        source_height_mean=("source_height", "mean"),
    )
    features = compact.merge(aspect, on=["identity", "fold", "group"], how="left", validate="one_to_one")
    if features.identity.nunique() != 800 or len(features) != 800:
        raise ValueError(f"Expected 800 unique identity feature rows; got {len(features)} rows / {features.identity.nunique()} identities.")

    print("Aggregating original-model genuine-pair errors to identity level...")
    outcomes, pair_audit = metric_rows(df, features)
    print(f"Outcome records: {len(outcomes):,}")
    print(f"Unique genuine-pair audit: {pair_audit}")

    print("Computing pooled and within-group associations...")
    associations = within_group_associations(outcomes)
    print("Fitting group/fold-adjusted descriptive regressions...")
    regressions = fixed_effect_regression(outcomes)
    print("Running leave-one-original-fold-out prediction...")
    predictions, prediction_metrics = build_oof_predictions(outcomes)

    # Save outputs.
    outputs = {
        "identity_outcomes": f"{PREFIX}_identity_outcomes.csv",
        "associations": f"{PREFIX}_associations.csv",
        "adjusted_regressions": f"{PREFIX}_adjusted_regressions.csv",
        "oof_predictions": f"{PREFIX}_oof_predictions.csv",
        "prediction_metrics": f"{PREFIX}_prediction_metrics.csv",
    }
    outcomes.to_csv(out_dir / outputs["identity_outcomes"], index=False)
    associations.to_csv(out_dir / outputs["associations"], index=False)
    regressions.to_csv(out_dir / outputs["adjusted_regressions"], index=False)
    predictions.to_csv(out_dir / outputs["oof_predictions"], index=False)
    prediction_metrics.to_csv(out_dir / outputs["prediction_metrics"], index=False)

    manifest = {
        "experiment": "Program 21 - Cross-Representation Geometry and Verification-Error Analysis",
        "source_bfw_csv": str(data_path),
        "source_bfw_rows": int(len(df)),
        "source_bfw_sha256": sha256_file(data_path),
        "program20_identity_compactness_rows": int(len(compact)),
        "program20_image_metadata_rows": int(len(images)),
        "identity_count": int(features.identity.nunique()),
        "unit_of_analysis": "identity-level false-rejection rate over 300 unique non-self unordered genuine image pairs",
        "genuine_pair_cleaning_audit": pair_audit,
        "primary_outcomes": [f"identity_FNR_{m}" for m in MODELS],
        "predictor_representation": "InceptionResnetV1-CASIA-WebFace 512-D embeddings summarized as identity compactness; no FaceNet verification error used as a predictor or outcome",
        "models": MODELS,
        "thresholds": THRESHOLDS,
        "group_convention": "a1 first endpoint, consistent with AuditFlux Programs 1-20",
        "association_analysis": "pooled and within-demographic-group demeaned Spearman correlations",
        "adjusted_analysis": "identity-level OLS with demographic-group and original-fold fixed effects; HC3 robust standard errors; Holm correction across all 108 model x threshold x feature tests; descriptive, not causal",
        "predictive_validation": "leave-one-original-BFW-fold-out; group-only baseline compared with group plus FaceNet geometry; preprocessing fit on training folds only",
        "predictor_features": ["within_identity_cosine_mean", "within_identity_cosine_median", "within_identity_cosine_sd", "within_identity_cosine_q10", "within_identity_cosine_q90", "source_aspect_ratio_mean", "source_aspect_ratio_sd"],
        "prediction_model": "Ridge regression alpha=10",
        "limitations": [
            "Only five original BFW folds are available; held-out-fold performance is exploratory.",
            "FaceNet compactness and original-model FNR are measured on the same BFW identities; this is cross-representation association, not independent-dataset validation.",
            "FaceNet compactness is derived from genuine-image similarities and is not used to predict FaceNet's own FNR.",
            "Within-group demeaned correlations and group/fold fixed effects reduce, but do not eliminate, confounding.",
            "Each identity contributes exactly 300 unique non-self unordered genuine image pairs after canonicalizing pair orientation, removing self-comparisons, and deduplicating repeated rows.",
            "Identity-level FNR estimates have equal denominators but remain dependent because image pairs share images and identities.",
            "Adjusted-regression p-values are exploratory and Holm-adjusted across the 108 prespecified model x threshold x feature tests; associations do not establish causal mechanisms.",
            "The analysis focuses on false rejections; false-positive identity attribution is dyadic and is not treated as an independent identity-level outcome here.",
            "The representation is one FaceNet-CASIA model and does not establish generality across all representation architectures."
        ],
        "outputs": outputs,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with (out_dir / f"{PREFIX}_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nPROGRAM 21 COMPLETE")
    print(f"Identity outcome rows: {len(outcomes):,}")
    print(f"Association rows: {len(associations):,}")
    print(f"Adjusted regression rows: {len(regressions):,}")
    print(f"OOF prediction rows: {len(predictions):,}")
    print(f"Prediction metric rows: {len(prediction_metrics):,}")
    print(f"Outputs: {out_dir}")
    print(f"Elapsed: {manifest['elapsed_seconds']} seconds")


if __name__ == "__main__":
    main()
