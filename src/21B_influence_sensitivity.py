"""
Program 21B — Influence and Robust-Regression Sensitivity
Uses Program 21 identity-level outcomes; does not alter Program 21 outputs.
Primary sensitivity: threshold 0.50, three core compactness features.
Secondary sensitivity: Huber-vs-OLS coefficient signs across all thresholds.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from config import OUTPUT_DIR

INPUT = Path(OUTPUT_DIR) / "21_cross_representation_identity_outcomes.csv"
OUT_DIR = Path(OUTPUT_DIR)
PRIMARY_THRESHOLD = 0.50
FEATURES = [
    "within_identity_cosine_mean",
    "within_identity_cosine_sd",
    "within_identity_cosine_q10",
]
MODELS = ["vgg16", "resnet50", "senet50"]
TOP_K = 10

def main():
    d_all = pd.read_csv(INPUT)
    required = {"model", "threshold", "identity", "fold", "group",
                "identity_FNR", *FEATURES}
    missing = required - set(d_all.columns)
    if missing:
        raise ValueError(f"Program 21 outcome file missing columns: {sorted(missing)}")

    primary_rows, loo_rows, all_threshold_rows = [], [], []

    for model in MODELS:
        for threshold in sorted(d_all["threshold"].unique()):
            d = d_all[(d_all["model"] == model) &
                      (d_all["threshold"] == threshold)].copy()
            if len(d) != 800 or d["identity"].nunique() != 800:
                raise ValueError(
                    f"Expected 800 unique identities for {model}, {threshold}; "
                    f"found {len(d)} rows / {d['identity'].nunique()} identities."
                )
            for feature in FEATURES:
                formula = f"identity_FNR ~ {feature} + C(group) + C(fold)"
                ols = smf.ols(formula, data=d).fit()
                huber = smf.rlm(
                    formula, data=d, M=sm.robust.norms.HuberT()
                ).fit()
                oc, hc = float(ols.params[feature]), float(huber.params[feature])
                all_threshold_rows.append({
                    "model": model, "threshold": threshold, "feature": feature,
                    "ols_coefficient": oc, "huber_coefficient": hc,
                    "same_sign": bool(np.sign(oc) == np.sign(hc)),
                    "relative_coefficient_change": abs(hc - oc) /
                        max(abs(oc), 1e-12),
                })

                if np.isclose(threshold, PRIMARY_THRESHOLD):
                    influence = ols.get_influence()
                    feature_idx = list(ols.params.index).index(feature)
                    dfbeta = influence.dfbetas[:, feature_idx]
                    cooks = influence.cooks_distance[0]
                    robust_hc3 = ols.get_robustcov_results(cov_type="HC3")
                    hc3_se = float(robust_hc3.bse[feature_idx])
                    primary_rows.append({
                        "model": model, "threshold": threshold, "feature": feature,
                        "n_identities": int(ols.nobs),
                        "ols_coefficient": oc, "hc3_se": hc3_se,
                        "huber_coefficient": hc, "same_sign_ols_huber":
                            bool(np.sign(oc) == np.sign(hc)),
                        "huber_relative_change": abs(hc - oc) /
                            max(abs(oc), 1e-12),
                        "max_abs_feature_DFBETA": float(np.max(np.abs(dfbeta))),
                        "max_cooks_distance": float(np.max(cooks)),
                        "count_cooks_gt_4_over_n": int(np.sum(cooks > 4 / len(d))),
                    })
                    top = np.argsort(np.abs(dfbeta))[::-1][:TOP_K]
                    for j in top:
                        reduced = d.drop(d.index[j])
                        loo = smf.ols(formula, data=reduced).fit()
                        lc = float(loo.params[feature])
                        loo_rows.append({
                            "model": model, "threshold": threshold,
                            "feature": feature,
                            "identity": d.iloc[j]["identity"],
                            "group": d.iloc[j]["group"],
                            "fold": d.iloc[j]["fold"],
                            "feature_DFBETA": float(dfbeta[j]),
                            "cooks_distance": float(cooks[j]),
                            "full_ols_coefficient": oc,
                            "leave_one_identity_out_coefficient": lc,
                            "absolute_coefficient_change": abs(lc - oc),
                            "relative_coefficient_change": abs(lc - oc) /
                                max(abs(oc), 1e-12),
                            "sign_reversed": bool(np.sign(lc) != np.sign(oc)),
                        })

    primary = pd.DataFrame(primary_rows)
    loo = pd.DataFrame(loo_rows)
    all_threshold = pd.DataFrame(all_threshold_rows)

    # Add concise top-10 LOO ranges to the primary summary.
    loo_summary = loo.groupby(["model", "feature"]).agg(
        top10_loo_min=("leave_one_identity_out_coefficient", "min"),
        top10_loo_max=("leave_one_identity_out_coefficient", "max"),
        top10_loo_sign_reversals=("sign_reversed", "sum"),
        top10_max_relative_change=("relative_coefficient_change", "max"),
    ).reset_index()
    primary = primary.merge(loo_summary, on=["model", "feature"], how="left")

    primary.to_csv(OUT_DIR / "21B_influence_primary_threshold.csv", index=False)
    loo.to_csv(OUT_DIR / "21B_top10_leave_one_identity_out.csv", index=False)
    all_threshold.to_csv(OUT_DIR / "21B_huber_all_thresholds.csv", index=False)

    print("Program 21B complete.")
    print(f"Input: {INPUT}")
    print(f"Primary diagnostics: {len(primary)} rows")
    print(f"Top-influence leave-one-out refits: {len(loo)} rows")
    print(f"All-threshold OLS/Huber comparisons: {len(all_threshold)} rows")
    print("OLS/Huber sign agreement:",
          f"{int(all_threshold['same_sign'].sum())}/{len(all_threshold)}")
    print("Outputs saved in:", OUT_DIR)

if __name__ == "__main__":
    main()
