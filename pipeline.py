"""
One place that runs the whole study end to end, so `app.py` stays a view layer
and the same results can be reproduced from the command line.

Train/test split is stratified on default and kept identical across every
model, so the segment comparison is like-for-like.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import expected_loss as el
import explainability as xai
import ml_models
import validation as val
from data_generator import DEFAULT_LGD, generate_population, prepare
from woe_scorecard import Scorecard


@dataclass(frozen=True)
class Config:
    n_applicants: int = 8_000
    thin_file_share: float = 0.40
    seed: int = 7
    test_size: float = 0.30
    iv_threshold: float = 0.02
    max_bins: int = 6
    pdo: int = 20
    base_score: int = 600
    search_iterations: int = 15
    xgb_overrides: tuple = ()      # tuple of (param, value) pairs, hashable for caching
    lgd: float = DEFAULT_LGD
    n_tiers: int = 5

    def overrides(self) -> dict | None:
        return dict(self.xgb_overrides) or None


def run(config: Config) -> dict:
    pop = generate_population(
        n=config.n_applicants,
        thin_file_share=config.thin_file_share,
        seed=config.seed,
    )
    pop, sets = prepare(pop)

    train, test = train_test_split(
        pop, test_size=config.test_size, stratify=pop["default"], random_state=config.seed
    )
    y_tr, y_te = train["default"], test["default"]

    # ------------------------------------------------------------ scorecards
    cards, scores = {}, {}
    for name, cols in sets.items():
        card = Scorecard(
            iv_threshold=config.iv_threshold,
            max_bins=config.max_bins,
            pdo=config.pdo,
            base_score=config.base_score,
        ).fit(train[cols], y_tr)
        cards[name] = card
        scores[f"Scorecard ({name})"] = card.predict_proba(test[cols])

    # ------------------------------------------------------------ ml models
    aug = sets["augmented"]
    xgb = ml_models.train_xgboost(
        train[aug], y_tr, n_iter=config.search_iterations,
        seed=config.seed, overrides=config.overrides(),
    )
    lgbm = ml_models.train_lightgbm(
        train[aug], y_tr, n_iter=config.search_iterations, seed=config.seed
    )
    trad_xgb = ml_models.train_xgboost(
        train[sets["traditional"]], y_tr, n_iter=max(config.search_iterations // 2, 5),
        seed=config.seed,
    )

    scores["XGBoost (augmented)"] = xgb.predict_proba(test[aug])
    scores["LightGBM (augmented)"] = lgbm.predict_proba(test[aug])
    scores["XGBoost (traditional)"] = trad_xgb.predict_proba(test[sets["traditional"]])

    # ----------------------------------------------------------- validation
    overall = val.compare_models({k: (y_te, v) for k, v in scores.items()})
    by_segment = val.segment_report(y_te, scores, test["segment"])
    roc = {k: val.roc_points(y_te, v) for k, v in scores.items()}

    best_name = overall.iloc[0]["model"]
    best_scores = scores[best_name]

    # ------------------------------------------------------------ stability
    v1 = test["batch"] == "vintage_1"
    psi_value, psi_detail = val.psi(
        cards["augmented"].score(test.loc[v1, aug]),
        cards["augmented"].score(test.loc[~v1, aug]),
    )

    feature_psi = []
    for col in aug:
        if test[col].notna().sum() < 100:
            continue
        a = test.loc[v1, col].dropna()
        b = test.loc[~v1, col].dropna()
        if len(a) < 50 or len(b) < 50 or a.nunique() < 3:
            continue
        value, _ = val.psi(a.values, b.values, bins=8)
        feature_psi.append({"feature": col, "psi": value, "verdict": val.psi_verdict(value)})
    feature_psi = pd.DataFrame(feature_psi).sort_values("psi", ascending=False)

    # -------------------------------------------------------- explainability
    shap_vals, shap_sample = xai.shap_values(xgb, test[aug])
    shap_imp = xai.shap_importance(shap_vals, aug)
    shap_seg = xai.shap_by_segment(shap_vals, shap_sample, test["segment"])

    card_auc = float(overall.loc[overall["model"] == "Scorecard (augmented)", "auc"].iloc[0])
    tradeoff = xai.tradeoff_table(
        card_auc,
        {
            "XGBoost (augmented)": float(overall.loc[overall["model"] == "XGBoost (augmented)", "auc"].iloc[0]),
            "LightGBM (augmented)": float(overall.loc[overall["model"] == "LightGBM (augmented)", "auc"].iloc[0]),
        },
        n_scorecard_rules=len(cards["augmented"].scorecard_table()),
        n_ml_trees=xgb.estimator.get_params().get("n_estimators", 0),
    )
    agreement = xai.rank_agreement(cards["augmented"].points_range(), shap_imp)

    # -------------------------------------------------------- expected loss
    # Ranking is unaffected by class weighting, the level is not. Undo the
    # weighting before any PD gets multiplied by an exposure.
    weight = ml_models.scale_pos_weight(y_tr)
    if best_name.startswith(("XGBoost", "LightGBM")):
        pd_for_el = ml_models.prior_correct(best_scores, weight)
    else:
        pd_for_el = best_scores
    calibration = ml_models.calibration_check(y_te, pd_for_el)
    uncalibrated = ml_models.calibration_check(y_te, best_scores)

    tiers = el.expected_loss_table(test, pd_for_el, lgd=config.lgd, n_tiers=config.n_tiers)
    portfolio = el.portfolio_summary(tiers)
    el_segment = el.expected_loss_by_segment(test, pd_for_el, lgd=config.lgd)
    cutoffs = el.cutoff_analysis(test, pd_for_el, lgd=config.lgd)

    return {
        "config": asdict(config),
        "population": pop,
        "train": train,
        "test": test,
        "feature_sets": sets,
        "cards": cards,
        "models": {"xgboost": xgb, "lightgbm": lgbm, "xgboost_traditional": trad_xgb},
        "scores": scores,
        "overall_metrics": overall,
        "segment_metrics": by_segment,
        "roc": roc,
        "psi": {"score_psi": psi_value, "detail": psi_detail, "features": feature_psi},
        "shap": {"values": shap_vals, "sample": shap_sample, "importance": shap_imp, "by_segment": shap_seg},
        "tradeoff": tradeoff,
        "rank_agreement": agreement,
        "best_model": best_name,
        "best_scores": best_scores,
        "pd_for_el": pd_for_el,
        "calibration": {"calibrated": calibration, "raw": uncalibrated, "weight": weight},
        "tiers": tiers,
        "portfolio": portfolio,
        "el_segment": el_segment,
        "cutoffs": cutoffs,
    }


def headline(results: dict) -> pd.DataFrame:
    """The one table worth putting in a summary email."""
    seg = results["segment_metrics"]
    keep = ["Scorecard (traditional)", "Scorecard (augmented)",
            "XGBoost (traditional)", "XGBoost (augmented)"]
    out = seg[seg["model"].isin(keep)].pivot(index="model", columns="segment", values="auc")
    return out.reindex(keep).round(3)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    res = run(Config())
    print("\n=== AUC by model and segment ===")
    print(headline(res).to_string())
    print("\n=== Overall ===")
    print(res["overall_metrics"][["model", "auc", "gini", "ks", "recall_default"]].round(3).to_string(index=False))
    print("\n=== Score PSI (vintage 1 vs 2) ===")
    print(round(res["psi"]["score_psi"], 4), val.psi_verdict(res["psi"]["score_psi"]))
    print("\n=== Top SHAP features ===")
    print(res["shap"]["importance"].head(8).round(4).to_string(index=False))
    print("\n=== Risk tiers ===")
    print(res["tiers"][["tier", "borrowers", "avg_pd", "total_ead", "expected_loss",
                        "actual_default_rate"]].round(3).to_string(index=False))
    print("\n=== Calibration ===")
    print(res["calibration"])
    print("\n=== Portfolio ===")
    print(res["portfolio"])
