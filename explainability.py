"""
SHAP values for the boosted models, and the explicit trade-off view against
the scorecard.

The argument the project is making: the scorecard is auditable by anyone —
you can print it on a page and a regulator can trace every point. SHAP gives
you attribution for the boosted model too, but it is a *post-hoc local
approximation*, not the decision rule itself, and it can't be handed to a
customer as a reason code without further work. The comparison below puts a
number on what that difference buys in AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def shap_values(model, X: pd.DataFrame, max_rows: int = 1_500, seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """TreeSHAP on a sample. Returns (values, the rows they correspond to)."""
    estimator = getattr(model, "estimator", model)
    sample = X.sample(min(max_rows, len(X)), random_state=seed) if len(X) > max_rows else X
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(sample)
    if isinstance(values, list):          # some versions return one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                  # (rows, features, classes)
        values = values[:, :, -1]
    return values, sample


def shap_importance(values: np.ndarray, features: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature — the global ranking."""
    mean_abs = np.abs(values).mean(axis=0)
    signed = values.mean(axis=0)
    return (
        pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs, "mean_shap": signed})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def shap_by_segment(values: np.ndarray, sample: pd.DataFrame, segments: pd.Series) -> pd.DataFrame:
    """Which features drive the model for each borrower segment separately."""
    seg = segments.loc[sample.index]
    frames = []
    for name in sorted(seg.unique()):
        mask = (seg == name).values
        if mask.sum() == 0:
            continue
        imp = shap_importance(values[mask], list(sample.columns))
        imp["segment"] = name
        imp["share_of_attribution"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
        frames.append(imp)
    return pd.concat(frames, ignore_index=True)


def explain_applicant(values: np.ndarray, sample: pd.DataFrame, row: int = 0) -> pd.DataFrame:
    """Local attribution for one scored applicant."""
    return (
        pd.DataFrame(
            {
                "feature": sample.columns,
                "value": sample.iloc[row].values,
                "shap": values[row],
            }
        )
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .drop(columns="abs_shap")
        .reset_index(drop=True)
    )


def tradeoff_table(scorecard_auc: float, ml_aucs: dict[str, float], n_scorecard_rules: int,
                   n_ml_trees: int) -> pd.DataFrame:
    """One table: what each model costs in transparency and returns in AUC."""
    rows = [
        {
            "model": "WOE scorecard",
            "auc": scorecard_auc,
            "auc_uplift_vs_scorecard": 0.0,
            "decision_rule": f"{n_scorecard_rules} published point buckets",
            "auditable_by_hand": "yes",
            "reason_codes": "direct — the bucket is the reason",
            "regulatory_position": "accepted under most model-governance regimes",
        }
    ]
    for name, value in ml_aucs.items():
        rows.append(
            {
                "model": name,
                "auc": value,
                "auc_uplift_vs_scorecard": value - scorecard_auc,
                "decision_rule": f"~{n_ml_trees} boosted trees",
                "auditable_by_hand": "no",
                "reason_codes": "post-hoc via SHAP, needs separate validation",
                "regulatory_position": "requires additional model-risk documentation",
            }
        )
    return pd.DataFrame(rows)


def rank_agreement(scorecard_points: pd.DataFrame, shap_imp: pd.DataFrame) -> pd.DataFrame:
    """
    Do the two models agree on what matters? Scorecard influence is measured by
    points swing (max minus min points a feature can contribute).
    """
    a = scorecard_points.rename(columns={"swing": "scorecard_swing"})[["feature", "scorecard_swing"]]
    b = shap_imp[["feature", "mean_abs_shap"]]
    merged = a.merge(b, on="feature", how="outer")
    merged["scorecard_rank"] = merged["scorecard_swing"].rank(ascending=False)
    merged["shap_rank"] = merged["mean_abs_shap"].rank(ascending=False)
    merged["rank_gap"] = (merged["scorecard_rank"] - merged["shap_rank"]).abs()
    return merged.sort_values("shap_rank").reset_index(drop=True)
