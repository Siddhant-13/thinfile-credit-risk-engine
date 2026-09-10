"""
Gradient-boosted challengers to the scorecard.

Both XGBoost and LightGBM handle NaN natively, which suits this problem: a
thin-file applicant's missing bureau score stays missing and the tree learns
its own default direction for it. The explicit `*_missing` flags are kept as
well so the models can use missingness as a feature in its own right.

Imbalance (~12% default) is handled with scale_pos_weight / class_weight
rather than resampling. That improves ranking but deliberately distorts the
level of the predicted probabilities — a weighted model predicts as if
defaults were half the book. Ranking metrics (AUC, Gini, KS) are unaffected,
but expected loss is not, so `prior_correct` undoes the weighting before any
PD is multiplied by an exposure. Skipping that step is a common and expensive
mistake: it inflates portfolio EL by roughly the weight factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

XGB_GRID = {
    "n_estimators": [200, 300, 450, 600],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.02, 0.05, 0.08, 0.12],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 15],
    "reg_lambda": [1.0, 3.0, 10.0],
}

LGBM_GRID = {
    "n_estimators": [200, 300, 450, 600],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.02, 0.05, 0.08, 0.12],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_samples": [20, 50, 100],
    "reg_lambda": [0.0, 1.0, 5.0],
}


def scale_pos_weight(y) -> float:
    y = np.asarray(y)
    pos = max(int(y.sum()), 1)
    return float((len(y) - pos) / pos)


@dataclass
class TunedModel:
    name: str
    estimator: object
    params: dict
    cv_auc: float

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


def _tune(base, grid, X, y, n_iter, seed, name) -> TunedModel:
    search = RandomizedSearchCV(
        base,
        grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return TunedModel(name, search.best_estimator_, search.best_params_, float(search.best_score_))


def train_xgboost(X, y, n_iter: int = 15, seed: int = 1, overrides: dict | None = None) -> TunedModel:
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight(y),
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    if overrides:
        # Sidebar-pinned hyperparameters: fit directly, skip the search.
        base.set_params(**overrides)
        base.fit(X, y)
        return TunedModel("XGBoost", base, overrides, float("nan"))
    return _tune(base, XGB_GRID, X, y, n_iter, seed, "XGBoost")


def train_lightgbm(X, y, n_iter: int = 15, seed: int = 1, overrides: dict | None = None) -> TunedModel:
    base = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    if overrides:
        base.set_params(**overrides)
        base.fit(X, y)
        return TunedModel("LightGBM", base, overrides, float("nan"))
    return _tune(base, LGBM_GRID, X, y, n_iter, seed, "LightGBM")


def prior_correct(p: np.ndarray, weight: float) -> np.ndarray:
    """
    Undo class weighting so probabilities read as PDs again.

    Weighting the positive class by w multiplies the modelled odds by w, so
    dividing the odds back by w restores the population prior:

        odds_true = odds_weighted / w
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    odds = p / (1 - p) / weight
    return odds / (1 + odds)


def calibration_check(y_true, p) -> dict:
    """Cheap sanity check: does the average PD match the observed default rate?"""
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    return {
        "mean_predicted_pd": float(p.mean()),
        "observed_default_rate": float(y_true.mean()),
        "ratio": float(p.mean() / max(y_true.mean(), 1e-9)),
    }


def feature_importance(model: TunedModel, features: list[str]) -> pd.DataFrame:
    imp = getattr(model.estimator, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": features, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
