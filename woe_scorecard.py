"""
Weight-of-Evidence binning, Information Value selection and a points-based
logistic scorecard.

Conventions used throughout (the ones a credit risk team would recognise):

    WOE_i = ln( (goods_i / goods_total) / (bads_i / bads_total) )
    IV    = sum_i (goods_i/goods_total - bads_i/bads_total) * WOE_i

"bad" is default = 1. So a **higher WOE means a safer bin**, and the fitted
logistic coefficients on WOE come out negative. Points are scaled with the
usual PDO transform, so a **higher score means a lower probability of
default**.

Missing values get their own bin. They are never imputed — for a thin-file
applicant the absence of a bureau record is the single most informative thing
on the application.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

MISSING_LABEL = "Missing"


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------

def _tree_edges(x: pd.Series, y: pd.Series, max_bins: int, min_bin_frac: float) -> list[float]:
    """Supervised split points from a shallow tree — bins track the target."""
    mask = x.notna()
    if mask.sum() < 50 or y[mask].nunique() < 2:
        return []
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=max(int(min_bin_frac * len(x)), 30),
        random_state=0,
    )
    tree.fit(x[mask].to_frame(), y[mask])
    edges = sorted(t for t in tree.tree_.threshold if t != -2)
    return [float(round(e, 4)) for e in edges]


def _cut(x: pd.Series, edges: list[float]) -> pd.Series:
    """Map a column onto its bin labels, with NaN routed to its own bin."""
    if not edges:
        labels = pd.Series(np.where(x.notna(), "All", MISSING_LABEL), index=x.index)
        return labels
    bounds = [-np.inf] + list(edges) + [np.inf]
    binned = pd.cut(x, bins=bounds, right=True, duplicates="drop")
    out = binned.astype(str)
    out[x.isna()] = MISSING_LABEL
    return out


@dataclass
class WOEBinner:
    """Fits bins + WOE tables for every column handed to it."""

    max_bins: int = 6
    min_bin_frac: float = 0.05
    smoothing: float = 0.5

    edges_: dict[str, list[float]] = field(default_factory=dict)
    tables_: dict[str, pd.DataFrame] = field(default_factory=dict)
    iv_: dict[str, float] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEBinner":
        y = pd.Series(y).astype(int)
        total_bad = max(int(y.sum()), 1)
        total_good = max(int((1 - y).sum()), 1)

        for col in X.columns:
            x = X[col]
            edges = _tree_edges(x, y, self.max_bins, self.min_bin_frac)
            self.edges_[col] = edges
            bins = _cut(x, edges)

            grp = pd.DataFrame({"bin": bins, "y": y.values}).groupby("bin", observed=True)["y"]
            table = pd.DataFrame({"count": grp.size(), "bad": grp.sum()})
            table["good"] = table["count"] - table["bad"]
            table["bad_rate"] = table["bad"] / table["count"]
            table["dist_bad"] = (table["bad"] + self.smoothing) / (total_bad + self.smoothing * len(table))
            table["dist_good"] = (table["good"] + self.smoothing) / (total_good + self.smoothing * len(table))
            table["woe"] = np.log(table["dist_good"] / table["dist_bad"])
            table["iv"] = (table["dist_good"] - table["dist_bad"]) * table["woe"]
            table["population_pct"] = table["count"] / len(x)

            table = table.reset_index().rename(columns={"bin": "bucket"})
            table.insert(0, "feature", col)
            self.tables_[col] = table
            self.iv_[col] = float(table["iv"].sum())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in X.columns:
            if col not in self.tables_:
                continue
            bins = _cut(X[col], self.edges_[col])
            mapping = self.tables_[col].set_index("bucket")["woe"]
            out[f"woe_{col}"] = bins.map(mapping).astype(float).fillna(0.0)
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": f, "iv": v, "strength": iv_strength(v)} for f, v in self.iv_.items()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def woe_table(self, features: list[str] | None = None) -> pd.DataFrame:
        keys = features or list(self.tables_)
        return pd.concat([self.tables_[k] for k in keys], ignore_index=True)


def iv_strength(iv: float) -> str:
    """Siddiqi's rule of thumb, used everywhere in scorecard practice."""
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "suspiciously strong"


# --------------------------------------------------------------------------
# Scorecard
# --------------------------------------------------------------------------

@dataclass
class Scorecard:
    """Logistic regression on WOE, published as a points table."""

    iv_threshold: float = 0.02
    max_bins: int = 6
    min_bin_frac: float = 0.05
    pdo: int = 20            # points to double the odds
    base_score: int = 600
    base_odds: float = 50.0  # good:bad odds at base_score
    C: float = 1.0

    binner_: WOEBinner | None = None
    selected_: list[str] = field(default_factory=list)
    model_: LogisticRegression | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Scorecard":
        self.binner_ = WOEBinner(self.max_bins, self.min_bin_frac).fit(X, y)
        self.selected_ = [f for f, iv in self.binner_.iv_.items() if iv >= self.iv_threshold]
        if not self.selected_:
            # Nothing clears the bar — keep the single best feature so the
            # model still runs and the weakness shows up in the metrics.
            self.selected_ = [max(self.binner_.iv_, key=self.binner_.iv_.get)]

        woe = self.binner_.transform(X[self.selected_])
        self.model_ = LogisticRegression(C=self.C, max_iter=1000)
        self.model_.fit(woe, y)

        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)
        return self

    # --------------------------------------------------------------- scoring
    def _woe(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.binner_.transform(X[self.selected_])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._woe(X))[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Points. Higher = safer."""
        woe = self._woe(X)
        n = len(self.selected_)
        alpha = float(self.model_.intercept_[0])
        contrib = np.zeros(len(X))
        for beta, col in zip(self.model_.coef_[0], woe.columns):
            contrib += -(woe[col].values * beta + alpha / n) * self.factor + self.offset / n
        return np.round(contrib, 0)

    # ------------------------------------------------------------ publishing
    def scorecard_table(self) -> pd.DataFrame:
        """The human-readable artefact: feature, bucket, points."""
        n = len(self.selected_)
        alpha = float(self.model_.intercept_[0])
        betas = dict(zip(self.selected_, self.model_.coef_[0]))

        frames = []
        for feat in self.selected_:
            t = self.binner_.tables_[feat].copy()
            t["points"] = np.round(
                -(t["woe"] * betas[feat] + alpha / n) * self.factor + self.offset / n, 0
            )
            t["coefficient"] = betas[feat]
            frames.append(
                t[["feature", "bucket", "count", "population_pct", "bad_rate",
                   "woe", "iv", "coefficient", "points"]]
            )
        return pd.concat(frames, ignore_index=True)

    def explain(self, applicant: dict) -> tuple[pd.DataFrame, float, float]:
        """Score one hypothetical applicant and show where every point came from."""
        row = pd.DataFrame([applicant])
        for col in self.selected_:
            if col not in row:
                row[col] = np.nan

        table = self.scorecard_table()
        n = len(self.selected_)
        alpha = float(self.model_.intercept_[0])
        betas = dict(zip(self.selected_, self.model_.coef_[0]))

        rows = []
        for feat in self.selected_:
            bucket = _cut(row[feat], self.binner_.edges_[feat]).iloc[0]
            match = table[(table["feature"] == feat) & (table["bucket"] == bucket)]
            woe = float(match["woe"].iloc[0]) if len(match) else 0.0
            pts = -(woe * betas[feat] + alpha / n) * self.factor + self.offset / n
            rows.append(
                {
                    "feature": feat,
                    "value": applicant.get(feat, np.nan),
                    "bucket": bucket,
                    "woe": round(woe, 4),
                    "points": round(pts, 0),
                }
            )

        breakdown = pd.DataFrame(rows)
        total = float(breakdown["points"].sum())
        pd_est = float(self.predict_proba(row)[0])
        return breakdown, total, pd_est

    def points_range(self) -> pd.DataFrame:
        """Min/max points per feature — how much each one can actually swing."""
        t = self.scorecard_table()
        g = t.groupby("feature")["points"]
        out = pd.DataFrame({"min_points": g.min(), "max_points": g.max()})
        out["swing"] = out["max_points"] - out["min_points"]
        return out.sort_values("swing", ascending=False).reset_index()


if __name__ == "__main__":
    from data_generator import generate_population, prepare
    from sklearn.model_selection import train_test_split

    pop = generate_population()
    pop, sets = prepare(pop)
    X, y = pop[sets["augmented"]], pop["default"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)

    card = Scorecard().fit(Xtr, ytr)
    print(card.binner_.iv_table().to_string(index=False))
    print("\nselected:", card.selected_)
    print("\n", card.points_range().to_string(index=False))
    print("\nscore range:", card.score(Xte).min(), "-", card.score(Xte).max())
