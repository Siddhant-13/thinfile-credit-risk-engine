"""
From a probability to a number the business actually books.

    EL = PD x LGD x EAD

PD comes from the model. LGD is a fixed illustrative assumption (60% is the
usual placeholder for unsecured consumer lending) — it is *not* modelled here,
and the app says so on the tab, because a made-up LGD dressed up as an
estimate is worse than an honest assumption. EAD is the sanctioned limit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIER_NAMES = {
    3: ["Low", "Medium", "High"],
    4: ["A", "B", "C", "D"],
    5: ["A - lowest risk", "B", "C", "D", "E - highest risk"],
    6: ["A", "B", "C", "D", "E", "F"],
    7: ["A", "B", "C", "D", "E", "F", "G"],
}


def assign_tiers(pd_estimates, n_tiers: int = 5, index=None) -> pd.Series:
    """Rank borrowers by PD and cut into equal-population tiers (A = safest)."""
    labels = TIER_NAMES.get(n_tiers, [f"T{i+1}" for i in range(n_tiers)])
    series = pd.Series(np.asarray(pd_estimates), index=index)
    ranks = series.rank(method="first", pct=True)
    tiers = pd.cut(ranks, bins=np.linspace(0, 1, n_tiers + 1), labels=labels, include_lowest=True)
    return tiers.astype(str)


def expected_loss_table(
    df: pd.DataFrame,
    pd_estimates: np.ndarray,
    lgd: float = 0.60,
    n_tiers: int = 5,
    ead_col: str = "ead",
    actual_col: str | None = "default",
) -> pd.DataFrame:
    """One row per risk tier: PD, exposure, expected loss, realised default rate."""
    work = df.copy()
    work["pd"] = np.asarray(pd_estimates)
    work["tier"] = assign_tiers(pd_estimates, n_tiers, index=work.index)
    work["el"] = work["pd"] * lgd * work[ead_col]

    labels = TIER_NAMES.get(n_tiers, [f"T{i+1}" for i in range(n_tiers)])
    g = work.groupby("tier", observed=True)

    out = pd.DataFrame(
        {
            "borrowers": g.size(),
            "avg_pd": g["pd"].mean(),
            "min_pd": g["pd"].min(),
            "max_pd": g["pd"].max(),
            "avg_ead": g[ead_col].mean(),
            "total_ead": g[ead_col].sum(),
            "expected_loss": g["el"].sum(),
        }
    )
    if actual_col and actual_col in work:
        out["actual_default_rate"] = g[actual_col].mean()
        out["actual_loss"] = g.apply(
            lambda d: (d[actual_col] * lgd * d[ead_col]).sum(), include_groups=False
        )

    out["lgd"] = lgd
    out["el_rate_on_exposure"] = out["expected_loss"] / out["total_ead"]
    out["share_of_expected_loss"] = out["expected_loss"] / out["expected_loss"].sum()

    out = out.reindex([l for l in labels if l in out.index])
    return out.reset_index()


def portfolio_summary(tier_table: pd.DataFrame) -> dict:
    total_ead = tier_table["total_ead"].sum()
    total_el = tier_table["expected_loss"].sum()
    summary = {
        "borrowers": int(tier_table["borrowers"].sum()),
        "total_exposure": float(total_ead),
        "total_expected_loss": float(total_el),
        "expected_loss_rate": float(total_el / total_ead) if total_ead else 0.0,
    }
    if "actual_loss" in tier_table:
        actual = tier_table["actual_loss"].sum()
        summary["realised_loss"] = float(actual)
        summary["realised_loss_rate"] = float(actual / total_ead) if total_ead else 0.0
    return summary


def expected_loss_by_segment(
    df: pd.DataFrame,
    pd_estimates: np.ndarray,
    lgd: float = 0.60,
    ead_col: str = "ead",
) -> pd.DataFrame:
    """Thin-file books carry a higher PD on a smaller ticket — this shows the net."""
    work = df.copy()
    work["pd"] = np.asarray(pd_estimates)
    work["el"] = work["pd"] * lgd * work[ead_col]
    g = work.groupby("segment")
    out = pd.DataFrame(
        {
            "borrowers": g.size(),
            "avg_pd": g["pd"].mean(),
            "total_ead": g[ead_col].sum(),
            "expected_loss": g["el"].sum(),
        }
    )
    out["el_rate_on_exposure"] = out["expected_loss"] / out["total_ead"]
    return out.reset_index()


def cutoff_analysis(
    df: pd.DataFrame,
    pd_estimates: np.ndarray,
    lgd: float = 0.60,
    ead_col: str = "ead",
    steps: int = 20,
) -> pd.DataFrame:
    """
    Approve everyone below a PD cut-off and see what the book looks like.
    Turns the score into an underwriting policy question rather than a metric.
    """
    work = df.copy()
    work["pd"] = np.asarray(pd_estimates)
    rows = []
    for q in np.linspace(0.3, 1.0, steps):
        cut = float(np.quantile(pd_estimates, q))
        book = work[work["pd"] <= cut]
        if book.empty:
            continue
        el = (book["pd"] * lgd * book[ead_col]).sum()
        rows.append(
            {
                "approval_rate": len(book) / len(work),
                "pd_cutoff": cut,
                "approved_exposure": book[ead_col].sum(),
                "expected_loss": el,
                "el_rate": el / book[ead_col].sum(),
                "thin_file_share_of_approvals": (book["segment"] == "thin_file").mean(),
                "actual_default_rate": book["default"].mean() if "default" in book else np.nan,
            }
        )
    return pd.DataFrame(rows)
