"""Discrimination, separation and stability metrics for every model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


def auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def gini(y_true, y_score) -> float:
    return 2 * auc(y_true, y_score) - 1


def ks_statistic(y_true, y_score) -> float:
    """Max separation between the cumulative good and bad score distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def metrics(y_true, y_score, threshold: float | None = None) -> dict:
    """Headline metrics. Threshold defaults to the KS-optimal cut-off."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    if threshold is None:
        threshold = float(thr[np.argmax(tpr - fpr)])
    y_hat = (y_score >= threshold).astype(int)
    rep = classification_report(y_true, y_hat, output_dict=True, zero_division=0)
    return {
        "n": int(len(y_true)),
        "defaults": int(y_true.sum()),
        "default_rate": float(y_true.mean()),
        "auc": auc(y_true, y_score),
        "gini": gini(y_true, y_score),
        "ks": float(np.max(tpr - fpr)),
        "threshold": threshold,
        "precision_default": rep["1"]["precision"],
        "recall_default": rep["1"]["recall"],
        "f1_default": rep["1"]["f1-score"],
        "accuracy": rep["accuracy"],
    }


def classification_table(y_true, y_score, threshold: float) -> pd.DataFrame:
    y_hat = (np.asarray(y_score) >= threshold).astype(int)
    rep = classification_report(y_true, y_hat, output_dict=True, zero_division=0,
                                target_names=["repaid", "default"])
    return pd.DataFrame(rep).T.reset_index().rename(columns={"index": "class"})


def roc_points(y_true, y_score) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    """
    Population Stability Index between two score distributions.

        PSI = sum (actual% - expected%) * ln(actual% / expected%)

    Market convention: < 0.10 stable, 0.10-0.25 watch, > 0.25 investigate.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts = np.histogram(expected, bins=edges)[0].astype(float)
    a_counts = np.histogram(actual, bins=edges)[0].astype(float)
    e_pct = np.clip(e_counts / e_counts.sum(), 1e-6, None)
    a_pct = np.clip(a_counts / a_counts.sum(), 1e-6, None)

    contrib = (a_pct - e_pct) * np.log(a_pct / e_pct)
    detail = pd.DataFrame(
        {
            "bucket": [f"{edges[i]:.1f} to {edges[i+1]:.1f}" for i in range(len(edges) - 1)],
            "expected_pct": e_pct,
            "actual_pct": a_pct,
            "psi_contribution": contrib,
        }
    )
    return float(contrib.sum()), detail


def psi_verdict(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "monitor"
    return "unstable — investigate"


def compare_models(results: dict[str, tuple]) -> pd.DataFrame:
    """results: {model_name: (y_true, y_score)} -> one row per model."""
    rows = []
    for name, (y_true, y_score) in results.items():
        row = {"model": name}
        row.update(metrics(y_true, y_score))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def segment_report(
    y_true: pd.Series,
    scores: dict[str, np.ndarray],
    segments: pd.Series,
) -> pd.DataFrame:
    """Every model x every segment (plus the full book). The core comparison."""
    rows = []
    groups = [("all", pd.Series(True, index=y_true.index))]
    groups += [(s, segments == s) for s in sorted(segments.unique())]

    for model, score in scores.items():
        score = pd.Series(np.asarray(score), index=y_true.index)
        for seg, mask in groups:
            if y_true[mask].nunique() < 2:
                continue
            row = {"model": model, "segment": seg}
            row.update(metrics(y_true[mask], score[mask]))
            rows.append(row)
    return pd.DataFrame(rows)
