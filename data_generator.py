"""
Synthetic loan-applicant population for the alternative-data credit risk study.

The population is deliberately built in two segments:

  traditional : full bureau footprint (score, history, prior loans, DTI)
  thin_file   : new-to-credit / underbanked. Bureau fields are mostly absent.
                Missing is left as NaN and flagged, never imputed with a
                fake value, because the whole point of the study is that the
                traditional feature set has nothing to say about these people.

Default is generated from the *observed* features, with segment-specific
weights: bureau-dominant for traditional borrowers, alternative-data-dominant
for thin-file borrowers. That relationship is what the segment analysis is
supposed to recover later, so it is written explicitly here rather than
smuggled in through a shared latent variable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Feature inventory
# --------------------------------------------------------------------------

TRADITIONAL_FEATURES = [
    "credit_score",
    "credit_history_years",
    "num_prior_loans",
    "debt_to_income",
]

ALTERNATIVE_FEATURES = [
    "txn_count_3m",
    "days_since_last_txn",
    "merchant_category_count",
    "account_age_months",
    "device_consistency",
    "income_cv",
    "bnpl_ontime_rate",
]

# Book-keeping columns that are never model inputs.
META_COLUMNS = ["applicant_id", "segment", "is_thin_file", "batch", "ead", "default"]

TARGET = "default"

DEFAULT_LGD = 0.60


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _z(x: pd.Series) -> pd.Series:
    """Z-score that tolerates NaN (missing rows contribute 0 to the index)."""
    return ((x - x.mean()) / (x.std(ddof=0) + 1e-9)).fillna(0.0)


def _solve_intercept(index: np.ndarray, target_rate: float) -> float:
    """Bisect on the intercept so the realised default probability matches."""
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _sigmoid(index + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_population(
    n: int = 8_000,
    thin_file_share: float = 0.40,
    default_rate_traditional: float = 0.09,
    default_rate_thin_file: float = 0.17,
    seed: int = 7,
) -> pd.DataFrame:
    """Return a scored-ready applicant table. Overall default rate lands ~12-13%."""
    rng = np.random.default_rng(seed)

    is_thin = rng.random(n) < thin_file_share
    n_thin = int(is_thin.sum())

    df = pd.DataFrame(
        {
            "applicant_id": np.arange(1, n + 1),
            "is_thin_file": is_thin.astype(int),
            "segment": np.where(is_thin, "thin_file", "traditional"),
        }
    )

    # ---------------------------------------------------------------- bureau
    # Latent creditworthiness only shapes the bureau block; it keeps the
    # traditional features internally coherent (a good score goes with a low
    # DTI) without making them a proxy for the alternative block.
    q = rng.normal(size=n)

    credit_score = np.clip(rng.normal(690 + 55 * q, 55), 300, 900)
    history_years = np.clip(rng.gamma(4.0, 1.6, n) + 1.5 * q, 0.1, 35)
    prior_loans = rng.poisson(np.clip(2.5 + 0.8 * q, 0.2, None))
    dti = np.clip(rng.beta(2.4, 5.5, n) * 1.15 - 0.10 * q, 0.01, 0.95)

    # Thin-file borrowers: almost no bureau footprint.
    #   ~75% have no score at all, the rest carry a provisional low-confidence
    #   score. History is short by construction, DTI is only known where an
    #   income document exists.
    has_score = rng.random(n) < 0.25
    has_dti = rng.random(n) < 0.45

    credit_score = np.where(
        is_thin,
        np.where(has_score, np.clip(rng.normal(640, 45, n), 300, 780), np.nan),
        credit_score,
    )
    history_years = np.where(is_thin, np.round(rng.uniform(0.0, 1.5, n), 2), history_years)
    prior_loans = np.where(is_thin, rng.binomial(1, 0.35, n), prior_loans)
    dti = np.where(is_thin, np.where(has_dti, np.clip(rng.beta(2.0, 4.0, n), 0.01, 0.95), np.nan), dti)

    df["credit_score"] = np.round(credit_score, 0)
    df["credit_history_years"] = np.round(history_years, 2)
    df["num_prior_loans"] = prior_loans.astype(float)
    df["debt_to_income"] = np.round(dti, 3)

    # ----------------------------------------------------- alternative block
    # Driven by a *separate* latent behaviour factor. Thin-file applicants get
    # a slightly richer digital footprint (they are the app-native cohort) and
    # a cleaner signal-to-noise ratio, which is why alternative data can carry
    # the underwriting decision for them.
    b = rng.normal(size=n)
    noise = np.where(is_thin, 0.55, 0.85)  # traditional block is noisier
    b_obs = b + rng.normal(0, noise, n)

    txn_count = rng.poisson(np.exp(2.85 + 0.30 * b_obs)).astype(float)
    days_since = np.round(rng.exponential(np.clip(9.0 - 2.4 * b_obs, 1.0, None)), 1)
    cat_count = rng.binomial(16, np.clip(_sigmoid(0.45 * b_obs - 0.25), 0.02, 0.98)).astype(float)
    acct_age = np.round(np.clip(rng.gamma(3.0, 9.0, n) + 6 * b_obs, 1, 240), 0)
    device_cons = np.round(np.clip(rng.beta(6.0, 2.2, n) + 0.06 * b_obs, 0.05, 1.0), 3)
    income_cv = np.round(np.clip(rng.gamma(2.4, 0.16, n) - 0.05 * b_obs, 0.02, 2.5), 3)

    # Thin-file applicants are the ones actually using BNPL, so coverage of
    # the small-ticket repayment history is higher for them.
    has_bnpl = rng.random(n) < np.where(is_thin, 0.78, 0.42)
    ontime = np.clip(rng.beta(np.clip(6.0 + 2.2 * b_obs, 0.5, None), 1.9, n), 0.0, 1.0)
    df["bnpl_ontime_rate"] = np.where(has_bnpl, np.round(ontime, 3), np.nan)

    df["txn_count_3m"] = txn_count
    df["days_since_last_txn"] = days_since
    df["merchant_category_count"] = cat_count
    df["account_age_months"] = acct_age
    df["device_consistency"] = device_cons
    df["income_cv"] = income_cv

    # --------------------------------------------------------------- default
    trad_index = (
        -0.95 * _z(df["credit_score"])
        + 0.55 * _z(df["debt_to_income"])
        - 0.35 * _z(df["credit_history_years"])
        + 0.20 * _z(df["num_prior_loans"])
    )
    alt_index = (
        -0.55 * _z(df["txn_count_3m"])
        + 0.45 * _z(df["days_since_last_txn"])
        - 0.35 * _z(df["merchant_category_count"])
        - 0.30 * _z(df["account_age_months"])
        - 0.35 * _z(df["device_consistency"])
        + 0.45 * _z(df["income_cv"])
        - 0.70 * _z(df["bnpl_ontime_rate"].fillna(df["bnpl_ontime_rate"].mean()))
        + 0.15 * df["bnpl_ontime_rate"].isna().astype(float)
    )

    index = np.where(
        is_thin,
        0.30 * trad_index + 1.35 * alt_index,   # thin-file: behaviour decides
        1.15 * trad_index + 0.30 * alt_index,   # traditional: bureau decides
    )
    index = index + rng.normal(0, 0.35, n)

    prob = np.empty(n)
    for mask, rate in ((~is_thin, default_rate_traditional), (is_thin, default_rate_thin_file)):
        a = _solve_intercept(index[mask], rate)
        prob[mask] = _sigmoid(index[mask] + a)

    df["default"] = rng.binomial(1, prob)

    # ------------------------------------------------------ exposure & batch
    # Thin-file tickets are smaller — that matters for expected loss, since a
    # higher PD on a smaller exposure is not automatically a worse portfolio.
    ead = np.where(
        is_thin,
        rng.lognormal(9.9, 0.55, n),
        rng.lognormal(10.7, 0.60, n),
    )
    df["ead"] = np.round(np.clip(ead, 3_000, 900_000), 0)

    # Two time-simulated vintages for the PSI check, with mild genuine drift
    # in the digital-footprint features (newer cohort is younger on-book).
    df["batch"] = np.where(rng.random(n) < 0.5, "vintage_1", "vintage_2")
    v2 = df["batch"] == "vintage_2"
    df.loc[v2, "account_age_months"] = np.clip(
        df.loc[v2, "account_age_months"] * rng.normal(0.88, 0.05, v2.sum()), 1, 240
    ).round(0)
    df.loc[v2, "txn_count_3m"] = np.clip(
        df.loc[v2, "txn_count_3m"] * rng.normal(1.06, 0.05, v2.sum()), 0, None
    ).round(0)

    return df[META_COLUMNS + TRADITIONAL_FEATURES + ALTERNATIVE_FEATURES]


# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------

def add_missing_flags(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Append explicit `<col>_missing` indicators. Missingness is information."""
    out = df.copy()
    flags = []
    for col in columns:
        if out[col].isna().any():
            flag = f"{col}_missing"
            out[flag] = out[col].isna().astype(int)
            flags.append(flag)
    return out, flags


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """The two parallel views the whole project compares."""
    _, trad_flags = add_missing_flags(df, TRADITIONAL_FEATURES)
    _, all_flags = add_missing_flags(df, TRADITIONAL_FEATURES + ALTERNATIVE_FEATURES)
    return {
        "traditional": TRADITIONAL_FEATURES + trad_flags,
        "augmented": TRADITIONAL_FEATURES + ALTERNATIVE_FEATURES + all_flags,
    }


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Attach missing flags and return the frame plus the two feature lists."""
    out, _ = add_missing_flags(df, TRADITIONAL_FEATURES + ALTERNATIVE_FEATURES)
    return out, feature_sets(df)


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("segment")
    summary = pd.DataFrame(
        {
            "applicants": g.size(),
            "share": g.size() / len(df),
            "default_rate": g["default"].mean(),
            "avg_exposure": g["ead"].mean(),
            "bureau_score_coverage": g["credit_score"].apply(lambda s: s.notna().mean()),
            "bnpl_history_coverage": g["bnpl_ontime_rate"].apply(lambda s: s.notna().mean()),
        }
    )
    return summary.reset_index()


if __name__ == "__main__":
    pop = generate_population()
    print(pop.head())
    print()
    print(segment_summary(pop).to_string(index=False))
    print(f"\noverall default rate: {pop['default'].mean():.3f}")
