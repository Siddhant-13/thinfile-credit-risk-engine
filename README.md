# Thin-File Credit Risk Engine
 
Alternative-data credit scoring for new-to-credit borrowers, with a segment-level
validation framework that exposes a failure mode portfolio-level metrics hide.
 
Python · scikit-learn · XGBoost · LightGBM · SHAP · Streamlit
 
---
 
## The problem
 
A bureau-only scorecard cannot underwrite someone who has no bureau record. In markets with
large underbanked populations, a meaningful share of applicants are new-to-credit: no score,
no history, no prior loans. The conventional model doesn't fail loudly on these applicants -
it produces a number, and the number is noise.
 
This project quantifies that, and shows what behavioural data recovers.
 
## The finding
 
Held-out AUC. Same models, evaluated on each borrower segment separately:
 
| Model | Full book | Traditional borrowers | Thin-file borrowers |
|---|---:|---:|---:|
| Scorecard - bureau features only | 0.711 | 0.787 | **0.521** |
| Scorecard - bureau + alternative | 0.848 | 0.815 | **0.907** |
| XGBoost - bureau features only | 0.685 | 0.772 | **0.489** |
| XGBoost - bureau + alternative | 0.862 | 0.808 | **0.925** |
 
The bureau-only scorecard is not a weak model. On borrowers who have a credit file it scores
0.787 - a workable retail scorecard. On borrowers who don't, it scores 0.521, statistically
indistinguishable from guessing. The portfolio-level 0.711 conceals this entirely.
 
Adding alternative data (transaction frequency and recency, spending diversity, digital
footprint, income volatility, small-ticket repayment history) lifts the thin-file segment to
0.907 - a 0.39 AUC gain on the segment that needed it, with no material loss elsewhere.
 
Two secondary results:
 
- Augmenting features **slightly degrades** performance on the traditional segment
  (0.815 → 0.787). Behavioural noise dilutes an already-strong bureau signal. A production
  lender would run segment-specific models rather than one augmented model for everyone.
- Gradient boosting beats the WOE scorecard by roughly 0.01–0.02 AUC on identical features.
  That is the honest magnitude of the interpretability trade-off, and it is small.
## Running it
 
```bash
git clone <repo>
cd thin-file-credit-risk
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
 
python pipeline.py        # full study to stdout, ~30s
streamlit run app.py      # interactive interface
```
 
No external dataset. The population is generated in-process from a fixed seed.
 
## Architecture
 
| Module | Responsibility |
|---|---|
| `data_generator.py` | Synthetic population. Bureau and behavioural blocks are driven by **separate** latent factors, so alternative data is not a disguised proxy for the credit score. Default is generated from observed features with segment-specific weights; intercepts solved by bisection to hit target default rates. |
| `woe_scorecard.py` | Supervised WOE binning (decision-tree split points), Information Value per feature, IV-threshold selection, logistic regression on WOE, PDO scaling into a published points table. Missing values get their own bin - never imputed. |
| `ml_models.py` | XGBoost and LightGBM, randomised hyperparameter search on stratified 3-fold CV. Class weighting for imbalance, plus prior correction to reverse it before any PD is used for currency. |
| `validation.py` | AUC, Gini, KS, per-class precision/recall/F1, PSI with standard 0.10/0.25 thresholds, and per-segment reporting across all models. |
| `explainability.py` | TreeSHAP attribution, global and per-segment, plus an explicit scorecard-vs-boosting trade-off table and feature-rank agreement. |
| `expected_loss.py` | Risk tiering, EL = PD × LGD × EAD, per-segment loss decomposition, approval cut-off analysis. |
| `pipeline.py` | Orchestration. One shared stratified train/test split so every comparison is like-for-like. |
| `app.py` | Five-tab Streamlit interface: population, scorecard builder, model comparison, segment analysis, risk tiers and expected loss. Adjustable IV threshold, hyperparameters, LGD and tier count. |
 
## Two design decisions worth explaining
 
**Missingness is a feature, not a gap to fill.**
Thin-file applicants carry `NaN` in bureau columns plus an explicit `*_missing` indicator.
Median-imputing a credit score would tell the model that a new-to-credit applicant is an
average-risk borrower - precisely the wrong prior. Boosted trees learn their own split
direction for `NaN`; the scorecard gives missing its own WOE bucket. The SHAP-by-segment
view shows the model switching evidence bases depending on what is populated, without being
told to.
 
**Class weighting preserves ranking and destroys calibration.**
`scale_pos_weight ≈ 7.2` lifts mean predicted PD to ~28% against an observed default rate of
12%. AUC, Gini and KS are rank statistics and are unaffected. Expected loss is not:
uncorrected, portfolio EL overstates by ~2.3×. `prior_correct()` divides the modelled odds
back by the weight before tiering, and the interface surfaces the residual calibration gap
rather than hiding it.
 
## Limitations
 
Stated plainly, because the numbers above should not be read without them.
 
- **The data is synthetic and the default process is authored.** Thin-file defaults were
  constructed to depend on behavioural features. What this demonstrates is that the
  *methodology* recovers a segment-specific signal, and that segment-blind evaluation hides
  it - not that alternative data achieves 0.9 AUC on real underbanked borrowers. Real-world
  uplift over a no-file baseline is meaningful but substantially smaller.
- **Information Values run implausibly high** (0.3–0.7). On a real book, IV above 0.5
  usually indicates target leakage rather than a strong feature. Artefact of clean synthetic
  generation.
- **LGD is a fixed 60% assumption**, not an estimate. No recovery, collateral or cure
  modelling.
- **No fair-lending analysis.** Device consistency and digital footprint correlate with
  income and geography in ways a bureau score does not. Proxy-discrimination testing is a
  prerequisite for any production use of this approach. Its absence is this project's
  largest gap.
- **PSI drift is simulated** - vintage 2 has account age shifted down ~12% and transaction
  counts up ~6% so the stability check has something to detect.
- Single train/test split, no nested CV. Reported AUCs carry a standard error of a few
  points at these sample sizes.