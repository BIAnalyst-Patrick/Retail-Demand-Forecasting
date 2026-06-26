# Retail Demand Forecasting: A 4-Model Bake-Off

**Industry**: Retail
**Tools**: Python (statsmodels, Prophet, scikit-learn)

---

## 🚀 Executive Summary

**Problem**: The business needed a reliable sales forecast for inventory and staffing planning, but had no structured way to know which forecasting method — or even whether a simple baseline — would actually perform best on its own sales pattern.

**Action**: Built a daily sales series (Jan 2022 – Jun 2025) showing real trend, weekly, and annual seasonality, then ran a head-to-head bake-off across four forecasting approaches — Seasonal Naive baseline, Holt-Winters Exponential Smoothing, SARIMA (selected via AIC grid search), and Prophet — evaluated on a common 90-day holdout.

**Result**: **Prophet won decisively**, cutting forecast error to **6.88% MAPE** versus 10.70% for Holt-Winters, 14.00% for SARIMA, and 14.96% for the naive baseline — a result driven by Prophet's native handling of both weekly *and* yearly seasonality, which the other models could only partially capture.

---

## 🎯 Problem Statement

**Core Issue**: Forecasting method selection is often made by habit or familiarity ("we always use SARIMA") rather than by evidence. Without a fair, holdout-based comparison, the business risks planning inventory and staffing around a forecast that isn't actually the best fit for its data.

**Key Questions**:
1. How much better is even a basic statistical model (Holt-Winters, SARIMA) than a naive seasonal repeat?
2. Does a model that captures multiple seasonal cycles (Prophet) meaningfully outperform models that only capture one (SARIMA at weekly resolution)?
3. Which model should be put into production, and what does that decision cost in forecast accuracy if gotten wrong?

---

## 📈 Objectives & Key Metrics

| Objective | Metric Tracked | Result Achieved |
|---|---|---|
| Establish a credible baseline | Seasonal Naive MAPE | **14.96%** |
| Test classical exponential smoothing | Holt-Winters MAPE | **10.70%** (28% better than baseline) |
| Test ARIMA-family seasonal modeling | SARIMA MAPE (AIC-selected order) | **14.00%** (barely beat baseline) |
| Test multi-seasonality decomposition | Prophet MAPE | **6.88%** (54% better than baseline) |
| Select the production model on evidence | Lowest holdout MAPE | **Prophet** |

---

## 📂 Data Overview

**Data Sources**: A synthetic but behaviorally realistic daily sales series (`data/generate_data.py` → `data/daily_sales.csv`) covering **1,277 days** (Jan 2022 – Jun 2025), built with deliberate trend, weekly, and annual seasonality so the forecasting comparison reflects genuine signal rather than random noise.

**Key Variables**: `date`, `sales` (daily revenue), `promo_flag` (binary flag for Black Friday, Christmas/New Year, and recurring mid-month promotions).

**Data Challenges**: As a synthetic series, it doesn't carry real-world disruptions (stockouts, macro shocks, competitor actions). It's designed specifically to give all four models a fair, realistic pattern to forecast against — trend + weekly seasonality + annual seasonality + noise — not to represent a live business.

---

## 🔧 Methodology

**Train/Test Split**: Last 90 days (Apr 2 – Jun 30, 2025) held out; all models trained only on data before that window.

**Models Compared**:
1. **Seasonal Naive** — repeats the last observed 7-day cycle forward; the baseline every other model must beat to justify its complexity.
2. **Holt-Winters Exponential Smoothing** (`statsmodels`) — additive trend + weekly seasonality.
3. **SARIMA** (`statsmodels SARIMAX`) — small grid search over `(p,d,q)(P,D,Q,7)` combinations, best order selected by **AIC on training data only** (no test-set leakage), then evaluated on the holdout.
4. **Prophet** — automatic decomposition into trend, weekly seasonality, and yearly seasonality.

**Evaluation**: MAPE and RMSE on the 90-day holdout, decided by evidence rather than assumed in advance.

**Tools**: Python (`pandas`, `statsmodels` for Holt-Winters/SARIMA, `prophet` for the Bayesian-decomposition model, `scikit-learn` for MAPE/RMSE scoring).

**Notebook**: [`notebooks/Retail_Demand_Forecasting.ipynb`](notebooks/Retail_Demand_Forecasting.ipynb) — full executed analysis with decomposition plots, model fitting, and a forecast-vs-actual comparison chart.

---

## 💡 Key Insights

#### Insight 1: A Naive Baseline Is a Real Benchmark — and SARIMA Barely Cleared It
**What**: SARIMA (14.00% MAPE) only marginally beat the seasonal naive baseline (14.96% MAPE), despite being a far more complex model.
**So What**: Model sophistication doesn't guarantee better forecasts. At weekly seasonal resolution, SARIMA couldn't capture the annual holiday pattern that materially shapes this business's sales — a reminder to always benchmark against a naive baseline before trusting a complex model's added value.

#### Insight 2: Multi-Seasonality Decomposition Is the Real Differentiator
**What**: Prophet (6.88% MAPE) outperformed every other model by a wide margin, including Holt-Winters (10.70%) which also models weekly seasonality but not yearly seasonality.
**So What**: The performance gap isn't about "Prophet vs. SARIMA" as algorithms — it's about whether the model can represent more than one seasonal cycle at once. Any production forecasting setup for this kind of data should prioritize multi-seasonality support over a specific algorithm brand.

#### Insight 3: The SARIMA Fit Showed Convergence Issues Worth Flagging
**What**: The selected SARIMA model produced a `ConvergenceWarning` during maximum-likelihood fitting, meaning its parameter estimates may not be fully optimal even within the AIC-selected order.
**So What**: This is a legitimate caveat rather than a reason to discard SARIMA broadly — it suggests the order grid or differencing strategy would need further tuning before SARIMA could be fairly judged against Prophet on this data.

---

## ✅ Recommendations & Business Impact

| Priority | Recommendation | Expected Impact | Owner |
|---|---|---|---|
| High | Deploy Prophet as the production forecasting model for inventory/staffing planning | Cuts forecast error from a 14-15% baseline/SARIMA range to **6.88%**, directly reducing over/under-stocking | Operations / Planning Team |
| Medium | Re-run this bake-off whenever a new sales channel or major seasonal pattern is introduced | Prevents silently using a stale "best" model as the business's seasonal structure changes | Data/Analytics Team |
| Medium | Investigate the SARIMA convergence warning before ruling SARIMA out long-term | A properly converged SARIMA might close some of the gap; cheap to verify, avoids a premature conclusion | Data/Analytics Team |
| Low | Extend Prophet with holiday/promo regressors (the `promo_flag` column is already collected) | Likely improves accuracy further around known promotional spikes | Data/Analytics Team |

---

## 📉 Caveats & Next Steps

**Limitations**:
- Dataset is synthetic — generated with realistic but author-defined trend/seasonality patterns, not collected from a live business. The forecasting methodology and evaluation rigor are fully production-ready; the specific numbers reflect this dataset only.
- The SARIMA grid search covered a deliberately small set of orders for runtime reasons; a wider/auto-ARIMA search might find a better-fitting SARIMA model.
- The 90-day holdout is a single test window; results could shift with a different holdout period (e.g., one spanning the Nov/Dec peak).

**Next Steps**:
- Re-evaluate the bake-off with a holdout window that includes the Nov/Dec holiday peak, to test whether Prophet's advantage holds during the highest-stakes forecasting period.
- Add `promo_flag` as a Prophet regressor and re-score to quantify the incremental lift from including known promotional events.
- Productionize the winning model with a scheduled weekly refresh and a monitoring alert if live MAPE drifts materially above the 6.88% holdout benchmark.

---

## 👤 Author

**Analyst:** Patrick Mwangi Gichuki
**Portfolio:** [https://gichuki.site](https://gichuki.site)
**LinkedIn:** [https://linkedin.com/in/patrick-gichuki-datascientist](https://linkedin.com/in/patrick-gichuki-datascientist)
