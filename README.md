# PredictEvent — Event-Driven Congestion Intelligence for BTP
**Theme 2** | BTP × Flipkart Hackathon | Dataset: Astram event log (8,173 Bengaluru events, Nov 2023–Apr 2024)

## Problem
Political rallies, festivals, sports, construction, VIP movement and sudden incidents cause
localized traffic breakdowns. Today impact isn't quantified in advance, manpower deployment is
experience-driven, and there's no post-event learning loop.

## Our solution
A decision-support engine that **forecasts where events will cluster**, **scores each event's
risk of requiring road closure/barricading**, and **converts both into concrete resource actions**
(personnel count, barricade yes/no, diversion plan) — all on a laptop, no GPU.

### Three ML components (honest, time-based train/test split)
| Component | What it predicts | Result | Baseline | Use |
|---|---|---|---|---|
| **Hotspot Load Forecaster** | next-day event load per corridor | **R² 0.536, MAE 2.19** | seasonal-naive R² −0.32 | manpower allocation |
| **Closure-Risk Classifier** | will event need road closure? | **ROC-AUC 0.813, recall 0.76** | PR-AUC 0.072 | barricading / triage |
| Duration Predictor | event clearance time | not reliably learnable | — | see *Data findings* |

Threshold tuned for **F2 (recall-favoring)** — for BTP a missed closure costs more than a false alarm.
Top closure drivers (permutation importance): **event_cause** >> corridor > location > vehicle type.

**Deep ST-model benchmark** (`src/deep_forecast.py`, corridor embedding + GRU + dual-stream fusion,
identical split): MAE **2.12** vs GBM **2.24** (~5% better, consistent over 3 seeds), R² tied at 0.55.
Ensemble (GBM+deep) R² 0.55. GBM kept in production for robustness; deep model is the upgrade path.

**Rigor adds:**
- **Calibrated probabilities** — closure-risk is isotonic-calibrated: ECE 0.33→0.03, Brier 0.185→0.058
  (the OR engine consumes the probability directly, so it must mean what it says).
- **Recall-favoring safety floor** — displayed risk = max(model prob, empirical-Bayes historical cause
  rate), so rare high-impact causes (VIP movement: ~20 samples, 80% historical) are never under-flagged.
- **Cross-validation ± CI** (`src/validate.py`) — 5-fold expanding-window: hotspot R² 0.56±0.14,
  closure ROC-AUC 0.73±0.05. Reports the spread, not a cherry-picked single split.

### Resource recommendation layer
Transparent rules on top of model outputs (jury-defensible, not a black box):
- **Manpower** = f(forecasted load × corridor arterial weight)
- **Action tier** RED/AMBER/GREEN = f(closure risk, priority)
- **Barricade / Diversion** triggered by closure risk + event cause (VIP, public event, procession)

## Key data findings (drives credibility)
1. **Timestamps are local IST mislabeled `+00`.** Validated via bimodal rush-hour signature
   (evening 7–11 PM + night heavy-vehicle window). We use stored hour as-is.
2. **`closed_datetime` is administrative ticket-close, not true on-ground clearance** → exact
   duration is dominated by reporting latency and is not learnable. We forecast *load* and
   *closure-risk* instead, which are robust. *Future work: capture real clearance timestamps.*
3. **Caught & removed target leakage:** `endlatitude/endlongitude` (a "road stretch" flag) is
   filled *because* a closure happened (98% vs 0% closure rate) — using it gave a fake ROC-AUC of
   1.0. Permutation importance exposed it; we dropped it for an honest 0.81.
4. Event-driven causes (public_event, procession, vip_movement, protest, construction) have the
   highest closure rates (40–80%) — these are the true Theme-2 targets vs background breakdowns.

## Run
```bash
pip install -r requirements.txt
python src/eda.py        # figures -> outputs/
python src/train.py      # closure + duration models -> outputs/
python src/hotspot.py    # load forecaster -> outputs/
streamlit run app.py     # interactive dashboard demo
```

## Dashboard (finale demo)
- **Tab 1 — Hotspot Forecast + OR Allocation:** corridor load forecast → M/M/c SLA-sized officer
  counts + MCLP coverage-optimal barricade placement (risk×load weighted).
- **Tab 2 — Live Event Triage:** per-event closure risk %, RED/AMBER/GREEN tier, barricade &
  diversion advice, incident map.
- **Tab 3 — Analytics:** historical hour/day/duration/spatial patterns + feature importance.

Sidebar exposes the operational SLA (max wait, mean handling time) that drives the queueing engine.

## OR optimization engine
- **M/M/c queueing (Erlang-C)** — sizes officers so expected response wait ≤ SLA (service-level guarantee).
- **MCLP (PuLP/CBC)** — places P units to cover maximum risk×load-weighted demand within a radius.
- Evidence base: SMU IJCAI-2019 (forecast→optimize), Vlahogianni 2019 (KDE+MCLP), Miao/Easa 2021 (queueing→staffing).

## Where it fits in BTP's stack (verified)
BATCS optimizes *signals*; ASTRAM provides *awareness* (our data source). Neither yields a
prescriptive police-deployment plan. PredictEvent is the **prescriptive OR layer on top of ASTRAM**.
See `BTP_POSITIONING.md`.

## Repo
```
data/events.csv          provided dataset (only HackerEarth data used — no external)
src/eda.py               EDA + figures
src/train.py             closure classifier (leakage-checked) + duration models
src/hotspot.py           spatio-temporal load forecaster (headline)
src/recommend.py         cause-aware action playbook (RED/AMBER/GREEN, barricade, diversion, officers)
src/bengaluru_context.py real-world context: corridor congestion, commute peaks, monsoon, festivals,
                         metro construction, heavy-vehicle bans (researched 2024-26)
src/test_recommendations.py validation of the recommendation engine on the real data
src/optimize.py          OR engine — M/M/c (Erlang-C) staffing + MCLP placement
src/deep_forecast.py     deep ST-model benchmark (corridor embedding + GRU + fusion) + ensemble
src/validate.py          time-series cross-validation with confidence intervals
src/make_ppt.py          generates the finale slide deck
app.py                   Streamlit dashboard (4 tabs)
CONCEPT_NOTE.md          submission concept note
BTP_POSITIONING.md       verified ASTRAM/BATCS positioning
outputs/                 models (.joblib), metrics.json, panel.csv, figures, deck.pptx
```
