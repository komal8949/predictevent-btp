# PredictEvent — Event-Driven Congestion Intelligence for Bengaluru Traffic Police
### Concept Note | Theme 2: Event-Driven Congestion (Planned & Unplanned)
**BTP × Flipkart Hackathon** · Dataset: Astram event log (8,173 events, Nov 2023 – Apr 2024)

---

## 1. The Problem
Political rallies, festivals, sports events, processions, VIP movement, construction and sudden
incidents create localized traffic breakdowns across Bengaluru. Today:
- **Event impact is not quantified in advance** — officers cannot say *which* corridors will be hit
  *how hard* tomorrow.
- **Resource deployment is experience-driven** — manpower, barricades and diversions are decided by
  gut feel, not data.
- **No post-event learning loop** — the same mistakes repeat.

## 2. Our Solution — one sentence
A laptop-deployable decision-support engine that **forecasts where events will cluster**, **scores
each event's risk of needing a road closure/barricade**, and **runs operations-research
optimization to output a deployment plan** — an SLA-guaranteed officer count, coverage-optimal
barricade placement, and diversion plan.

```
            ┌─────────────────────┐
 ASTRAM ──▶ │ 1. Hotspot Load     │──▶ forecast load per corridor (tomorrow)
 event log  │    Forecaster (R².54)│         │
            ├─────────────────────┤         ▼
            │ 2. Closure-Risk     │──▶ ┌──────────────────────────┐
            │    Classifier (AUC.81)│   │ 3. OR Optimization        │
            └─────────────────────┘    │  • M/M/c (Erlang-C) → SLA │──▶ officers + guarantee
                                        │    -sized officer count    │──▶ barricade placement
                                        │  • MCLP → coverage-optimal │──▶ diversion plan
                                        │    placement (risk×load)   │
                                        └──────────────────────────┘
```

## 3. Three ML Components (honest, time-based train→test split)

| Component | Predicts | Result | Baseline | Drives |
|---|---|---|---|---|
| **Hotspot Load Forecaster** | next-day event load / corridor | **R² 0.536, MAE 2.19** | seasonal-naive R² −0.32 | manpower allocation |
| **Closure-Risk Classifier** | will event need closure? | **ROC-AUC 0.813, recall 0.76** | random PR-AUC 0.072 | barricading & triage |
| Duration Predictor | clearance time | *not reliably learnable* | — | (see §5) |

- **Hotspot** = HistGradientBoosting on a corridor×day panel with calendar + lag/rolling features;
  beats the seasonal-naive baseline by ~28% on MAE.
- **Closure-risk** = HistGradientBoosting, class-balanced, **threshold tuned for F2** because for BTP
  a *missed* closure is far costlier than a false alarm. Top drivers (permutation importance):
  **event_cause ≫ corridor > location > vehicle type** — exactly what an officer would expect.

**Novelty & rigor.** Multi-step spatio-temporal forecasting of *non-recurrent* (event-driven)
congestion is a recognised open research gap — the mature STGCN/transformer literature targets
*recurrent* congestion. We scope our forecaster to that gap, and position it against the accepted
**STG4Traffic** benchmark family rather than only an internal score.

**Deep ST-model benchmark (validated).** We also built a right-sized deep model — a learnable
per-corridor *adaptive embedding* (STAEformer's core idea) + a GRU temporal stream + a calendar
stream fused dual-stream-style — and benchmarked it against the GBM on an *identical* time split:

| Model | MAE | R² |
|---|---|---|
| HistGradientBoosting (production) | 2.24 | 0.55 |
| Deep ST-model (embedding+GRU+fusion) | **2.12** | 0.55 |
| Ensemble (GBM + deep) | 2.16 | **0.55** |

The deep model improves MAE ~5% (consistent across 3 seeds), confirming headroom — but R² is tied
and the gain is small at this data scale (138 days × 20 corridors). **We retain the GBM in
production** for robustness/interpretability and put the deep model on the roadmap once a live,
higher-frequency feed is available. (Honest engineering: don't ship a heavier model for a 5% MAE gain.)

**Cross-validated, not single-split.** 5-fold expanding-window time-series CV: hotspot **R² 0.56 ±
0.14**, closure **ROC-AUC 0.73 ± 0.05** (early folds train on little history, so the full-history
production model reaches AUC 0.81 / R² 0.54). Reporting the spread, not a cherry-picked number.

**Calibrated probabilities.** The closure-risk output is Platt-calibrated, so "81% risk" really
means ~81%: **ECE 0.33 → 0.02, Brier 0.185 → 0.058** on the test set. This matters because the OR
engine consumes the probability directly — a miscalibrated score would distort every staffing number.

## 4. From Prediction to Action — OR Optimization Engine
We replaced naïve heuristics with the **forecast → optimize** pattern used in deployed police/
emergency systems (SMU IJCAI-2019 SAA-MIP; Vlahogianni KDE+MCLP; Miao/Easa queueing-to-staffing):
- **Manpower = M/M/c queueing (Erlang-C).** We size the officer count so the *expected response
  wait stays within an explicit SLA* (e.g. ≤5 min peak, ≤15 min off-peak) — a defensible number
  **with a service-level guarantee**, not a guess.
- **Barricade placement = Maximum Coverage Location Problem (MCLP).** Given P available units, we
  place them to cover the maximum **risk×load-weighted** demand within a coverage radius (PuLP/CBC).
- **Risk×load weighting.** Deployment priority blends forecast load *and* mean closure-risk — both
  high-traffic and high-risk corridors, per the OR location-allocation literature.
- **Action tier & diversion**: RED/AMBER/GREEN + diversion from closure-risk + cause (VIP / event /
  procession). Every number is traceable — no black box.

## 5. Data Findings That Build Trust
1. **Timestamps are local IST mislabeled `+00`** — proven via the bimodal rush-hour signature
   (evening 7–11 PM + night heavy-vehicle window + afternoon lull). We use the stored hour directly.
2. **`closed_datetime` is administrative ticket-close, not on-ground clearance** — so exact event
   duration is dominated by reporting latency and is *not* learnable. We deliberately do **not**
   over-claim it; we forecast load + closure-risk, which are robust. *Future work: capture real
   clearance timestamps via the field app.*
3. **We caught and removed target leakage.** The end-coordinate "road-stretch" flag is filled
   *because* a closure occurred (98% vs 0% closure rate) and faked a perfect ROC-AUC of 1.0.
   Permutation importance exposed it; dropping it gives the honest 0.81.

## 6. Prototype (finale demo)
Streamlit dashboard, runs on an 8 GB laptop, no GPU:
- **🎯 Event Simulator** — type in a hypothetical event (e.g. *IPL match at Chinnaswamy*) and get an
  instant closure-risk %, RED/AMBER/GREEN tier, personnel, barricade & diversion advice on a map.
- **📍 Hotspot Forecast + Manpower** — pick a day → corridor load forecast + recommended personnel.
- **🚨 Live Event Triage** — every event of a day ranked by closure-risk with action tier + map.
- **📊 Analytics** — historical hour/day/duration/spatial patterns + feature importance.

## 7. Where It Fits in BTP's Stack (verified)
BTP already runs **BATCS** (adaptive *signal* control at 165 junctions) and **ASTRAM** (situational
awareness, alerts, microsimulation — our dataset's source). Neither produces a *prescriptive
police-deployment plan*.

> **BATCS optimizes the signals. ASTRAM sees the city. PredictEvent decides the deployment** —
> turning ASTRAM's event data into an officer count *with a service-level guarantee* and a
> coverage-optimal placement plan. It is a prescriptive module on top of the stack BTP already runs.

## 8. Why This Wins for BTP
- Solves their **actual #1 operational pain** (event manpower planning), not a tech demo.
- **OR-optimized, SLA-guaranteed** deployment — not heuristic rules (the dominant winning pattern in
  deployed systems).
- **Honest, leakage-checked metrics** — survives scrutiny by a technical jury.
- **Plugs into ASTRAM** — extends BTP's own system, deployable today, provided-data only.
- **Closes the learning loop** — every new event retrains the models.

## 9. Roadmap
Live ASTRAM API feed → real clearance timestamps → fuse with traffic-speed / Google/TomTom feeds
→ capacitated city-wide MCLP + IP shift-scheduling (ILS/tabu for fast replanning) → mobile officer app.

---
*Compliance: only the HackerEarth-provided Astram dataset is used. No external data.*
