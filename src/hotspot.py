"""
Spatio-temporal HOTSPOT LOAD forecaster (the headline model).
Aggregate events to (corridor x day) panel, build calendar + lag features,
forecast daily event load per corridor. Honest time-based split.
This load directly drives MANPOWER allocation in the demo.
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

df = pd.read_csv(os.path.join(ROOT, "data", "events.csv"), low_memory=False)
st = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
df["date"] = st.dt.normalize()
df["corridor"] = df["corridor"].fillna("Non-corridor")
df = df[df["date"].notna()]

# weight: event-driven / closure events count more toward congestion load
EVENT_CAUSES = {"public_event", "procession", "vip_movement", "protest", "construction"}
df["w"] = 1.0 + df["event_cause"].isin(EVENT_CAUSES) * 1.0 + \
          (df["requires_road_closure"].astype(str).str.lower() == "true") * 1.0

# ---- build complete corridor x day panel ----
corridors = df["corridor"].value_counts().head(20).index.tolist()
d = df[df["corridor"].isin(corridors)]
days = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
panel = (d.groupby(["corridor", "date"])
           .agg(load=("w", "sum"), n=("id", "count")).reset_index())
full = (pd.MultiIndex.from_product([corridors, days], names=["corridor", "date"])
          .to_frame(index=False))
panel = full.merge(panel, on=["corridor", "date"], how="left").fillna({"load": 0, "n": 0})
panel = panel.sort_values(["corridor", "date"])

# ---- calendar + lag/rolling features (per corridor) ----
panel["dow"] = panel["date"].dt.dayofweek
panel["is_weekend"] = (panel["dow"] >= 5).astype(int)
panel["month"] = panel["date"].dt.month
panel["dom"] = panel["date"].dt.day
g = panel.groupby("corridor")["load"]
panel["lag1"] = g.shift(1)
panel["lag7"] = g.shift(7)
panel["roll7"] = g.shift(1).rolling(7).mean().reset_index(0, drop=True)
panel["roll14"] = g.shift(1).rolling(14).mean().reset_index(0, drop=True)
panel["corridor_cat"] = panel["corridor"].astype("category")
panel = panel.dropna(subset=["lag7", "roll7", "roll14"])

FEAT = ["corridor_cat", "dow", "is_weekend", "month", "dom",
        "lag1", "lag7", "roll7", "roll14"]
cat_mask = [f == "corridor_cat" for f in FEAT]

# ---- time split ----
cut = panel["date"].quantile(0.8)
tr = panel[panel["date"] <= cut]; te = panel[panel["date"] > cut]
Xtr, ytr = tr[FEAT], tr["load"].values
Xte, yte = te[FEAT], te["load"].values

model = HistGradientBoostingRegressor(
    loss="absolute_error", categorical_features=cat_mask,
    max_depth=6, learning_rate=0.07, max_iter=500, l2_regularization=1.0,
    random_state=42)
model.fit(Xtr, ytr)
pred = np.clip(model.predict(Xte), 0, None)

# baselines: seasonal-naive (lag7) and rolling-7 mean
base_lag7 = te["lag7"].values
base_roll = te["roll7"].values
res = {
    "n_train": int(len(tr)), "n_test": int(len(te)),
    "corridors": len(corridors),
    "MAE_model": round(mean_absolute_error(yte, pred), 3),
    "MAE_seasonal_naive_lag7": round(mean_absolute_error(yte, base_lag7), 3),
    "MAE_roll7_mean": round(mean_absolute_error(yte, base_roll), 3),
    "R2_model": round(r2_score(yte, pred), 3),
    "R2_seasonal_naive": round(r2_score(yte, base_lag7), 3),
}
joblib.dump(model, os.path.join(OUT, "model_hotspot.joblib"))
panel.to_parquet(os.path.join(OUT, "panel.parquet")) if False else \
    panel.to_csv(os.path.join(OUT, "panel.csv"), index=False)

print(json.dumps(res, indent=2))
# merge into metrics.json
mp = os.path.join(OUT, "metrics.json")
m = json.load(open(mp)) if os.path.exists(mp) else {}
m["hotspot_load"] = res
json.dump(m, open(mp, "w"), indent=2)

# top forecasted hotspots on the last test day (for demo sanity)
last = te[te["date"] == te["date"].max()].copy()
last["pred_load"] = np.clip(model.predict(last[FEAT]), 0, None)
print("\nTop forecasted hotspot corridors on", te["date"].max().date(), ":")
print(last.sort_values("pred_load", ascending=False)[["corridor", "pred_load", "load"]].head(8).to_string(index=False))
