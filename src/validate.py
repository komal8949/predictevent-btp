"""
Cross-validation with confidence intervals (standalone, read-only — touches no production file).
Time-series K-fold (expanding window) so every fold trains on the past, tests on the future.
Reports metric mean +/- std for both models -> "R2 0.54 +/- 0.0x" rigor for the jury.
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
N_SPLITS = 5

def ci(vals):
    a = np.array(vals, dtype=float)
    return {"mean": round(a.mean(), 3), "std": round(a.std(), 3),
            "folds": [round(v, 3) for v in a]}

# ============ HOTSPOT FORECASTER CV (R2) ============
df = pd.read_csv(os.path.join(ROOT, "data", "events.csv"), low_memory=False)
st = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
df["date"] = st.dt.normalize(); df["corridor"] = df["corridor"].fillna("Non-corridor")
df = df[df["date"].notna()]
EVENT_CAUSES = {"public_event", "procession", "vip_movement", "protest", "construction"}
df["w"] = 1.0 + df["event_cause"].isin(EVENT_CAUSES) * 1.0 + \
          (df["requires_road_closure"].astype(str).str.lower() == "true") * 1.0
corr = df["corridor"].value_counts().head(20).index.tolist()
d = df[df["corridor"].isin(corr)]
days = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
panel = d.groupby(["corridor", "date"]).agg(load=("w", "sum")).reset_index()
full = pd.MultiIndex.from_product([corr, days], names=["corridor", "date"]).to_frame(index=False)
panel = full.merge(panel, on=["corridor", "date"], how="left").fillna({"load": 0.0}).sort_values(["corridor", "date"])
panel["dow"] = panel["date"].dt.dayofweek; panel["is_weekend"] = (panel["dow"] >= 5).astype(int)
panel["month"] = panel["date"].dt.month; panel["dom"] = panel["date"].dt.day
g = panel.groupby("corridor")["load"]
panel["lag1"] = g.shift(1); panel["lag7"] = g.shift(7)
panel["roll7"] = g.shift(1).rolling(7).mean().reset_index(0, drop=True)
panel["roll14"] = g.shift(1).rolling(14).mean().reset_index(0, drop=True)
panel = panel.dropna(subset=["lag7", "roll7", "roll14"]).sort_values("date").reset_index(drop=True)
panel["corridor_cat"] = panel["corridor"].astype("category")
HFEAT = ["corridor_cat", "dow", "is_weekend", "month", "dom", "lag1", "lag7", "roll7", "roll14"]
hmask = [f == "corridor_cat" for f in HFEAT]

r2s, maes = [], []
for tri, tei in TimeSeriesSplit(n_splits=N_SPLITS).split(panel):
    Xtr, Xte = panel.iloc[tri][HFEAT], panel.iloc[tei][HFEAT]
    ytr, yte = panel.iloc[tri]["load"].values, panel.iloc[tei]["load"].values
    m = HistGradientBoostingRegressor(loss="absolute_error", categorical_features=hmask,
        max_depth=6, learning_rate=0.07, max_iter=500, l2_regularization=1.0, random_state=42)
    m.fit(Xtr, ytr); p = np.clip(m.predict(Xte), 0, None)
    r2s.append(r2_score(yte, p)); maes.append(mean_absolute_error(yte, p))

# ============ CLOSURE CLASSIFIER CV (ROC-AUC) ============
df2 = df.copy()
df2["hour"] = st.dt.hour.values if len(st) == len(df) else pd.to_datetime(df2["start_datetime"], errors="coerce", utc=True).dt.tz_localize(None).dt.hour
s2 = pd.to_datetime(df2["start_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
df2["hour"] = s2.dt.hour; df2["dow"] = s2.dt.dayofweek; df2["month"] = s2.dt.month
df2["is_weekend"] = (df2["dow"] >= 5).astype(int)
df2["daypart"] = pd.cut(df2["hour"], [-1, 6, 11, 16, 23], labels=[0, 1, 2, 3]).astype("float")
df2["latitude"] = pd.to_numeric(df2["latitude"], errors="coerce")
df2["longitude"] = pd.to_numeric(df2["longitude"], errors="coerce")
df2["ord"] = s2.values
CAT = ["event_type", "event_cause", "veh_type", "corridor", "zone", "priority"]
NUM = ["hour", "dow", "month", "is_weekend", "daypart", "latitude", "longitude"]
cmask = [c in CAT for c in (CAT + NUM)]
cdf = df2[df2["requires_road_closure"].notna()].copy()
cdf["y"] = cdf["requires_road_closure"].astype(bool).astype(int)
cdf = cdf[s2[df2["requires_road_closure"].notna()].notna().values].sort_values("ord")
for c in CAT:
    cdf[c] = cdf[c].astype("category")

aucs, praucs = [], []
for tri, tei in TimeSeriesSplit(n_splits=N_SPLITS).split(cdf):
    tr, te = cdf.iloc[tri], cdf.iloc[tei]
    if te["y"].sum() < 3:           # skip folds with too few positives to score
        continue
    m = HistGradientBoostingClassifier(categorical_features=cmask, max_depth=6,
        learning_rate=0.06, max_iter=600, min_samples_leaf=25, l2_regularization=2.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.15, random_state=42)
    m.fit(tr[CAT + NUM], tr["y"].values)
    p = m.predict_proba(te[CAT + NUM])[:, 1]
    aucs.append(roc_auc_score(te["y"].values, p))
    praucs.append(average_precision_score(te["y"].values, p))

res = {
    "n_splits": N_SPLITS,
    "hotspot_R2": ci(r2s),
    "hotspot_MAE": ci(maes),
    "closure_ROC_AUC": ci(aucs),
    "closure_PR_AUC": ci(praucs),
}
json.dump(res, open(os.path.join(OUT, "cv_metrics.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
print(f"\nHotspot R2  = {res['hotspot_R2']['mean']} +/- {res['hotspot_R2']['std']}")
print(f"Closure AUC = {res['closure_ROC_AUC']['mean']} +/- {res['closure_ROC_AUC']['std']}")
