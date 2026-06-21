"""
Model training for Theme 2 - Event-Driven Congestion.
TIME-BASED split (train on earlier, test on later) for honest forecasting.
  1) Road-Closure / Barricading classifier (tuned threshold + permutation importance)
  2) Duration regressor + band classifier (secondary; honest data-limited result)
Saves models + metrics + importance chart to outputs/.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             average_precision_score, classification_report,
                             accuracy_score, f1_score, precision_recall_curve)
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyRegressor
import joblib

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "data", "events.csv"), low_memory=False)

# ---- time (timestamps are local IST mislabeled +00 -> use as-is) ----
st = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
cl = pd.to_datetime(df["closed_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
df = df[st.notna()].copy(); cl = cl[st.notna()]; st = st[st.notna()]
df["hour"] = st.dt.hour
df["dow"] = st.dt.dayofweek
df["month"] = st.dt.month
df["is_weekend"] = (df["dow"] >= 5).astype(int)
# coarse part-of-day bucket (0 night,1 morning,2 afternoon,3 evening)
df["daypart"] = pd.cut(df["hour"], [-1, 6, 11, 16, 23], labels=[0, 1, 2, 3]).astype(int)
df["start_ord"] = st.values
dur = (cl - st).dt.total_seconds() / 60.0
df["duration_min"] = dur.where((dur > 0) & (dur < 24 * 60)).values

# ---- features ----
# NOTE: endlat/endlon, authenticated, junction-fill are populated AFTER closure decision
# (target leakage / no signal) -> deliberately excluded. Validated via crosstab + perm-importance.
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

# ---- feature sets (only pre-event / known-at-report-time signals) ----
CAT = ["event_type", "event_cause", "veh_type", "corridor", "zone", "priority"]
NUM = ["hour", "dow", "month", "is_weekend", "daypart", "latitude", "longitude"]
for c in CAT:
    df[c] = df[c].astype("category")
CAT_MASK = [c in CAT for c in (CAT + NUM)]

def time_split(frame, frac=0.8):
    frame = frame.sort_values("start_ord")
    k = int(len(frame) * frac)
    return frame.iloc[:k], frame.iloc[k:]

def make_X(frame):
    return frame[CAT + NUM].copy()

results = {}

# ================= 1) ROAD-CLOSURE / BARRICADING CLASSIFIER =================
clf_df = df[df["requires_road_closure"].notna()].copy()
clf_df["y"] = clf_df["requires_road_closure"].astype(bool).astype(int)
tr, te = time_split(clf_df, 0.8)
# carve validation slice (last 15% of train, by time) for threshold tuning -- no test leakage
trn, val = time_split(tr, 0.85)

clf = HistGradientBoostingClassifier(
    categorical_features=CAT_MASK,
    max_depth=6, learning_rate=0.06, max_iter=600, min_samples_leaf=25,
    l2_regularization=2.0, class_weight="balanced",
    early_stopping=True, validation_fraction=0.15, random_state=42)
# --- probability calibration so the % means what it says ---
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

def ece(y, p, bins=10):
    """Expected Calibration Error."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

# Production model: cross-val sigmoid calibration on the FULL train set (uses all data + calibrated).
cal = CalibratedClassifierCV(clf, method="isotonic", cv=5)   # isotonic preserves the high-risk end
cal.fit(make_X(tr), tr["y"].values)

# Threshold: recall-floor operating point on a time-based val with prefit calibration (no leakage).
# BTP-defensible: choose the highest threshold (best precision) that still catches >=75% of closures.
TARGET_RECALL = 0.75
clf.fit(make_X(trn), trn["y"].values)
cal_thr = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic").fit(make_X(val), val["y"].values)
val_pc = cal_thr.predict_proba(make_X(val))[:, 1]
prec, rec, thr = precision_recall_curve(val["y"].values, val_pc)
ok = rec[:-1] >= TARGET_RECALL                       # align rec with thr (len thr = len rec - 1)
best_thr = float(thr[ok][-1]) if ok.any() else float(thr[0])
best_thr = round(min(max(best_thr, 0.03), 0.5), 3)

# evaluate on TEST: base (uncalibrated, full-train) vs calibrated probabilities
clf.fit(make_X(tr), tr["y"].values)                   # refit base on full train for fair 'before'
yte = te["y"].values
proba_base = clf.predict_proba(make_X(te))[:, 1]      # before calibration
proba = cal.predict_proba(make_X(te))[:, 1]           # after calibration (production)
pred_default = (proba >= 0.5).astype(int)
pred_tuned = (proba >= best_thr).astype(int)
results["road_closure"] = {
    "n_train": int(len(tr)), "n_test": int(len(te)),
    "positive_rate_test": round(float(yte.mean()), 3),
    "ROC_AUC": round(roc_auc_score(yte, proba), 3),
    "PR_AUC": round(average_precision_score(yte, proba), 3),
    "PR_AUC_baseline": round(float(yte.mean()), 3),
    "tuned_threshold": best_thr,
    "recall_pos@tuned": round(pred_tuned[yte == 1].sum() / max(1, (yte == 1).sum()), 3),
    "f1_pos@tuned": round(f1_score(yte, pred_tuned), 3),
    "calibration": {
        "brier_before": round(brier_score_loss(yte, proba_base), 4),
        "brier_after": round(brier_score_loss(yte, proba), 4),
        "ece_before": round(ece(yte, proba_base), 4),
        "ece_after": round(ece(yte, proba), 4),
    },
}
print("\n--- closure report @ tuned threshold", best_thr, "---")
print(classification_report(yte, pred_tuned, digits=3))
print("calibration: Brier", results["road_closure"]["calibration"]["brier_before"],
      "->", results["road_closure"]["calibration"]["brier_after"],
      "| ECE", results["road_closure"]["calibration"]["ece_before"],
      "->", results["road_closure"]["calibration"]["ece_after"])

# reliability diagram (before vs after)
plt.figure(figsize=(6, 6))
for p, lab, c in [(proba_base, "before", "#cbd5e0"), (proba, "after (Platt)", "#2b6cb0")]:
    fp, mp = calibration_curve(yte, p, n_bins=8, strategy="quantile")
    plt.plot(mp, fp, "o-", label=lab, color=c)
plt.plot([0, 1], [0, 1], "--", color="#999")
plt.xlabel("predicted closure probability"); plt.ylabel("observed frequency")
plt.title("Closure-risk calibration (reliability)"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "closure_calibration.png"), dpi=110); plt.close()

# production model is the CALIBRATED classifier (app uses predict_proba -> unchanged interface)
joblib.dump({"model": cal, "threshold": best_thr, "cat": CAT, "num": NUM},
            os.path.join(OUT, "model_closure.joblib"))

# permutation importance (interpretability for jury)
pi = permutation_importance(clf, make_X(te), yte, scoring="roc_auc",
                            n_repeats=8, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=(CAT + NUM)).sort_values()
plt.figure(figsize=(8, 6))
imp.tail(12).plot(kind="barh", color="#2b6cb0")
plt.title("Closure-risk: permutation feature importance (ROC-AUC drop)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "closure_importance.png"), dpi=110); plt.close()
results["road_closure"]["top_features"] = imp.tail(6).round(4).to_dict()

# ================= 2a) DURATION REGRESSION (secondary, MAE-optimized) =================
reg_df = df[df["duration_min"].notna()].copy()
tr, te = time_split(reg_df)
reg = HistGradientBoostingRegressor(
    loss="absolute_error", categorical_features=CAT_MASK,
    max_depth=5, learning_rate=0.06, max_iter=400, l2_regularization=2.0, random_state=42)
reg.fit(make_X(tr), tr["duration_min"].values)
pred = reg.predict(make_X(te)); yte = te["duration_min"].values
base = DummyRegressor(strategy="median").fit(make_X(tr), tr["duration_min"].values).predict(make_X(te))
results["duration_reg"] = {
    "n_test": int(len(te)),
    "MAE_min": round(mean_absolute_error(yte, pred), 2),
    "MAE_baseline_median": round(mean_absolute_error(yte, base), 2),
    "note": "administrative close-time dominates; not reliably learnable",
}
joblib.dump(reg, os.path.join(OUT, "model_duration.joblib"))

# ================= 2b) DURATION BAND =================
def band(m): return 0 if m < 30 else (1 if m <= 120 else 2)
reg_df["band"] = reg_df["duration_min"].apply(band)
trb, teb = time_split(reg_df)
bclf = HistGradientBoostingClassifier(
    categorical_features=CAT_MASK, max_depth=6, learning_rate=0.08, max_iter=400,
    l2_regularization=1.0, class_weight="balanced", random_state=42)
bclf.fit(make_X(trb), trb["band"].values)
bpred = bclf.predict(make_X(teb)); yteb = teb["band"].values
results["duration_band"] = {
    "accuracy": round(accuracy_score(yteb, bpred), 3),
    "macro_f1": round(f1_score(yteb, bpred, average="macro"), 3),
    "baseline_majority_acc": round(float(pd.Series(yteb).value_counts(normalize=True).max()), 3),
}
joblib.dump(bclf, os.path.join(OUT, "model_duration_band.joblib"))

# merge metrics (preserve hotspot if present)
mp = os.path.join(OUT, "metrics.json")
m = json.load(open(mp)) if os.path.exists(mp) else {}
m.update(results)
json.dump(m, open(mp, "w"), indent=2)
print("\n=== METRICS ===")
print(json.dumps(results, indent=2))
print("\nModels + importance chart saved to outputs/.")
