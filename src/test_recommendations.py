"""Validate cause-aware recommendations on the REAL dataset."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recommend import load_models, event_action

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = load_models()
raw = pd.read_csv(os.path.join(ROOT, "data", "events.csv"), low_memory=False)

# EB historical cause rates on ALL labelled events
lab = raw[raw["requires_road_closure"].notna()].copy()
lab["c"] = lab["requires_road_closure"].astype(str).str.lower().eq("true")
base = lab["c"].mean()
g = lab.groupby("event_cause")["c"].agg(["mean", "size"])
K = 8
ebr = ((g["size"] * g["mean"] + K * base) / (g["size"] + K)).to_dict()

# feature prep
st = pd.to_datetime(raw["start_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
df = raw[st.notna()].copy(); st = st[st.notna()]
df["hour"] = st.dt.hour; df["dow"] = st.dt.dayofweek; df["month"] = st.dt.month
df["is_weekend"] = (df["dow"] >= 5).astype(int)
df["daypart"] = pd.cut(df["hour"], [-1, 6, 11, 16, 23], labels=[0, 1, 2, 3]).astype("float")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["corridor"] = df["corridor"].fillna("Non-corridor")
CAT = ["event_type", "event_cause", "veh_type", "corridor", "zone", "priority"]
NUM = ["hour", "dow", "month", "is_weekend", "daypart", "latitude", "longitude"]
for c in CAT:
    df[c] = df[c].astype("category")

mp = M["closure"].predict_proba(df[CAT + NUM])[:, 1]
recs = []
for p, cause, prio, corr, hr, dw, mo, vt in zip(
        mp, df["event_cause"].astype(str), df["priority"].astype(str), df["corridor"].astype(str),
        df["hour"], df["dow"], df["month"], df["veh_type"].astype(str)):
    risk = max(float(p), ebr.get(cause, base))
    a = event_action(risk, cause, prio, corridor=corr, hour=hr, dow=dw, month=mo, veh_type=vt)
    recs.append({"cause": cause, "corridor": corr, "risk": risk, "RED": a["tier"].startswith("RED"),
                 "barricade": a["barricade"] == "Yes", "diversion": a["diversion_plan"] == "Activate",
                 "crowd_off": a["crowd_officers"], "mult": a["disruption_mult"]})
r = pd.DataFrame(recs)

print(f"base closure rate: {base*100:.1f}%\n")
print("== By cause ==")
print(f"{'cause':18s} {'n':>5} {'risk%':>6} {'RED%':>5} {'barr%':>6} {'div%':>5} {'crowd_off':>9} {'mult':>5}")
for cz in ["vip_movement", "public_event", "protest", "procession", "construction",
           "water_logging", "accident", "vehicle_breakdown"]:
    s = r[r.cause == cz]
    if len(s):
        print(f"{cz:18s} {len(s):5d} {s.risk.mean()*100:6.1f} {s.RED.mean()*100:5.0f} "
              f"{s.barricade.mean()*100:6.0f} {s.diversion.mean()*100:5.0f} {s.crowd_off.mean():9.1f} {s['mult'].mean():5.2f}")

print("\n== Context multiplier by corridor (avg ambient disruption) ==")
top = r["corridor"].value_counts().head(10).index
for cz in top:
    s = r[r.corridor == cz]
    print(f"  {cz:18s} n={len(s):4d}  mult~{s['mult'].mean():.2f}  crowd_off~{s.crowd_off.mean():.1f}")
