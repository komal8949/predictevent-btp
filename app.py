"""
PredictEvent - BTP Event Impact & Resource Engine (Theme 2 demo).
Run:  streamlit run app.py
"""
import os, sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from recommend import load_models, event_action, CORRIDOR_WEIGHT
from optimize import min_officers_for_sla, mclp, risk_load_weight

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")

st.set_page_config(page_title="PredictEvent - BTP", layout="wide", page_icon="🚦")

CAT = ["event_type", "event_cause", "veh_type", "corridor", "zone", "priority"]
NUM = ["hour", "dow", "month", "is_weekend", "daypart", "latitude", "longitude"]

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(ROOT, "data", "events.csv"), low_memory=False)
    st_ = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df[st_.notna()].copy(); st_ = st_[st_.notna()]   # drop rows with no start time
    df["dt"] = st_
    df["date"] = st_.dt.normalize()
    df["hour"] = st_.dt.hour; df["dow"] = st_.dt.dayofweek
    df["month"] = st_.dt.month; df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["daypart"] = pd.cut(df["hour"], [-1, 6, 11, 16, 23], labels=[0, 1, 2, 3]).astype(int)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    for c in CAT:
        df[c] = df[c].astype("category")
    return df.dropna(subset=["dt"])

@st.cache_resource
def models():
    return load_models()

df = load_data()
M = models()
panel = pd.read_csv(os.path.join(OUT, "panel.csv"), parse_dates=["date"])
BASE_RATE = float(df["requires_road_closure"].astype(str).str.lower().eq("true").mean())  # ~0.08

@st.cache_data
def cause_eb_rates(K=8):
    """Empirical-Bayes historical closure rate per event cause (shrunk toward base by sample size).
    Used as a recall-favoring safety floor so rare high-impact causes (e.g. VIP movement, only ~20
    samples) are never under-flagged just because the ML model lacks training examples for them.
    Computed on ALL labelled events (closure label needs no start time) so rare causes keep signal."""
    raw = pd.read_csv(os.path.join(ROOT, "data", "events.csv"), low_memory=False)
    raw = raw[raw["requires_road_closure"].notna()]
    c = raw["requires_road_closure"].astype(str).str.lower().eq("true")
    base = c.mean()
    g = c.groupby(raw["event_cause"]).agg(["mean", "size"])
    eb = (g["size"] * g["mean"] + K * base) / (g["size"] + K)
    return eb.to_dict()

CAUSE_EB = cause_eb_rates()

def event_risk(model_prob, cause):
    """Final displayed closure risk = max(calibrated model prob, historical cause rate)."""
    return max(float(model_prob), float(CAUSE_EB.get(cause, BASE_RATE)))

@st.cache_data
def corridor_stats():
    """Centroid + mean closure-risk per corridor -> drives MCLP weights."""
    g = df[df["latitude"].between(12.7, 13.3) & df["longitude"].between(77.2, 77.9)].copy()
    g["risk"] = M["closure"].predict_proba(g[CAT + NUM])[:, 1]
    s = g.groupby("corridor", observed=True).agg(
        lat=("latitude", "median"), lon=("longitude", "median"),
        mean_risk=("risk", "mean"), n=("dt", "size")).reset_index()
    return s

CSTATS = corridor_stats()

# ---- sidebar: operational SLA parameters (used by the OR engine) ----
st.sidebar.header("⚙️ Deployment SLA")
SLA_MIN = st.sidebar.slider("Max acceptable wait (min)", 2, 20, 5,
                            help="Erlang-C staffing target: expected officer response wait.")
SERVICE_MIN = st.sidebar.slider("Mean event handling time (min)", 20, 90, 45,
                                help="Service time per event (Astram median ~46 min).")
st.sidebar.caption("Staffing = M/M/c queueing (Erlang-C). "
                   "Placement = Maximum Coverage Location Problem (MCLP).")

st.title("🚦 PredictEvent - Event-Driven Congestion Intelligence for BTP")
st.caption("Theme 2 | Forecast impact -> OR-optimized manpower, barricading & diversion | "
           f"Data: {df['date'].min().date()} to {df['date'].max().date()}, {len(df):,} events")

tab0, tab1, tab2, tab3 = st.tabs(["🎯 Event Simulator",
                            "📍 Hotspot Forecast + Manpower",
                            "🚨 Live Event Triage", "📊 Analytics"])

# ---------------- TAB 0: EVENT SIMULATOR (finale showpiece) ----------------
with tab0:
    st.subheader("Plan a planned/unplanned event -> instant deployment recommendation")
    st.caption("e.g. an IPL match at Chinnaswamy, a rally on MG Road, a procession in Chamrajpet.")

    def opts(col, n=15):
        return sorted([str(x) for x in df[col].dropna().unique()])[:n] if df[col].dtype.name != "category" \
            else sorted([str(x) for x in df[col].cat.categories])

    cc = st.columns(3)
    sim_cause = cc[0].selectbox("Event cause", ["public_event", "procession", "vip_movement",
        "protest", "construction", "accident", "vehicle_breakdown", "water_logging"])
    sim_corr = cc[1].selectbox("Corridor", sorted([str(x) for x in df["corridor"].cat.categories]))
    sim_type = cc[2].selectbox("Type", ["planned", "unplanned"])

    cc2 = st.columns(3)
    sim_zone = cc2[0].selectbox("Zone", ["Central Zone 1", "Central Zone 2", "East Zone 1",
        "East Zone 2", "West Zone 1", "West Zone 2", "North Zone 1", "North Zone 2",
        "South Zone 1", "South Zone 2"])
    sim_prio = cc2[1].selectbox("Priority", ["High", "Low"])
    sim_veh = cc2[2].selectbox("Vehicle type", ["others", "private_car", "bmtc_bus",
        "heavy_vehicle", "truck", "private_bus", "ksrtc_bus", "lcv", "taxi"])

    cc3 = st.columns(3)
    sim_hour = cc3[0].slider("Hour of day", 0, 23, 19)
    sim_dow = cc3[1].selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"], index=5)
    sim_lat = cc3[2].number_input("Latitude", value=12.9788, format="%.4f")
    sim_lon = st.number_input("Longitude", value=77.5996, format="%.4f")

    dow_idx = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(sim_dow)
    row = pd.DataFrame([{
        "event_type": sim_type, "event_cause": sim_cause, "veh_type": sim_veh,
        "corridor": sim_corr, "zone": sim_zone, "priority": sim_prio,
        "hour": sim_hour, "dow": dow_idx, "month": 4,
        "is_weekend": int(dow_idx >= 5),
        "daypart": int(np.clip(np.digitize(sim_hour, [6, 11, 16]), 0, 3)),
        "latitude": sim_lat, "longitude": sim_lon,
    }])
    for c in CAT:
        row[c] = row[c].astype("category")

    cc4 = st.columns(2)
    exp_incidents = cc4[0].slider("Expected incidents during event", 1, 30, 8,
        help="Sub-incidents (breakdowns, jams, scuffles) the event is expected to generate.")
    window_h = cc4[1].slider("Event window (hours)", 1, 12, 3)

    model_prob = float(M["closure"].predict_proba(row[CAT + NUM])[:, 1][0])
    prob = event_risk(model_prob, sim_cause)            # max(model, historical cause rate)
    act = event_action(prob, sim_cause, sim_prio)
    # OR staffing: size officers so expected wait <= SLA over the concentrated event window
    st_res = min_officers_for_sla(exp_incidents, SLA_MIN, SERVICE_MIN, active_hours=window_h)

    incident_officers = st_res["officers"]          # M/M/c incident-response
    crowd_officers = act["crowd_officers"]           # cause-driven crowd control (bandobast)
    total_officers = incident_officers + crowd_officers

    st.markdown("### Recommendation")
    m = st.columns(4)
    color = "🔴" if act["tier"].startswith("RED") else ("🟠" if act["tier"].startswith("AMBER") else "🟢")
    mult = prob / BASE_RATE if BASE_RATE else 0
    m[0].metric("Closure / barricade risk", f"{prob*100:.0f}%",
                delta=f"{mult:.1f}× city avg ({BASE_RATE*100:.0f}%)", delta_color="inverse",
                help="max(calibrated model prob, historical cause rate). Shown vs citywide base rate.")
    m[1].metric("Action tier", f"{color} {act['tier'].split(' - ')[0]}")
    m[2].metric("Total officers", total_officers,
                help=f"{crowd_officers} crowd-control (cause-driven) + {incident_officers} incident-response (M/M/c)")
    m[3].metric("Incident response wait", f"{st_res['expected_wait_min']} min")

    b = st.columns(3)
    b[0].info(f"**Barricading:** {act['barricade']}")
    b[1].info(f"**Route diversion:** {act['diversion_plan']}")
    sla_txt = "✅ SLA met" if st_res["sla_met"] else "⚠️ SLA NOT met"
    b[2].info(f"**Officers:** {crowd_officers} crowd + {incident_officers} response  \n{sla_txt}")
    st.success(f"**Deployment:** {act['tier'].split(' - ')[1].capitalize()}  ·  "
               f"**Why:** {act['why']}")
    st.map(row.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
    st.caption(f"Risk = max(calibrated model {model_prob*100:.0f}%, historical {sim_cause} rate "
               f"{CAUSE_EB.get(sim_cause, BASE_RATE)*100:.0f}%) — a recall-favoring safety floor so rare "
               "high-impact events (VIP) are never under-flagged. Officers from M/M/c queueing (Erlang-C) "
               "tied to the wait SLA; barricade/diversion from the playbook.")

# ---------------- TAB 1: hotspot forecast + OR allocation ----------------
with tab1:
    st.subheader("Daily corridor load forecast -> SLA-based staffing + coverage-optimal placement")
    days = sorted(panel["date"].dt.date.unique())
    sel = st.select_slider("Forecast day", options=days, value=days[-1])
    snap = panel[panel["date"].dt.date == sel].copy()
    if snap.empty:
        st.info("No panel row for this day."); st.stop()
    feat = ["corridor_cat", "dow", "is_weekend", "month", "dom",
            "lag1", "lag7", "roll7", "roll14"]
    snap["corridor_cat"] = snap["corridor"].astype("category")
    snap["pred_load"] = np.clip(M["hotspot"].predict(snap[feat]), 0, None)
    # merge corridor risk (mean closure prob) for risk x load weighting
    snap = snap.merge(CSTATS[["corridor", "mean_risk", "lat", "lon"]], on="corridor", how="left")
    snap["mean_risk"] = snap["mean_risk"].fillna(0.1)
    # M/M/c (Erlang-C) staffing per corridor against the SLA
    staff = snap["pred_load"].apply(lambda l: min_officers_for_sla(l, SLA_MIN, SERVICE_MIN))
    snap["officers"] = [s["officers"] for s in staff]
    snap["exp_wait_min"] = [s["expected_wait_min"] for s in staff]
    snap["priority_score"] = [risk_load_weight(l, r) for l, r in zip(snap["pred_load"], snap["mean_risk"])]
    snap = snap.sort_values("priority_score", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total forecasted load", f"{snap['pred_load'].sum():.0f}")
    c2.metric("Officers (SLA-sized)", int(snap["officers"].sum()))
    c3.metric(f"SLA met (≤{SLA_MIN}min)", f"{int(snap['exp_wait_min'].le(SLA_MIN).sum())}/{len(snap)}")
    c4.metric("High-load corridors (>3)", int((snap["pred_load"] > 3).sum()))

    st.bar_chart(snap.set_index("corridor")["priority_score"].head(12))
    st.dataframe(
        snap[["corridor", "pred_load", "mean_risk", "priority_score",
              "officers", "exp_wait_min"]].rename(columns={
            "pred_load": "forecast_load", "mean_risk": "closure_risk",
            "priority_score": "risk×load", "exp_wait_min": "exp_wait_min"}).round(2),
        use_container_width=True, height=300)

    st.markdown("#### 🛡️ Barricade-unit placement (Maximum Coverage Location Problem)")
    cc = st.columns(2)
    n_units = cc[0].slider("Mobile barricade/enforcement units available", 1, 10, 4)
    radius = cc[1].slider("Coverage radius (km)", 1.0, 8.0, 3.0, 0.5)
    dem = snap.dropna(subset=["lat", "lon"]).copy()
    dem = dem.rename(columns={"corridor": "name", "priority_score": "weight"})[
        ["name", "lat", "lon", "weight"]]
    if len(dem) >= 1 and dem["weight"].sum() > 0:
        res = mclp(dem, n_units=n_units, radius_km=radius)
        m1, m2 = st.columns(2)
        m1.metric("Weighted demand covered", f"{res['weighted_coverage_pct']}%")
        m2.metric("Corridors covered", f"{res['n_covered']}/{res['n_demand']}")
        st.success("**Place units at:** " + ", ".join(res["units_placed"]))
        placed = dem[dem["name"].isin(res["units_placed"])]
        st.map(placed.rename(columns={"lat": "lat", "lon": "lon"})[["lat", "lon"]])
    st.caption("Staffing via M/M/c queueing (Erlang-C) to meet the wait SLA; "
               "placement via MCLP maximizing risk×load-weighted coverage (PuLP/CBC).")

# ---------------- TAB 2: event triage ----------------
with tab2:
    st.subheader("Per-event closure risk + recommended action")
    day2 = st.select_slider("Event day", options=sorted(df["date"].dt.date.unique()),
                            value=sorted(df["date"].dt.date.unique())[-2], key="d2")
    ev = df[df["date"].dt.date == day2].copy()
    if len(ev):
        mp_ = M["closure"].predict_proba(ev[CAT + NUM])[:, 1]
        ev["closure_prob"] = [event_risk(p, c) for p, c in zip(mp_, ev["event_cause"])]
        acts = ev.apply(lambda r: event_action(r["closure_prob"], r["event_cause"], r["priority"]), axis=1)
        ev["tier"] = [a["tier"] for a in acts]
        ev["barricade"] = [a["barricade"] for a in acts]
        ev["diversion"] = [a["diversion_plan"] for a in acts]
        ev["officers"] = [a["crowd_officers"] + 1 for a in acts]   # crowd-control + 1 response unit
        ev = ev.sort_values("closure_prob", ascending=False)

        red = (ev["tier"].str.startswith("RED")).sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events", len(ev)); c2.metric("RED (deploy now)", int(red))
        c3.metric("Barricades advised", int((ev["barricade"] == "Yes").sum()))
        c4.metric("Officers (total)", int(ev["officers"].sum()))

        show = ev[["dt", "corridor", "event_cause", "priority", "closure_prob",
                   "tier", "barricade", "diversion", "officers"]].copy()
        show["closure_prob"] = (show["closure_prob"] * 100).round(1)
        st.dataframe(show.rename(columns={"closure_prob": "closure_risk_%"}),
                     use_container_width=True, height=340)
        mp = ev.dropna(subset=["latitude", "longitude"])
        mp = mp[mp["latitude"].between(12.7, 13.3) & mp["longitude"].between(77.2, 77.9)]
        st.map(mp.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
    else:
        st.info("No events on this day.")

# ---------------- TAB 3: analytics ----------------
with tab3:
    st.subheader("Historical patterns")
    colA, colB = st.columns(2)
    for col, img in zip([colA, colB, colA, colB, colA],
                        ["events_by_hour.png", "events_by_dow.png",
                         "duration_hist.png", "spatial_scatter.png", "daily_timeline.png"]):
        p = os.path.join(OUT, img)
        if os.path.exists(p):
            col.image(p, use_container_width=True)
