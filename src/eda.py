"""
EDA for BTP Event-Driven Congestion (Theme 2).
Generates insight tables + figures into outputs/.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "data", "events.csv"), low_memory=False)
print("rows, cols:", df.shape)

# --- datetime parsing ---
for c in ["start_datetime", "closed_datetime", "created_date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

# Bengaluru local time
df["start_local"] = df["start_datetime"].dt.tz_convert("Asia/Kolkata")
df["hour"] = df["start_local"].dt.hour
df["dow"] = df["start_local"].dt.dayofweek          # 0=Mon
df["dow_name"] = df["start_local"].dt.day_name()
df["date"] = df["start_local"].dt.date

# --- duration (minutes), capped to sane window ---
dur = (df["closed_datetime"] - df["start_datetime"]).dt.total_seconds() / 60.0
df["duration_min"] = dur.where((dur > 0) & (dur < 24 * 60))

# --- keep events inside Bengaluru bbox ---
df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")

# is this an "event-driven" cause (the Theme-2 hotspots) vs background incidents
EVENT_CAUSES = {"public_event", "procession", "vip_movement", "protest", "construction"}
df["is_event_driven"] = df["event_cause"].isin(EVENT_CAUSES)

geo = df[(df.lat.between(12.7, 13.3)) & (df.lon.between(77.2, 77.9))]

# ---------- console insight tables ----------
def show(title, s):
    print(f"\n=== {title} ===")
    print(s)

show("event_type", df.event_type.value_counts(dropna=False))
show("event_cause", df.event_cause.value_counts(dropna=False).head(15))
show("event-driven share", df.is_event_driven.value_counts())
show("road closure", df.requires_road_closure.value_counts(dropna=False))
show("priority", df.priority.value_counts(dropna=False))
show("duration stats (min)", df.duration_min.describe().round(1))

show("median duration by cause (min)",
     df.groupby("event_cause").duration_min.median().sort_values(ascending=False).round(1).head(15))
show("closure rate by cause",
     (df.groupby("event_cause").requires_road_closure.mean()*100).sort_values(ascending=False).round(1).head(15))
show("top corridors by event count", df.corridor.value_counts().head(12))
show("events by hour", df.groupby("hour").size())

# ---------- figures ----------
plt.figure(figsize=(9,4))
df.groupby("hour").size().plot(kind="bar", color="#2b6cb0")
plt.title("Events by hour of day (IST)"); plt.xlabel("hour"); plt.ylabel("events")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"events_by_hour.png"), dpi=110); plt.close()

plt.figure(figsize=(9,4))
order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
df.dow_name.value_counts().reindex(order).plot(kind="bar", color="#2f855a")
plt.title("Events by day of week"); plt.ylabel("events")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"events_by_dow.png"), dpi=110); plt.close()

plt.figure(figsize=(9,4))
df.duration_min.clip(upper=600).hist(bins=40, color="#dd6b20")
plt.title("Event duration distribution (min, capped 600)"); plt.xlabel("minutes")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"duration_hist.png"), dpi=110); plt.close()

# spatial scatter colored by event-driven
plt.figure(figsize=(7,7))
bg = geo[~geo.is_event_driven]
ev = geo[geo.is_event_driven]
plt.scatter(bg.lon, bg.lat, s=3, c="#cbd5e0", label="background incidents")
plt.scatter(ev.lon, ev.lat, s=10, c="#c53030", label="event-driven")
plt.legend(); plt.title("Bengaluru events (red = event-driven hotspots)")
plt.xlabel("lon"); plt.ylabel("lat")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"spatial_scatter.png"), dpi=110); plt.close()

# daily timeline
plt.figure(figsize=(11,4))
df.groupby("date").size().plot(color="#6b46c1")
plt.title("Daily event volume"); plt.ylabel("events/day")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"daily_timeline.png"), dpi=110); plt.close()

print("\nFigures written to outputs/. EDA done.")
print("Date span:", df.start_local.min(), "->", df.start_local.max())
print("Geo rows kept:", len(geo), "of", len(df))
