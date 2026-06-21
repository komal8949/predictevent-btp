"""
Resource recommendation engine — turns model outputs into BTP actions.
Maps (predicted load, closure risk, severity) -> manpower, barricades, diversion.
Pure rules on top of ML; transparent & defensible to a jury.
"""
import os
import numpy as np
import pandas as pd
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

# corridor -> rough arterial weight (more lanes / chronic congestion = higher)
CORRIDOR_WEIGHT = {
    "ORR North 1": 1.4, "ORR North 2": 1.4, "ORR East 1": 1.4, "ORR East 2": 1.4,
    "Hosur Road": 1.3, "Bellary Road 1": 1.3, "Bellary Road 2": 1.3,
    "Mysore Road": 1.2, "Tumkur Road": 1.2, "Old Madras Road": 1.2,
    "Bannerghata Road": 1.1, "Magadi Road": 1.1,
}

def manpower_from_load(pred_load, corridor):
    """Daily personnel suggestion for a corridor from forecasted load."""
    w = CORRIDOR_WEIGHT.get(corridor, 1.0)
    base = pred_load * w
    return int(np.clip(np.ceil(base * 0.6) + 1, 1, 30))

def event_action(closure_prob, cause, priority):
    """Per-event playbook from closure risk + cause + priority."""
    barricade = closure_prob >= 0.35 or str(cause).lower() in {
        "vip_movement", "public_event", "procession", "protest"}
    diversion = closure_prob >= 0.5 or str(cause).lower() in {
        "vip_movement", "public_event"}
    if closure_prob >= 0.5 or str(priority).lower() == "high":
        tier = "RED - deploy now"
    elif closure_prob >= 0.25:
        tier = "AMBER - monitor + ready unit"
    else:
        tier = "GREEN - log only"
    return {
        "tier": tier,
        "barricade": "Yes" if barricade else "No",
        "diversion_plan": "Activate" if diversion else "Standby",
        "personnel": int(2 + round(closure_prob * 8)),
    }

def load_models():
    closure = joblib.load(os.path.join(OUT, "model_closure.joblib"))  # dict: model, threshold, cat, num
    return {
        "closure": closure["model"],
        "closure_threshold": closure.get("threshold", 0.5),
        "hotspot": joblib.load(os.path.join(OUT, "model_hotspot.joblib")),
    }

if __name__ == "__main__":
    # quick demo
    print("manpower (ORR North 1, load=8):", manpower_from_load(8, "ORR North 1"))
    print("action (closure_prob=0.6, vip):", event_action(0.6, "vip_movement", "High"))
    print("action (closure_prob=0.1, breakdown):", event_action(0.1, "vehicle_breakdown", "Low"))
