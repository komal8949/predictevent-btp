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

# Per-cause operational profile (encodes BTP ground reality):
#   crowd     = pedestrian/crowd-control intensity -> drives crowd-control officers (bandobast)
#   barricade = barricading typically required
#   diversion = route diversion typically required
#   why       = one-line operational rationale
CAUSE_PROFILE = {
    # crowd-heavy public gatherings: big footfall -> many officers + barricade + diversion
    "public_event": {"crowd": 1.00, "barricade": True,  "diversion": True,
                     "why": "large crowd + parking spillover; pedestrians on carriageway"},
    "protest":      {"crowd": 0.90, "barricade": True,  "diversion": True,
                     "why": "pedestrian crowd may occupy the road; pre-empt with diversion"},
    "procession":   {"crowd": 0.80, "barricade": True,  "diversion": True,
                     "why": "moving pedestrian crowd along the carriageway"},
    # VIP: smaller crowd but road is cleared for passage -> barricade + escort diversion
    "vip_movement": {"crowd": 0.35, "barricade": True,  "diversion": True,
                     "why": "carriageway cleared for the convoy; barricade + diversion mandatory"},
    # infrastructure blockages: lane/stretch lost -> diversion, barricade to seal it
    "construction": {"crowd": 0.25, "barricade": True,  "diversion": True,
                     "why": "lane/road closure; seal and divert"},
    "tree_fall":    {"crowd": 0.15, "barricade": True,  "diversion": True,
                     "why": "stretch blocked until cleared"},
    "water_logging":{"crowd": 0.20, "barricade": False, "diversion": True,
                     "why": "impassable stretch; divert around it"},
    # incidents: localized, fewer officers, divert only if severe
    "accident":     {"crowd": 0.30, "barricade": False, "diversion": False,
                     "why": "localized; clear quickly, divert only if it spills over"},
    "vehicle_breakdown": {"crowd": 0.10, "barricade": False, "diversion": False,
                          "why": "single-vehicle; tow + manage lane"},
    "pot_holes":    {"crowd": 0.10, "barricade": False, "diversion": False, "why": "road condition"},
    "congestion":   {"crowd": 0.30, "barricade": False, "diversion": False, "why": "flow management"},
}
DEFAULT_PROFILE = {"crowd": 0.30, "barricade": False, "diversion": False, "why": "general incident"}
CROWD_SCALE = 12          # officers for a max-crowd (public_event) event before priority weighting


def event_action(closure_prob, cause, priority, corridor=None, hour=None, dow=None,
                 month=None, veh_type=None, crowd_scale=CROWD_SCALE):
    """Per-event playbook: cause-aware + full Bengaluru-context barricade / diversion / officers + tier.
    Combines event cause with corridor congestion, commute peak, monsoon, festival season, metro
    construction and heavy-vehicle ban breaches (see bengaluru_context.disruption_context)."""
    from bengaluru_context import disruption_context
    p = CAUSE_PROFILE.get(str(cause).lower(), DEFAULT_PROFILE)
    hi_prio = str(priority).lower() == "high"
    ctx = disruption_context(corridor, hour, dow, month, veh_type)
    mult = ctx["multiplier"]
    heavy_disruptive = mult >= 1.6                          # significantly worse-than-normal conditions
    # barricade / diversion: triggered by the cause OR by high model-estimated closure risk
    barricade = p["barricade"] or closure_prob >= 0.35
    diversion = p["diversion"] or closure_prob >= 0.5
    # crowd-control officers scale with cause intensity, priority, AND ambient disruption
    # (cap the multiplier's effect on headcount so worst-case stays operationally sane)
    crowd_officers = int(np.ceil(p["crowd"] * crowd_scale * (1.3 if hi_prio else 1.0) * min(mult, 2.2)))
    # tier: RED for severe events OR a material event under heavy ambient disruption
    material = closure_prob >= 0.25 or p["crowd"] >= 0.35
    if closure_prob >= 0.45 or p["crowd"] >= 0.8 or (hi_prio and material) or (heavy_disruptive and material):
        tier = "RED - deploy now"
    elif material or hi_prio or heavy_disruptive:
        tier = "AMBER - monitor + ready unit"
    else:
        tier = "GREEN - log only"
    return {
        "tier": tier,
        "barricade": "Yes" if barricade else "No",
        "diversion_plan": "Activate" if diversion else "Standby",
        "crowd_officers": crowd_officers,
        "disruption_mult": mult,
        "context_breakdown": ctx["breakdown"],
        "commute_label": ctx["commute"]["label"],
        "heavy_vehicle": ctx["heavy_vehicle"]["in_ban"],
        "why": p["why"],
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
