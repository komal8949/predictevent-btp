"""
Bengaluru real-world traffic context (researched, 2024-2026).
Encodes domain factors so the recommendation engine reasons like a BTP officer:
corridor chronic congestion, IT-commute intensity, monsoon/rain vulnerability, metro
construction drag, festival season, day/time commute peaks, and heavy-vehicle ban windows.

All factors are transparent data tables + pure functions -> jury-defensible, no black box.
Sources: TomTom Traffic Index 2025; BTP choke-point data; BBMP waterlogging lists;
Namma Metro Blue/Pink/Green line status; BTP heavy-vehicle order (Aug 2024).
"""

# ---------------------------------------------------------------------------
# Per-corridor profile. Keys are matched by prefix so "ORR East 1"/"ORR East 2"
# both resolve to the "ORR East" profile, etc.
#   congestion  : chronic peak congestion severity (0-1)   [TomTom + BTP choke points]
#   it_commute  : IT/employment commute intensity (0-1)    [tech-park density]
#   rain_vuln   : waterlogging/flood vulnerability (0-1)    [BBMP flood spots]
#   construction: metro/flyover construction drag 2024-26 (0-1)
# ---------------------------------------------------------------------------
CORRIDOR_PROFILE = {
    "ORR East":        {"congestion": 1.00, "it_commute": 1.00, "rain_vuln": 0.95, "construction": 0.90},
    "ORR North":       {"congestion": 0.95, "it_commute": 0.85, "rain_vuln": 0.90, "construction": 0.90},
    "Hosur Road":      {"congestion": 0.95, "it_commute": 0.90, "rain_vuln": 0.85, "construction": 0.40},
    "Bellary Road":    {"congestion": 0.90, "it_commute": 0.80, "rain_vuln": 0.60, "construction": 0.80},
    "Old Madras Road": {"congestion": 0.80, "it_commute": 0.75, "rain_vuln": 0.85, "construction": 0.60},
    "Bannerghata":     {"congestion": 0.75, "it_commute": 0.50, "rain_vuln": 0.50, "construction": 0.80},
    "Tumkur Road":     {"congestion": 0.65, "it_commute": 0.60, "rain_vuln": 0.40, "construction": 0.50},
    "Mysore Road":     {"congestion": 0.60, "it_commute": 0.50, "rain_vuln": 0.50, "construction": 0.30},
    "Magadi Road":     {"congestion": 0.45, "it_commute": 0.30, "rain_vuln": 0.30, "construction": 0.45},
}
DEFAULT_CORRIDOR = {"congestion": 0.50, "it_commute": 0.40, "rain_vuln": 0.40, "construction": 0.30}


def corridor_profile(corridor):
    c = str(corridor or "")
    for key, prof in CORRIDOR_PROFILE.items():
        if c.startswith(key) or key.lower() in c.lower():
            return prof
    return DEFAULT_CORRIDOR


def corridor_weight(corridor):
    """Chronic-congestion weight for MCLP / officer scaling: 0.85 (quiet) .. 1.35 (ORR)."""
    return round(0.85 + 0.5 * corridor_profile(corridor)["congestion"], 3)


# ---------------------------------------------------------------------------
# Commute peaks (Bengaluru IT city). Weekday 7-9 AM & 5-9 PM rush; Mon heaviest;
# Fri evening boosted; Sat ~80% offices closed; Sun light.
# ---------------------------------------------------------------------------
def commute_context(hour, dow, corridor=None):
    if hour is None or dow is None:
        return {"factor": 1.0, "label": "—"}
    h, d = int(hour), int(dow)
    morning, evening = (7 <= h <= 9), (17 <= h <= 21)
    it = corridor_profile(corridor)["it_commute"] if corridor is not None else 0.6
    if d <= 4:                                                   # Mon-Fri
        if evening:
            f, lbl = 1.6, "Weekday evening peak (5-9 PM)"
            if d == 4: f, lbl = 1.7, "Friday evening peak"      # Fri leisure outflow
        elif morning:
            f, lbl = 1.5, "Weekday morning peak (7-9 AM)"
        elif 10 <= h <= 16:
            f, lbl = 1.0, "Weekday off-peak"
        else:
            f, lbl = 0.7, "Weekday night"
        if d == 0 and (morning or evening):                     # Monday is worst
            f += 0.1; lbl = "Monday " + lbl.split(" ", 1)[-1]
        # amplify the peak on IT-dense corridors (commute compounds there)
        if morning or evening:
            f *= (0.9 + 0.2 * it)
        return {"factor": round(f, 3), "label": lbl}
    if d == 5:                                                   # Saturday
        if morning or evening:
            return {"factor": 1.1, "label": "Saturday (partial offices + leisure)"}
        return {"factor": 0.85, "label": "Saturday off-peak"}
    if evening:                                                  # Sunday
        return {"factor": 0.85, "label": "Sunday evening (leisure)"}
    return {"factor": 0.65, "label": "Sunday (low commute)"}


# ---------------------------------------------------------------------------
# Monsoon / rain. SW monsoon Jun-Sep, NE monsoon Oct-Nov (worst urban floods),
# pre-monsoon bursts in May. Effective impact scales with the corridor's flood vuln.
# ---------------------------------------------------------------------------
MONTH_RAIN = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.05, 5: 1.2, 6: 1.1,
              7: 1.1, 8: 1.2, 9: 1.4, 10: 1.4, 11: 1.3, 12: 1.05}


def rain_context(month, corridor):
    if month is None:
        return {"factor": 1.0, "label": "—"}
    base = MONTH_RAIN.get(int(month), 1.0)
    vuln = corridor_profile(corridor)["rain_vuln"]
    factor = 1.0 + (base - 1.0) * vuln                          # dry month -> 1.0
    if factor >= 1.25:
        lbl = "Monsoon peak + flood-prone corridor"
    elif factor >= 1.1:
        lbl = "Rain season"
    else:
        lbl = "Dry / low rain risk"
    return {"factor": round(factor, 3), "label": lbl}


# ---------------------------------------------------------------------------
# Festival season (month-level city bump). Karaga (Mar-Apr), Ramzan/Eid (Feb-Mar),
# Ganesha (Aug-Sep), Kadalekai (Nov-Dec), Christmas/NYE (Dec).
# ---------------------------------------------------------------------------
MONTH_FESTIVAL = {2: 1.05, 3: 1.10, 4: 1.10, 8: 1.12, 9: 1.15, 11: 1.05, 12: 1.15}


def festival_context(month):
    if month is None:
        return {"factor": 1.0, "label": "—"}
    f = MONTH_FESTIVAL.get(int(month), 1.0)
    names = {2: "Ramzan", 3: "Karaga/Ramzan", 4: "Karaga", 8: "Ganesha", 9: "Ganesha",
             11: "Kadalekai", 12: "Christmas/NYE"}
    return {"factor": f, "label": (names.get(int(month), "festival") + " season") if f > 1 else "—"}


# ---------------------------------------------------------------------------
# Heavy-vehicle ban (BTP order, Aug 2024): goods vehicles barred in core city
# weekday 7-11 AM & 4-10 PM (Sat 10:30-14:30 & 16:30-21). A heavy vehicle present
# in a ban window is an anomaly that disrupts disproportionately.
# ---------------------------------------------------------------------------
HEAVY_TYPES = {"heavy_vehicle", "truck", "lcv", "trailer", "tanker"}


def heavy_vehicle_flag(hour, dow, veh_type):
    if hour is None or dow is None or str(veh_type).lower() not in HEAVY_TYPES:
        return {"in_ban": False, "label": "—"}
    h, d = int(hour), int(dow)
    weekday_ban = d <= 4 and (7 <= h < 11 or 16 <= h < 22)
    sat_ban = d == 5 and (10 <= h < 15 or 16 <= h < 21)
    if weekday_ban or sat_ban:
        return {"in_ban": True, "label": "Heavy vehicle in ban window (should be off-road)"}
    return {"in_ban": False, "label": "Heavy vehicle (permitted window)"}


# ---------------------------------------------------------------------------
# Master assessment: combine all factors into one disruption multiplier + a
# transparent breakdown (for the UI "why").
# ---------------------------------------------------------------------------
def disruption_context(corridor=None, hour=None, dow=None, month=None, veh_type=None):
    commute = commute_context(hour, dow, corridor)
    rain = rain_context(month, corridor)
    fest = festival_context(month)
    cw = corridor_weight(corridor)
    hv = heavy_vehicle_flag(hour, dow, veh_type)
    construction = corridor_profile(corridor)["construction"]
    constr_factor = 1.0 + 0.15 * construction                  # metro works add up to +15%
    hv_factor = 1.15 if hv["in_ban"] else 1.0
    combined = commute["factor"] * (cw / 1.1) * rain["factor"] * fest["factor"] * constr_factor * hv_factor
    combined = round(min(combined, 3.0), 3)                     # cap to keep sane
    breakdown = []
    if commute["label"] != "—": breakdown.append(f"{commute['label']} ×{commute['factor']:.2f}")
    breakdown.append(f"corridor congestion ×{cw/1.1:.2f}")
    if rain["factor"] > 1.0: breakdown.append(f"{rain['label']} ×{rain['factor']:.2f}")
    if fest["factor"] > 1.0: breakdown.append(f"{fest['label']} ×{fest['factor']:.2f}")
    if construction >= 0.5: breakdown.append(f"metro/construction ×{constr_factor:.2f}")
    if hv["in_ban"]: breakdown.append(f"heavy-vehicle ban breach ×{hv_factor:.2f}")
    return {
        "multiplier": combined,
        "commute": commute, "rain": rain, "festival": fest,
        "corridor_weight": cw, "heavy_vehicle": hv,
        "construction": construction,
        "breakdown": breakdown,
    }


if __name__ == "__main__":
    import json
    tests = [
        ("ORR East 1", 18, 0, 9, "private_car"),     # Mon evening, monsoon, ORR East
        ("Magadi Road", 14, 6, 1, "private_car"),    # Sun afternoon, dry, quiet corridor
        ("Hosur Road", 8, 1, 7, "truck"),            # Tue morning, truck in ban window
        ("Mysore Road", 20, 4, 12, "private_car"),   # Fri night, Dec festive
    ]
    for c, h, d, m, v in tests:
        r = disruption_context(c, h, d, m, v)
        print(f"\n{c}  h={h} dow={d} month={m} veh={v}")
        print(f"  multiplier = {r['multiplier']}")
        print("  " + " | ".join(r["breakdown"]))
