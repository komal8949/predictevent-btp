"""
OR-based resource allocation (upgrade over rule-based engine).
  1) M/M/c (Erlang-C) staffing  -> minimum officers to meet a waiting-time SLA
  2) Maximum Coverage Location Problem (MCLP, PuLP) -> place P units to cover
     maximum RISK x LOAD-weighted demand within a coverage radius
Pure Python, runs on a laptop. Grounded on Astram data (service ~46 min).

References (validated in research): Erlang-C staffing-to-SLA (Miao/Easa 2021);
MCLP+KDE patrol placement (Vlahogianni et al., Annals of OR 2019).
"""
import math
import numpy as np
import pandas as pd

# ----- defaults grounded on the dataset -----
DEFAULT_SERVICE_MIN = 45.0      # median event handling time from Astram
ACTIVE_HOURS = 18.0             # 06:00-24:00 operational window


# ====================== M/M/c (Erlang-C) ======================
def erlang_c(c: int, a: float) -> float:
    """Probability an arriving job must wait (P_wait). a = offered load (Erlangs)."""
    if c <= a:
        return 1.0                      # unstable: demand >= capacity
    # numerically stable recursive Erlang-B, then convert to C
    inv_b = 1.0
    for k in range(1, c + 1):
        inv_b = 1.0 + inv_b * k / a
    eb = 1.0 / inv_b                     # Erlang-B blocking prob
    rho = a / c
    return eb / (1.0 - rho + rho * eb)   # Erlang-C


def wait_minutes(c: int, lam_per_hr: float, mu_per_hr: float) -> float:
    """Expected waiting time in queue Wq, in minutes."""
    if lam_per_hr <= 0:
        return 0.0
    a = lam_per_hr / mu_per_hr
    if c <= a:
        return float("inf")
    pw = erlang_c(c, a)
    wq_hr = pw / (c * mu_per_hr - lam_per_hr)
    return wq_hr * 60.0


def min_officers_for_sla(forecast_load: float, sla_minutes: float = 5.0,
                         service_min: float = DEFAULT_SERVICE_MIN,
                         active_hours: float = ACTIVE_HOURS,
                         max_c: int = 60) -> dict:
    """Minimum officers so expected wait <= SLA, given a daily forecast load."""
    lam = max(forecast_load, 1e-6) / active_hours      # arrivals per hour
    mu = 60.0 / service_min                             # services per hour per officer
    a = lam / mu
    c = max(1, math.ceil(a) + 1)
    while c <= max_c and wait_minutes(c, lam, mu) > sla_minutes:
        c += 1
    wq = wait_minutes(c, lam, mu)
    return {
        "officers": int(c),
        "expected_wait_min": round(wq, 2),
        "utilization": round(a / c, 3),
        "offered_load_erlang": round(a, 3),
        "sla_met": bool(wq <= sla_minutes),
    }


# ====================== MCLP (Maximum Coverage) ======================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def mclp(demand: pd.DataFrame, n_units: int, radius_km: float = 3.0,
         sites: pd.DataFrame = None):
    """
    Maximum Coverage Location Problem.
    demand: DataFrame with columns [name, lat, lon, weight].
    sites : candidate facility locations (default = demand points).
    Returns chosen site names, covered demand names, and coverage %.
    """
    import pulp
    if sites is None:
        sites = demand[["name", "lat", "lon"]].copy()
    D, S = demand.reset_index(drop=True), sites.reset_index(drop=True)
    # coverage matrix a[i][j] = 1 if site j covers demand i
    cover = {}
    for i, di in D.iterrows():
        cover[i] = [j for j, sj in S.iterrows()
                    if haversine_km(di.lat, di.lon, sj.lat, sj.lon) <= radius_km]

    prob = pulp.LpProblem("MCLP", pulp.LpMaximize)
    x = {j: pulp.LpVariable(f"x_{j}", cat="Binary") for j in S.index}      # open site j?
    y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in D.index}      # demand i covered?
    prob += pulp.lpSum(D.loc[i, "weight"] * y[i] for i in D.index)
    for i in D.index:
        prob += y[i] <= pulp.lpSum(x[j] for j in cover[i]) if cover[i] else y[i] == 0
    prob += pulp.lpSum(x[j] for j in S.index) <= n_units
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    chosen = [S.loc[j, "name"] for j in S.index if x[j].value() and x[j].value() > 0.5]
    covered = [D.loc[i, "name"] for i in D.index if y[i].value() and y[i].value() > 0.5]
    tot_w = D["weight"].sum()
    cov_w = D[D["name"].isin(covered)]["weight"].sum()
    return {
        "units_placed": chosen,
        "covered_demand": covered,
        "weighted_coverage_pct": round(100 * cov_w / tot_w, 1) if tot_w else 0.0,
        "n_covered": len(covered), "n_demand": len(D),
    }


def risk_load_weight(load: float, risk: float, alpha: float = 0.6) -> float:
    """Combine forecast load and mean closure-risk into one deployment priority."""
    return alpha * load + (1 - alpha) * risk * 10.0


# ---------------- demo ----------------
if __name__ == "__main__":
    print("M/M/c staffing examples (SLA = 5 min, service = 45 min):")
    for load in [2, 6, 12, 25]:
        print(f"  load={load:>2} ->", min_officers_for_sla(load, 5.0))

    demo = pd.DataFrame({
        "name": ["Mysore Rd", "Bellary 1", "Tumkur Rd", "Hosur Rd", "ORR North 1", "Old Madras Rd"],
        "lat": [12.9620, 13.0147, 13.0320, 12.9196, 13.0267, 12.9792],
        "lon": [77.5665, 77.5848, 77.5347, 77.6215, 77.6307, 77.6287],
        "weight": [9, 7, 5, 6, 4, 3],
    })
    print("\nMCLP: place 3 units, 4km radius:")
    print(" ", mclp(demo, n_units=3, radius_km=4.0))
