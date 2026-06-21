# BTP Stack Positioning — where PredictEvent fits (verified)

Verified June 2026 against news/primary coverage of BTP's deployed systems.

| System | What it does | Nature |
|---|---|---|
| **BATCS** (Bengaluru Adaptive Traffic Control System) | Adaptive **signal timing** at 165 junctions (CoSiCoSt algorithm); dynamic cycle/phase/offset from camera density; "green waves". 15–20% travel-time cut. | Reactive **signal control** |
| **ASTRAM** (Actionable Intelligence for Sustainable Traffic Mgmt) | Situational awareness from CCTV/ANPR/open data; 15-min congestion alerts; incident reporting; **special-event management**; microsimulation; predictive analytics. *(This is the dataset we use.)* | **Awareness + monitoring** |
| **PredictEvent (ours)** | Converts ASTRAM's event intelligence into a **prescriptive resource plan**: SLA-guaranteed officer count (M/M/c queueing) + coverage-optimal barricade placement (MCLP) + risk×load-weighted deployment. | **Prescriptive OR optimization** |

## The honest gap
- **BATCS optimizes signals, not people.** It cannot tell BTP how many officers to send to a rally or where to place barricade units.
- **ASTRAM tells you *what & where* (awareness), not *how many & a guarantee*.** Its event management is monitoring/reporting/simulation, not OR-based manpower sizing with a service-level guarantee.
- **PredictEvent is the prescriptive layer that plugs into ASTRAM's own data** — it does not compete with ASTRAM, it extends it. Output: "Deploy 5 officers on Hosur Road for this event so expected response wait ≤ 5 min; place 4 barricade units at these coverage-optimal corridors covering 80% of risk×load."

## Positioning line for the jury
> "BATCS optimizes the signals. ASTRAM sees the city. PredictEvent decides the deployment —
> turning ASTRAM's event data into an officer count with a service-level guarantee and a
> coverage-optimal placement plan. It is a prescriptive module on top of the stack BTP already runs."

Sources: Deccan Herald & InsightsOnIndia (ASTRAM); Vartha Bharati & Deccan Herald (BATCS), 2025–2026.
