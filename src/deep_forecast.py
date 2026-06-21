"""
EXPERIMENT (standalone, does NOT touch the production GBM or the app).
Right-sized deep spatio-temporal model for the corridor-load forecast, to test whether
deep learning beats the HistGradientBoosting baseline on this small dataset.

Architecture (STAEformer + dual-stream ideas at the correct scale for ~2.7k rows):
  - learnable per-corridor adaptive embedding   (STAEformer's key idea)
  - GRU over a window of past daily loads        (temporal stream)
  - small MLP over target-day calendar features  (static stream)  -> dual-stream fusion
  - concat -> MLP head -> next-day load

Reports BEFORE (GBM) vs AFTER (deep) on an IDENTICAL time-based train/test split.
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

import torch
import torch.nn as nn
SEED = int(os.environ.get("SEED", "42"))
torch.manual_seed(SEED); np.random.seed(SEED)

W = 14            # look-back window (days)
EMB = 8           # corridor embedding dim
HID = 32          # GRU hidden

# ---------------- build the SAME panel as hotspot.py ----------------
df = pd.read_csv(os.path.join(ROOT, "data", "events.csv"), low_memory=False)
st = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
df["date"] = st.dt.normalize()
df["corridor"] = df["corridor"].fillna("Non-corridor")
df = df[df["date"].notna()]
EVENT_CAUSES = {"public_event", "procession", "vip_movement", "protest", "construction"}
df["w"] = 1.0 + df["event_cause"].isin(EVENT_CAUSES) * 1.0 + \
          (df["requires_road_closure"].astype(str).str.lower() == "true") * 1.0

corridors = df["corridor"].value_counts().head(20).index.tolist()
d = df[df["corridor"].isin(corridors)]
days = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
panel = (d.groupby(["corridor", "date"]).agg(load=("w", "sum")).reset_index())
full = pd.MultiIndex.from_product([corridors, days], names=["corridor", "date"]).to_frame(index=False)
panel = full.merge(panel, on=["corridor", "date"], how="left").fillna({"load": 0.0})
panel = panel.sort_values(["corridor", "date"]).reset_index(drop=True)

panel["dow"] = panel["date"].dt.dayofweek
panel["is_weekend"] = (panel["dow"] >= 5).astype(int)
panel["month"] = panel["date"].dt.month
panel["dom"] = panel["date"].dt.day
g = panel.groupby("corridor")["load"]
panel["lag1"] = g.shift(1)
panel["lag7"] = g.shift(7)
panel["roll7"] = g.shift(1).rolling(7).mean().reset_index(0, drop=True)
panel["roll14"] = g.shift(1).rolling(14).mean().reset_index(0, drop=True)
model_rows = panel.dropna(subset=["lag7", "roll7", "roll14"]).copy()   # identical to hotspot.py

# identical time split
cut = model_rows["date"].quantile(0.8)
tr_mask = model_rows["date"] <= cut
te_mask = model_rows["date"] > cut

# ---------------- BEFORE: GBM baseline on these exact rows ----------------
FEAT = ["corridor_cat", "dow", "is_weekend", "month", "dom", "lag1", "lag7", "roll7", "roll14"]
model_rows["corridor_cat"] = model_rows["corridor"].astype("category")
cat_mask = [f == "corridor_cat" for f in FEAT]
gbm = HistGradientBoostingRegressor(loss="absolute_error", categorical_features=cat_mask,
                                    max_depth=6, learning_rate=0.07, max_iter=500,
                                    l2_regularization=1.0, random_state=42)
gbm.fit(model_rows[tr_mask][FEAT], model_rows[tr_mask]["load"].values)
gpred = np.clip(gbm.predict(model_rows[te_mask][FEAT]), 0, None)
yte = model_rows[te_mask]["load"].values
gbm_mae = mean_absolute_error(yte, gpred); gbm_r2 = r2_score(yte, gpred)

# ---------------- build sequences for the deep model ----------------
cidx = {c: i for i, c in enumerate(corridors)}
# fast lookup: corridor -> ordered load array + date->pos
series = {c: panel[panel.corridor == c].sort_values("date").reset_index(drop=True) for c in corridors}
pos = {c: {dt: i for i, dt in enumerate(series[c]["date"])} for c in corridors}

def build(rows):
    Xw, Xc, Xcal, y = [], [], [], []
    for _, r in rows.iterrows():
        c = r["corridor"]; p = pos[c][r["date"]]
        if p < W:
            continue
        window = series[c]["load"].values[p - W:p]          # past W days
        Xw.append(window)
        Xc.append(cidx[c])
        Xcal.append([r["dow"] / 6.0, r["is_weekend"], r["month"] / 12.0, r["dom"] / 31.0])
        y.append(r["load"])
    return (np.array(Xw, dtype=np.float32), np.array(Xc, dtype=np.int64),
            np.array(Xcal, dtype=np.float32), np.array(y, dtype=np.float32))

Xw_tr, Xc_tr, Xcal_tr, y_tr = build(model_rows[tr_mask])
Xw_te, Xc_te, Xcal_te, y_te = build(model_rows[te_mask])

# standardize loads on TRAIN stats
mu, sd = y_tr.mean(), y_tr.std() + 1e-6
Xw_tr_s = (Xw_tr - mu) / sd; Xw_te_s = (Xw_te - mu) / sd
y_tr_s = (y_tr - mu) / sd

# carve val slice (last 15% of train by row order, which is corridor-then-date sorted)
n = len(y_tr); k = int(n * 0.85)
idx = np.argsort(model_rows[tr_mask]["date"].values, kind="stable")  # chronological
tr_i, val_i = idx[:k], idx[k:]

class Net(nn.Module):
    def __init__(self, n_cor):
        super().__init__()
        self.emb = nn.Embedding(n_cor, EMB)
        self.gru = nn.GRU(1, HID, batch_first=True)
        self.cal = nn.Sequential(nn.Linear(4, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(HID + EMB + 16, 32), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, xw, xc, xcal):
        _, h = self.gru(xw.unsqueeze(-1))          # h: (1,B,HID)
        h = h.squeeze(0)
        z = torch.cat([h, self.emb(xc), self.cal(xcal)], dim=1)
        return self.head(z).squeeze(-1)

net = Net(len(corridors))
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.L1Loss()                                # MAE, matches GBM objective

tw = torch.tensor(Xw_tr_s); tc = torch.tensor(Xc_tr); tcal = torch.tensor(Xcal_tr); ty = torch.tensor(y_tr_s)
best_val = 1e9; best_state = None; patience = 0
for epoch in range(400):
    net.train()
    perm = np.random.permutation(tr_i)
    for i in range(0, len(perm), 64):
        b = perm[i:i + 64]
        opt.zero_grad()
        out = net(tw[b], tc[b], tcal[b])
        loss = lossf(out, ty[b])
        loss.backward(); opt.step()
    # validate (in standardized space)
    net.eval()
    with torch.no_grad():
        vpred = net(tw[val_i], tc[val_i], tcal[val_i]).numpy() * sd + mu
    vmae = mean_absolute_error(y_tr[val_i], np.clip(vpred, 0, None))
    if vmae < best_val - 1e-4:
        best_val = vmae; best_state = {k: v.clone() for k, v in net.state_dict().items()}; patience = 0
    else:
        patience += 1
        if patience >= 30:
            break

net.load_state_dict(best_state)
net.eval()
with torch.no_grad():
    dpred = net(torch.tensor(Xw_te_s), torch.tensor(Xc_te), torch.tensor(Xcal_te)).numpy() * sd + mu
dpred = np.clip(dpred, 0, None)
deep_mae = mean_absolute_error(y_te, dpred); deep_r2 = r2_score(y_te, dpred)

# ensemble: average GBM + deep (both on identical test set)
ens = 0.5 * (gpred + dpred)
ens_mae = mean_absolute_error(y_te, ens); ens_r2 = r2_score(y_te, ens)

# ---------------- report ----------------
res = {
    "test_rows": int(len(y_te)),
    "BEFORE_gbm": {"MAE": round(gbm_mae, 3), "R2": round(gbm_r2, 3)},
    "AFTER_deep": {"MAE": round(deep_mae, 3), "R2": round(deep_r2, 3)},
    "ENSEMBLE_gbm+deep": {"MAE": round(ens_mae, 3), "R2": round(ens_r2, 3)},
    "winner": min([("gbm", gbm_mae), ("deep", deep_mae), ("ensemble", ens_mae)], key=lambda t: t[1])[0],
    "deep_mae_delta_pct": round(100 * (deep_mae - gbm_mae) / gbm_mae, 1),
    "ensemble_mae_delta_pct": round(100 * (ens_mae - gbm_mae) / gbm_mae, 1),
}
json.dump(res, open(os.path.join(OUT, "deep_metrics.json"), "w"), indent=2)
print("\n============== BEFORE vs AFTER vs ENSEMBLE (identical test set) ==============")
print(f"  test rows: {res['test_rows']}")
print(f"  GBM (production)     :  MAE {gbm_mae:.3f}   R2 {gbm_r2:.3f}")
print(f"  Deep ST-model        :  MAE {deep_mae:.3f}   R2 {deep_r2:.3f}   ({res['deep_mae_delta_pct']:+.1f}% MAE)")
print(f"  Ensemble (GBM+deep)  :  MAE {ens_mae:.3f}   R2 {ens_r2:.3f}   ({res['ensemble_mae_delta_pct']:+.1f}% MAE)")
print(f"  -> best: {res['winner'].upper()}")
print("=============================================================================")
