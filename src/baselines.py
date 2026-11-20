"""Baselines on Table A: best fixed action (chosen on training years), random policy, and an action-conditioned gradient-boosted Q-model (xgboost, GPU when available).
  python baselines.py fixed --table T --test_year 2020 ; python baselines.py xgb --table T --test_year 2020 [--pairs_per_event 24]"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
import evaluate as EV, actions as A, rewardview as RV

def split_rows(z, Y, horizon_days=3):
    t = z.t_ev.values.astype(np.int64); b = np.datetime64(f"{Y}-01-01", "ns").astype(np.int64); tr = np.flatnonzero(t + horizon_days * 86400 * 10 ** 9 < b); te = np.flatnonzero(z.year.values == Y); return tr, te

def fixed(a):
    z = pd.read_parquet(Path(a.table) / "zones.parquet"); tr, te = split_rows(z, a.test_year); out = EV.baselines(a.table, tr[:: max(1, len(tr) // 200000)], te); print(out[out.spread.isin([0.0, 1.3])].round(4).to_string(index=False)); out.to_csv(Path(a.out) / f"baseline_fixed_{a.test_year}.csv", index=False)

def xgb_q(a):
    import xgboost as xgb
    d = Path(a.table); z = pd.read_parquet(d / "zones.parquet"); X = np.load(d / "X.npy", mmap_mode="r"); tr, te = split_rows(z, a.test_year); rng = np.random.default_rng(0); tr = np.sort(rng.choice(tr, min(len(tr), a.max_events), replace=False))
    src = RV.RewardSource(d, z); R, V, F, RISK = src.get(tr); n = len(tr); ent = np.repeat(np.arange(A.ENT), A.NSTOP * A.NTGT * A.NDIR); slot = A.risk_slot(np.arange(A.NACT)); risk = RISK.reshape(n, -1)[:, slot]
    with np.errstate(divide="ignore", invalid="ignore"): net = R - np.where(F[:, ent] > 0, 1.0 / risk, 0.0)
    ok = (V > 0); e_, j_, k_, d_ = zip(*[A.decode(x) for x in range(A.NACT)]); e_, j_, k_, d_ = map(np.array, (e_, j_, k_, d_))
    def feats(rows, aidx):
        zz = z.iloc[rows]; base = np.column_stack([np.asarray(X[rows]), ((zz.top - zz.bot) / zz.atr15).values, zz.ny_hour.values, zz.ztf.values, zz.ztype.values, zz.pair.values]); return np.column_stack([base, e_[aidx], j_[aidx], k_[aidx], d_[aidx]])
    ev, ac = np.nonzero(ok); sel = rng.choice(len(ev), min(len(ev), a.max_pairs), replace=False); ev, ac = ev[sel], ac[sel]; Xtr = feats(tr[ev], ac); ytr = np.nan_to_num(net[ev, ac]); dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    m = xgb.XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7, tree_method="hist", device=dev, n_jobs=8).fit(Xtr, ytr); tv = EV.TableView(d, te); Q = np.zeros((len(te), A.NACT), np.float32)
    for aidx in range(A.NACT): Q[:, aidx] = m.predict(feats(te, np.full(len(te), aidx)))
    Qm = np.where(tv.V > 0, Q, -1e9); act = np.where(Qm.max(1) > 0, Qm.argmax(1), -1); rows = EV.summarize(tv, act, "xgb_action_conditioned"); out = pd.DataFrame(rows); print(out[out.spread.isin([0.0, 1.3])].round(4).to_string(index=False)); out.to_csv(Path(a.out) / f"baseline_xgb_{a.test_year}.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--table", required=True); ap.add_argument("--test_year", type=int, required=True); ap.add_argument("--out", default="."); ap.add_argument("--max_events", type=int, default=300000); ap.add_argument("--max_pairs", type=int, default=6000000)
    a = ap.parse_args(); {"fixed": fixed, "xgb": xgb_q}[a.cmd](a)
