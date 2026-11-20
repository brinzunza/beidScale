"""Table B: definition corpus. Sample BEID strategy DEFINITIONS (zone-engine parameters), run each over the M1 history for all pairs, and store per (definition, year, action) sufficient statistics:
n_filled, sum R, sum 1/risk_pips, sum R^2  (net R at any spread = mean R - spread * mean(1/risk)). Run on the VM, sharded by definition range, multiprocessing over definitions.
Usage: python definitions.py --out DIR --start 0 --end 500 --workers 8 [--max_zones 40000]"""
import sys, os, json, time, argparse, numpy as np, pandas as pd
from multiprocessing import Pool
import data as DD, zones as Z, rewards as RW, actions as A
YEARS = list(range(2005, 2027)); Y0 = 2005
SESS = [(0, 24), (2, 8), (8, 12), (12, 17), (17, 24), (0, 8), (8, 17)]

def sample_params(i, seed=7):
    r = np.random.default_rng(seed * 100003 + i); pick = lambda o: o[int(r.integers(len(o)))]
    s = pick(SESS); return Z.ZoneParams(tf=pick([1, 5, 15, 60]), reset=pick(["ny", "utc"]), pb_mode=pick([0, 1, 2]), confirm_close=pick([0, 1, 1]), flip_close=pick([0, 1, 1]), pb_min=pick([1, 1, 2, 3, 5, 8]),
                                       min_size_atr=pick([0.0, 0.0, 0.1, 0.25, 0.5, 1.0]), max_size_atr=pick([2.0, 5.0, 1e9, 1e9]), min_gap_bars=pick([0, 0, 3, 10]), h0=s[0], h1=s[1])

def param_vector(p):
    return np.array([p.tf, 0 if p.reset == "ny" else 1, p.pb_mode, p.confirm_close, p.flip_close, p.pb_min, p.min_size_atr, min(p.max_size_atr, 20.0), p.min_gap_bars, p.h0, p.h1], np.float32)
PARAM_NAMES = ["tf", "reset_utc", "pb_mode", "confirm_close", "flip_close", "pb_min", "min_size_atr", "max_size_atr", "min_gap_bars", "h0", "h1"]

def aggregate(zt, rw):
    """-> (Y, 420, 4) sums per year: n, sumR, sum_inv_risk, sumR2 over filled valid actions."""
    out = np.zeros((len(YEARS), A.NACT, 4), np.float64); R = rw["R"].astype(np.float32); V = rw["V"] > 0; F = rw["F"] > 0; ent = np.repeat(np.arange(A.ENT), A.NSTOP * A.NTGT * A.NDIR)
    ok = V & F[:, ent]; inv = 1.0 / np.maximum(rw["RISK"].reshape(len(zt), -1)[:, A.risk_slot(np.arange(A.NACT))], 1e-6); inv = np.nan_to_num(inv)
    yr = zt.year.values
    for yi, y in enumerate(YEARS):
        m = yr == y
        if not m.any(): continue
        o = ok[m]; out[yi, :, 0] = o.sum(0); out[yi, :, 1] = np.where(o, R[m], 0).sum(0); out[yi, :, 2] = np.where(o, inv[m], 0).sum(0); out[yi, :, 3] = np.where(o, R[m] ** 2, 0).sum(0)
    return out

_M1 = {}
def _load(pair):
    if pair not in _M1: _M1[pair] = DD.load_m1(pair)
    return _M1[pair]

def run_definition(args):
    i, max_zones, seed = args; p = sample_params(i, seed); tot = np.zeros((len(YEARS), A.NACT, 4)); nz = 0; t0 = time.time()
    for pair in DD.PAIRS:
        m1 = _load(pair); zt = Z.build_zone_table(m1, p); nz += len(zt)
        if len(zt) > max_zones: zt = zt.iloc[np.sort(np.random.default_rng(i).choice(len(zt), max_zones, replace=False))].reset_index(drop=True)
        if len(zt) == 0: continue
        tot += aggregate(zt, RW.build_rewards(m1, zt, DD.PIP[pair])) * (nz_scale(len(zt), max_zones, nz))
    return i, param_vector(p), tot.astype(np.float32), nz, time.time() - t0

def nz_scale(n_used, max_zones, nz_total): return 1.0       # sums are over the sampled zones; counts n are the sample sizes, so means stay unbiased

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=100); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--max_zones", type=int, default=40000); ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); ids = list(range(a.start, a.end)); res = []
    with Pool(a.workers) as pool:
        for i, pv, tot, nz, secs in pool.imap_unordered(run_definition, [(i, a.max_zones, a.seed) for i in ids]):
            res.append((i, pv, tot)); print(f"def {i} zones {nz} {secs:.0f}s", flush=True)
            if len(res) % 20 == 0 or len(res) == len(ids): save(a, res)
    save(a, res)

def save(a, res):
    res = sorted(res, key=lambda x: x[0]); np.savez(os.path.join(a.out, f"corpus_{a.start}_{a.end}.npz"), ids=np.array([r[0] for r in res]), params=np.stack([r[1] for r in res]), stats=np.stack([r[2] for r in res]), param_names=np.array(PARAM_NAMES), years=np.array(YEARS))

if __name__ == "__main__":
    main()
