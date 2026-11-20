"""Build Table A (event-action table) shards on the VM: zones for (pair, tf, reset) -> state metadata + reward tensors. Merge shards into one memory-mappable table.
Usage: python build_table.py shard --pair EURUSD --tf 5 --reset ny --out DIR | python build_table.py merge --out DIR"""
import sys, os, json, argparse, numpy as np, pandas as pd
from pathlib import Path
import data as DD, zones as Z, rewards as RW, actions as A, windows as W, features as FT
DAY_EXCL = ("pair", "date", "atr", "c", "o", "h", "l", "year", "next_date")

def days_table():
    f = DD.data_dir() / "days.parquet"
    d = pd.read_parquet(f); cols = [c for c in d.columns if c not in DAY_EXCL and not c.startswith(("y_", "mat_"))]; return d, cols

# Mirroring of the 46 day factors to the long perspective of the trade (sell zones). Flip: x -> -x; complement: x -> 1 - x; swap: exchange two columns; swapneg: exchange and negate.
DAY_FLIP = ("ret", "gap", "asia_ret", "lon_ret", "ny_ret", "slope5", "slope20", "slope60", "dma20", "dma50", "dma200", "ret5", "sA_trend", "xp1", "xp2", "xp3")
DAY_COMPLEMENT = ("cloc", "pos20", "pos60")
DAY_SWAP = (("upw", "dnw"), ("hi_hr", "lo_hr"), ("zBUY_size", "zSELL_size"), ("zBUY_near", "zSELL_near"))      # zBUY_*/zSELL_* are already expressed in each zone's own direction, so they swap without a sign change
DAY_SWAPNEG = (("d_prev_hi", "d_prev_lo"),)                                                                       # (c - prev_hi)/a and (c - prev_lo)/a: under price -> -price they become -(c - prev_lo)/a and -(c - prev_hi)/a

def mirror_day(X, cols, side):
    """X (n, len(cols)) day-factor rows -> long-perspective rows (copy); side +1 buy, -1 sell."""
    X = X.copy(); ix = {c: i for i, c in enumerate(cols)}; sell = np.asarray(side) < 0
    for c in DAY_FLIP: X[sell, ix[c]] *= -1
    for c in DAY_COMPLEMENT: X[sell, ix[c]] = 1.0 - X[sell, ix[c]]
    for a, b in DAY_SWAP: t = X[sell, ix[a]].copy(); X[sell, ix[a]] = X[sell, ix[b]]; X[sell, ix[b]] = t
    for a, b in DAY_SWAPNEG: t = X[sell, ix[a]].copy(); X[sell, ix[a]] = -X[sell, ix[b]]; X[sell, ix[b]] = -t
    return X

def day_features(zt, pair, d, cols):
    """last COMPLETED day's 46-factor vector for each event (strictly earlier FX day label)."""
    _, fx = DD.ny_parts(zt.t_ev.values); ev = pd.DataFrame({"date": fx, "k": np.arange(len(zt))}).sort_values("date"); dd = d[d.pair == pair][["date"] + cols].sort_values("date")
    m = pd.merge_asof(ev, dd, on="date", direction="backward", allow_exact_matches=False).sort_values("k"); return mirror_day(m[cols].values.astype(np.float32), cols, zt.side.values)

def shard(pair, tf, reset, out, params=None, start=None, end=None, fill_pen=None):
    m1 = DD.load_m1(pair, start, end); p = params or Z.ZoneParams(tf=tf, reset=reset); zt = Z.build_zone_table(m1, p); m15 = DD.resample(m1, 15); j15 = np.searchsorted(m15["e"], zt.ev.values, side="right") - 1
    ok = (zt.ev.values >= W.L1 + 5) & (j15 >= W.L2 + 5); zt = zt[ok].reset_index(drop=True); j15 = j15[ok]; rw = RW.build_rewards(m1, zt, DD.PIP[pair], fill_pen_pips=fill_pen); d, cols = days_table(); Xd = day_features(zt, pair, d, cols); X = np.concatenate([Xd, FT.event_features(m1, zt, pair, start, end), (~np.isfinite(Xd).all(1)).astype(np.float32)[:, None]], 1)       # last column: day factors unavailable (events before 2008)
    out = Path(out) / "shards"; out.mkdir(parents=True, exist_ok=True); name = f"{pair}_{p.tf}_{p.reset}"; zt = zt.assign(pair=DD.PAIRS.index(pair), j15=j15, pairname=pair, ztf=p.tf, reset=p.reset)
    np.savez(out / f"{name}.npz", R=rw["R"], F=rw["F"], RISK=rw["RISK"], V=rw["V"], FIL=rw["FIL"], X=X); zt.to_parquet(out / f"{name}.parquet"); json.dump(dict(cols=cols, feat_names=FT.NAMES + ["day_missing"]), open(out / "daycols.json", "w")); print(name, "zones", len(zt), flush=True)

def merge(out):
    out = Path(out); fs = sorted((out / "shards").glob("*.parquet")); parts = [pd.read_parquet(f) for f in fs]; zt = pd.concat(parts, ignore_index=True)
    order = np.argsort(zt.t_ev.values, kind="stable"); tdir = out / "tableA"; tdir.mkdir(exist_ok=True); n = len(zt)
    for key in ("R", "F", "RISK", "V", "FIL", "X"):
        arrs = [np.load(f.with_suffix(".npz"))[key] for f in fs]; cat = np.concatenate(arrs)[order]; np.save(tdir / f"{key}.npy", cat); print(key, cat.shape, cat.dtype, flush=True)
    zt = zt.iloc[order].reset_index(drop=True); zt.to_parquet(tdir / "zones.parquet"); print("merged", n)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--pair"); ap.add_argument("--tf", type=int, default=1); ap.add_argument("--reset", default="ny"); ap.add_argument("--out", required=True); ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--fill_pen", type=float, default=None, help="pips a limit level must be traded through to fill (0 = touch; 0.3 is stricter; default = env BEID_FILL_PEN or 0)"); a = ap.parse_args()
    if a.cmd == "shard": shard(a.pair, a.tf, a.reset, a.out, start=a.start, end=a.end, fill_pen=a.fill_pen)
    elif a.cmd == "all":
        for pr in DD.PAIRS:
            for tf in (1, 5, 15, 60):
                for rs in ("ny", "utc"): shard(pr, tf, rs, a.out, fill_pen=a.fill_pen)
        merge(a.out)
    elif a.cmd == "merge": merge(a.out)
