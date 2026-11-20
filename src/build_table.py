"""Build Table A (event-action table) shards on the VM: zones for (pair, tf, reset) -> state metadata + reward tensors. Merge shards into one memory-mappable table.
Usage: python build_table.py shard --pair EURUSD --tf 5 --reset ny --out DIR | python build_table.py merge --out DIR"""
import sys, os, json, argparse, numpy as np, pandas as pd
from pathlib import Path
import data as DD, zones as Z, rewards as RW, actions as A, windows as W
DAY_EXCL = ("pair", "date", "atr", "c", "o", "h", "l", "year", "next_date")

def days_table():
    f = DD.data_dir() / "days.parquet"
    d = pd.read_parquet(f); cols = [c for c in d.columns if c not in DAY_EXCL and not c.startswith(("y_", "mat_"))]; return d, cols

def day_features(zt, pair, d, cols):
    """last COMPLETED day's 46-factor vector for each event (strictly earlier FX day label)."""
    _, fx = DD.ny_parts(zt.t_ev.values); ev = pd.DataFrame({"date": fx, "k": np.arange(len(zt))}).sort_values("date"); dd = d[d.pair == pair][["date"] + cols].sort_values("date")
    m = pd.merge_asof(ev, dd, on="date", direction="backward", allow_exact_matches=False).sort_values("k"); return m[cols].values.astype(np.float32)

def shard(pair, tf, reset, out, params=None, start=None, end=None):
    m1 = DD.load_m1(pair, start, end); p = params or Z.ZoneParams(tf=tf, reset=reset); zt = Z.build_zone_table(m1, p); m15 = DD.resample(m1, 15); j15 = np.searchsorted(m15["e"], zt.ev.values, side="right") - 1
    ok = (zt.ev.values >= W.L1 + 5) & (j15 >= W.L2 + 5); zt = zt[ok].reset_index(drop=True); j15 = j15[ok]; rw = RW.build_rewards(m1, zt, DD.PIP[pair]); d, cols = days_table(); X = day_features(zt, pair, d, cols)
    out = Path(out) / "shards"; out.mkdir(parents=True, exist_ok=True); name = f"{pair}_{p.tf}_{p.reset}"; zt = zt.assign(pair=DD.PAIRS.index(pair), j15=j15, pairname=pair, ztf=p.tf, reset=p.reset)
    np.savez(out / f"{name}.npz", R=rw["R"], F=rw["F"], RISK=rw["RISK"], V=rw["V"], FIL=rw["FIL"], X=X); zt.to_parquet(out / f"{name}.parquet"); json.dump(dict(cols=cols), open(out / "daycols.json", "w")); print(name, "zones", len(zt), flush=True)

def merge(out):
    out = Path(out); fs = sorted((out / "shards").glob("*.parquet")); parts = [pd.read_parquet(f) for f in fs]; zt = pd.concat(parts, ignore_index=True)
    order = np.argsort(zt.t_ev.values, kind="stable"); tdir = out / "tableA"; tdir.mkdir(exist_ok=True); n = len(zt)
    for key in ("R", "F", "RISK", "V", "FIL", "X"):
        arrs = [np.load(f.with_suffix(".npz"))[key] for f in fs]; cat = np.concatenate(arrs)[order]; np.save(tdir / f"{key}.npy", cat); print(key, cat.shape, cat.dtype, flush=True)
    zt = zt.iloc[order].reset_index(drop=True); zt.to_parquet(tdir / "zones.parquet"); print("merged", n)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--pair"); ap.add_argument("--tf", type=int, default=1); ap.add_argument("--reset", default="ny"); ap.add_argument("--out", required=True); ap.add_argument("--start"); ap.add_argument("--end"); a = ap.parse_args()
    if a.cmd == "shard": shard(a.pair, a.tf, a.reset, a.out, start=a.start, end=a.end)
    elif a.cmd == "all":
        for pr in DD.PAIRS:
            for tf in (1, 5, 15, 60):
                for rs in ("ny", "utc"): shard(pr, tf, rs, a.out)
        merge(a.out)
    elif a.cmd == "merge": merge(a.out)
