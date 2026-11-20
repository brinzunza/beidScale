"""Controls at scale. plan: print the torchrun commands for circular-shift nulls and the planted-edge run. collect: evaluate them and compare with the real run.
  python controls.py plan --table T --runs R --years 2013-2022 --size small --shifts 8 --gpus 6
  python controls.py collect --table T --runs R --years 2013-2022 --shifts 8"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
import evaluate as EV, rewardview as RV, actions as A

def years_of(s): a, b = s.split("-"); return list(range(int(a), int(b) + 1))
SHIFTS = lambda k: [int(x) for x in np.random.default_rng(2026).integers(5000, 60000, k)]

def plan(a):
    sh = SHIFTS(a.shifts); lines = []
    for y in years_of(a.years):
        lines.append(f"torchrun --nproc_per_node={a.gpus} src/train.py --table {a.table} --out {a.runs}/real --size {a.size} --test_year {y} --epochs {a.epochs}")
    for k in sh:
        for y in years_of(a.years): lines.append(f"torchrun --nproc_per_node={a.gpus} src/train.py --table {a.table} --out {a.runs}/null_{k} --size {a.size} --test_year {y} --epochs {a.epochs} --null_shift {k}")
    for y in years_of(a.years): lines.append(f"torchrun --nproc_per_node={a.gpus} src/train.py --table {a.table} --out {a.runs}/plant --size {a.size} --test_year {y} --epochs {a.epochs} --plant {a.plant}")
    print("\n".join(lines))

def collect(a):
    yrs = years_of(a.years); T = Path(a.table); rows = []; files = lambda d: [str(Path(a.runs) / d / f"pred_{y}.npz") for y in yrs if (Path(a.runs) / d / f"pred_{y}.npz").exists()]
    real = EV.pooled(T, files("real")); real["run"] = "real"; rows.append(real); null = []
    for k in SHIFTS(a.shifts):
        f = files(f"null_{k}")
        if f: r = EV.pooled(T, f, null_shift=k); r["run"] = f"null_{k}"; null.append(r)
    nl = pd.concat(null) if null else pd.DataFrame(); out = pd.concat(rows + ([nl] if len(nl) else []), ignore_index=True)
    if len(nl):
        summ = []
        for rule in real.rule.unique():
            for sp in (0.0, 1.3):
                x = nl[(nl.rule == rule) & (nl.spread == sp)].mean_net_R.dropna(); r0 = real[(real.rule == rule) & (real.spread == sp)].mean_net_R
                if len(x) and len(r0): summ.append(dict(rule=rule, spread=sp, real=float(r0.iloc[0]), null_mean=float(x.mean()), null_sd=float(x.std()), null_max=float(x.max()), real_above_null_max=bool(r0.iloc[0] > x.max()), n_null=len(x)))
        pd.DataFrame(summ).to_csv(Path(a.runs) / "null_summary.csv", index=False); print(pd.DataFrame(summ).round(4).to_string(index=False))
    pf = files("plant")
    if pf:
        share = []
        for f in pf:
            P = dict(np.load(f)); z = pd.read_parquet(T / "zones.parquet").iloc[P["rows"]]; reg = RV.region_mask(z); am = P["pol"].astype(np.float32).argmax(1); share.append((am[reg] == RV.PLANT_ACTION).mean() if reg.any() else np.nan)
        print("planted-edge recovery: share of region events where the policy picks the planted action:", np.round(share, 3), "mean", float(np.nanmean(share)))
        pd.DataFrame(dict(year=[int(Path(f).stem.split("_")[1]) for f in pf], recovery_share=share)).to_csv(Path(a.runs) / "plant_recovery.csv", index=False)
    out.to_csv(Path(a.runs) / "controls_all.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--table", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--years", default="2013-2022"); ap.add_argument("--size", default="small"); ap.add_argument("--shifts", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=6); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--plant", type=float, default=0.10); a = ap.parse_args(); {"plan": plan, "collect": collect}[a.cmd](a)
