"""Refit with the validation year included: for every finished run, print the command that retrains on all years < test year for the epoch count that won on the validation year (use the same flags otherwise).
  python refit.py --runs RUNS/a1 --years 2013-2022 --table T --out RUNS/a1_refit --size small --gpus 6 [--extra "--min_year 2010 --ema 0.999"]"""
import argparse, pandas as pd
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--years", default="2013-2022"); ap.add_argument("--table", required=True); ap.add_argument("--out", required=True); ap.add_argument("--size", default="small"); ap.add_argument("--gpus", type=int, default=6); ap.add_argument("--extra", default=""); ap.add_argument("--select", default="ev")
a = ap.parse_args(); y0, y1 = [int(x) for x in a.years.split("-")]
for y in range(y0, y1 + 1):
    f = Path(a.runs) / f"log_{y}.csv"
    if not f.exists(): continue
    l = pd.read_csv(f); best = l.sort_values("ev_val", ascending=False).iloc[0] if a.select == "ev" else l.sort_values("q_val").iloc[0]
    print(f"torchrun --nproc_per_node={a.gpus} src/train.py --table {a.table} --out {a.out} --size {a.size} --test_year {y} --epochs {int(best.epoch) + 1} --include_val {a.extra}")
