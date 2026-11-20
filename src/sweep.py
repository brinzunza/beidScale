"""Hyperparameter search on DEVELOPMENT years only (default test years 2016 and 2019; validation = the year before). plan prints torchrun commands for N random configurations; collect ranks them by the best-epoch
policy expected net R on the validation year (ev_val), then Q loss. Never touches 2023+.
  python sweep.py plan --table T --runs R --n 12 --gpus 6 > sweep_cmds.sh ; bash sweep_cmds.sh ; python sweep.py collect --runs R"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
GRID = dict(lr=[1e-4, 3e-4, 1e-3], wd=[0.01, 0.05, 0.2], epochs=[3, 6, 12], ent=[0.0, 0.01, 0.05], train_spread=[0.5, 1.0, 1.3], ema=[0.0, 0.999], tf_pow=[0.0, 0.5, 1.0], objective=["net", "logg"])

def configs(n, seed=11):
    r = np.random.default_rng(seed); return [{k: v[int(r.integers(len(v)))] for k, v in GRID.items()} for _ in range(n)]

def plan(a):
    for i, c in enumerate(configs(a.n)):
        for y in a.years.split(","):
            flags = " ".join(f"--{k} {v}" for k, v in c.items()); print(f"torchrun --nproc_per_node={a.gpus} src/train.py --table {a.table} --out {a.runs}/sweep_{i} --size {a.size} --test_year {y} --batch 512 {flags} --min_year {a.min_year}")
    json.dump(configs(a.n), open(Path(a.runs) / "sweep_configs.json", "w"), indent=1) if Path(a.runs).exists() else None

def collect(a):
    cf = configs(a.n); rows = []
    for i, c in enumerate(cf):
        logs = [pd.read_csv(f) for f in sorted((Path(a.runs) / f"sweep_{i}").glob("log_*.csv"))]
        if not logs: continue
        best = [l.sort_values("ev_val", ascending=False).iloc[0] for l in logs]; rows.append(dict(config=i, years=len(logs), ev_val=float(np.mean([b.ev_val for b in best])), q_val=float(np.mean([b.q_val for b in best])), **c))
    df = pd.DataFrame(rows).sort_values(["ev_val", "q_val"], ascending=[False, True]); df.to_csv(Path(a.runs) / "sweep_results.csv", index=False); print(df.round(4).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--table"); ap.add_argument("--runs", required=True); ap.add_argument("--n", type=int, default=12); ap.add_argument("--years", default="2016,2019"); ap.add_argument("--size", default="small"); ap.add_argument("--gpus", type=int, default=6); ap.add_argument("--min_year", type=int, default=0)
    a = ap.parse_args(); {"plan": plan, "collect": collect}[a.cmd](a)
