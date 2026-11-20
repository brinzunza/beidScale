"""Scaling study: data fraction x model size. plan prints the commands; collect builds the scaling table (validation loss / expected net, top-q realized net R vs parameters and data).
  python scaling.py plan --table T --runs R --years 2016,2019,2022 --gpus 6 ; python scaling.py collect --table T --runs R --years 2016,2019,2022"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
import evaluate as EV, model as M
SIZES = ("tiny", "small", "base", "large"); FRACS = (0.1, 0.3, 1.0)
def plan(a):
    for s in SIZES:
        for f in FRACS:
            for y in a.years.split(","): print(f"torchrun --nproc_per_node={a.gpus} src/train.py --table {a.table} --out {a.runs}/scale_{s}_{f} --size {s} --frac {f} --test_year {y} --epochs {a.epochs} {'--grad_ckpt' if s in ('base', 'large') else ''}")
def collect(a):
    rows = []; ys = [int(y) for y in a.years.split(",")]
    for s in SIZES:
        npar = M.BEIDNet(s, n_ctx=4 + 46 + len(__import__("features").NAMES)).n_params()
        for f in FRACS:
            d = Path(a.runs) / f"scale_{s}_{f}"; logs = [pd.read_csv(d / f"log_{y}.csv") for y in ys if (d / f"log_{y}.csv").exists()]; files = [str(d / f"pred_{y}.npz") for y in ys if (d / f"pred_{y}.npz").exists()]
            if not logs: continue
            best = [l.sort_values("ev_val", ascending=False).iloc[0] for l in logs]; r = dict(size=s, params_M=npar / 1e6, frac=f, q_val=float(np.mean([b.q_val for b in best])), ev_val=float(np.mean([b.ev_val for b in best])), years=len(files))
            if len(files) >= 2:
                p = EV.pooled(a.table, files)
                for rule in ("greedy", "top1pct", "top3pct"):
                    for sp in (0.0, 1.3):
                        x = p[(p.rule == rule) & (p.spread == sp)]
                        if len(x) and "mean_net_R" in x: r[f"{rule}_net@{sp}"] = float(x.mean_net_R.iloc[0]); r[f"{rule}_n@{sp}"] = int(x.n_trades.iloc[0])
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(Path(a.runs) / "scaling.csv", index=False); print(df.round(4).to_string(index=False))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--table", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--years", default="2016,2019,2022"); ap.add_argument("--gpus", type=int, default=6); ap.add_argument("--epochs", type=int, default=6); a = ap.parse_args(); {"plan": plan, "collect": collect}[a.cmd](a)
