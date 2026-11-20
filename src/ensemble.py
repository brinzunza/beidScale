"""Average the predictions of several runs (different seeds / settings) on the same test rows into one pred_<year>.npz, to cut the noise of the arg-max over 420 actions.
  python ensemble.py --runs RUNS/s0 RUNS/s1 RUNS/s2 --year 2020 --out RUNS/ens"""
import argparse, numpy as np
from pathlib import Path

def average(files):
    P = [np.load(f) for f in files]; rows = P[0]["rows"]
    for p in P[1:]: assert np.array_equal(p["rows"], rows), "runs must predict the same rows"
    return dict(rows=rows, q=np.mean([p["q"].astype(np.float32) for p in P], 0).astype(np.float16), pol=np.mean([p["pol"].astype(np.float32) for p in P], 0).astype(np.float16))    # masked logits are equal in all runs, so the mean keeps them masked

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--year", type=int, required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); np.savez(out / f"pred_{a.year}.npz", **average([Path(r) / f"pred_{a.year}.npz" for r in a.runs])); print("saved", out / f"pred_{a.year}.npz")
