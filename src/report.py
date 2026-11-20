"""Approach 1 report: pooled evaluation of a run directory (pred_<year>.npz files) + baselines, written to CSV and printed.
  python report.py --table T --runs R/a1 --years 2013-2022 --out R/a1_report.csv"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
import evaluate as EV, baselines as BL

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--table", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--years", default="2013-2022"); ap.add_argument("--out", required=True); ap.add_argument("--baselines", action="store_true"); a = ap.parse_args()
    y0, y1 = [int(x) for x in a.years.split("-")]; files = [str(Path(a.runs) / f"pred_{y}.npz") for y in range(y0, y1 + 1) if (Path(a.runs) / f"pred_{y}.npz").exists()]; p = EV.pooled(a.table, files); p["source"] = "model"; parts = [p]
    if a.baselines:
        z = pd.read_parquet(Path(a.table) / "zones.parquet")
        for y in range(y0, y1 + 1):
            tr, te = BL.split_rows(z, y); b = EV.baselines(a.table, tr[:: max(1, len(tr) // 200000)], te); b["source"] = f"baseline_{y}"; b["year"] = y; parts.append(b)
    out = pd.concat(parts, ignore_index=True); out.to_csv(a.out, index=False); cols = ["rule", "spread", "n_trades", "fill_rate", "mean_gross_R", "mean_net_R", "t", "win", "trades_per_year", "pos_years", "K", "median_risk_pips", "share_wide_stop", "share_fade"]
    print(p[p.spread.isin([0.0, 1.0, 1.3])][[c for c in cols if c in p]].round(4).to_string(index=False))
if __name__ == "__main__":
    main()
