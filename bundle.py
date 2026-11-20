"""Pack the data needed on the VM (about 0.6 GB): M1 bars for the 4 pairs (m1_<PAIR>.parquet: time/open/high/low/close) and the 46-factor day table (days.parquet).
Mac-side helper (needs the original research folders; not used on the VM). Reads M1 from BEID_DATA (the goal311 cache layout) and the day table from DAYS_PARQUET.
Usage: BEID_DATA=<goal311 cache> python bundle.py [OUT_DIR]   (default ~/Documents/beidScale_bundle). Upload the folder to the VM and set BEID_DATA=<folder>."""
import sys, os, shutil, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src")); import data as DD
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Documents" / "beidScale_bundle"; out.mkdir(parents=True, exist_ok=True)
for p in DD.PAIRS:
    m = DD.load_m1(p); df = pd.DataFrame({"time": pd.to_datetime(m["t"]), "open": m["o"], "high": m["h"], "low": m["l"], "close": m["c"]}); df.to_parquet(out / f"m1_{p}.parquet", compression="zstd", index=False); print(p, len(df), df.time.iloc[0], df.time.iloc[-1], flush=True)
days = Path(os.environ.get("DAYS_PARQUET", Path(__file__).resolve().parents[1] / "similaritySearch" / "data" / "days.parquet")); shutil.copy(days, out / "days.parquet")
tot = sum(f.stat().st_size for f in out.iterdir()) / 1e6; print(f"bundle at {out}: {tot:.0f} MB")
