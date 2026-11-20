"""beidScale data layer: M1 bars, FX-day ids, resampling to M5/M15/H1 with M1 index mapping, ATR. Everything keyed to a bundle directory (BEID_DATA env var)."""
import os, numpy as np, pandas as pd
from pathlib import Path
PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "EURGBP"); PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}
def data_dir():
    d = os.environ.get("BEID_DATA")
    if d: return Path(d)
    local = Path(__file__).resolve().parents[1] / "data"      # <beidScale>/data also works without the env var
    if local.exists(): return local
    raise FileNotFoundError("set BEID_DATA to the bundle directory (m1_<PAIR>.parquet x4 + days.parquet) or put it in beidScale/data/")

def load_m1(pair, start=None, end=None):
    """-> dict(t int64 ns UTC-naive, o,h,l,c float64). Bundle layout: m1_<PAIR>.parquet with columns time/open/high/low/close; also accepts the goal311 cache layout <PAIR>_m1v.parquet (index=time)."""
    d = data_dir(); f = d / f"m1_{pair}.parquet"
    if f.exists(): x = pd.read_parquet(f); t = pd.DatetimeIndex(x["time"]) if "time" in x else x.index
    else: x = pd.read_parquet(d / f"{pair}_m1v.parquet"); t = x.index
    t = pd.DatetimeIndex(t)
    if start is not None or end is not None:
        m = np.ones(len(t), bool)
        if start is not None: m &= t >= pd.Timestamp(start)
        if end is not None: m &= t < pd.Timestamp(end)
        x = x[m]; t = t[m]
    return dict(t=t.values.astype("datetime64[ns]").astype(np.int64), o=x["open"].values.astype(np.float64), h=x["high"].values.astype(np.float64), l=x["low"].values.astype(np.float64), c=x["close"].values.astype(np.float64))

def ny_parts(t_ns):
    """returns (ny_hour, fx_day_id[int days]) for UTC ns timestamps; FX day = 17:00-17:00 America/New_York ('ny'); also utc day for 'utc'."""
    ts = pd.DatetimeIndex(t_ns.astype("datetime64[ns]")).tz_localize("UTC").tz_convert("America/New_York")
    return ts.hour.values, (ts + pd.Timedelta(hours=7)).normalize().tz_localize(None).values.astype("datetime64[D]").astype(np.int64)

def day_ids(t_ns, mode="ny"):
    if mode == "utc": return pd.DatetimeIndex(t_ns.astype("datetime64[ns]")).normalize().values.astype("datetime64[D]").astype(np.int64)
    return ny_parts(t_ns)[1]

def resample(m1, minutes):
    """aggregate M1 -> TF bars aligned to epoch multiples. Returns dict(t,o,h,l,c, s (first M1 index), e (one past last M1 index))."""
    if minutes == 1:
        n = len(m1["t"]); return dict(t=m1["t"], o=m1["o"], h=m1["h"], l=m1["l"], c=m1["c"], s=np.arange(n), e=np.arange(1, n + 1))
    step = minutes * 60 * 10 ** 9; key = m1["t"] // step; cut = np.flatnonzero(np.diff(key) != 0) + 1; s = np.r_[0, cut]; e = np.r_[cut, len(key)]
    o = m1["o"][s]; c = m1["c"][e - 1]; h = np.maximum.reduceat(m1["h"], s); l = np.minimum.reduceat(m1["l"], s)
    return dict(t=key[s] * step, o=o, h=h, l=l, c=c, s=s, e=e)

def wilder_atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]; tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)]); return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().values
