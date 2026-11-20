"""Parameterized BEID zone engine (generalizes goal335 raw_beid.zones_kernel; defaults reproduce it exactly) + zone-event table builder."""
import numpy as np, pandas as pd
from dataclasses import dataclass, asdict, field
from numba import njit
import data as DD

@dataclass
class ZoneParams:
    tf: int = 1                 # timeframe minutes: 1, 5, 15, 60
    reset: str = "ny"           # 'ny' | 'utc' | 'none'
    pb_mode: int = 0            # pullback start: 0 red close < prev green open (original), 1 any red candle, 2 close < previous low
    confirm_close: int = 1      # zone confirmed by a CLOSE beyond the old high (1) or by a WICK (0)
    flip_close: int = 1         # trend flip by a close (1) or a wick (0) through the floor/ceiling
    pb_min: int = 1             # minimum number of pullback candles before confirmation
    min_size_atr: float = 0.0   # zone height / ATR15 bounds
    max_size_atr: float = 1e9
    min_gap_bars: int = 0       # minimum TF bars since the previous zone
    h0: int = 0                 # allowed NY hour of the event [h0, h1)
    h1: int = 24
    def key(self): return "|".join(f"{k}={v}" for k, v in asdict(self).items())

@njit(cache=True)
def zones_kernel(o, h, l, c, day, daily, pb_mode, confirm_close, flip_close, pb_min):
    n = len(o); cap = n // 2 + 16
    zc = np.empty(cap, np.int64); zs = np.empty(cap, np.int8); ztop = np.empty(cap); zbot = np.empty(cap); nz = 0; nan = np.nan
    up = True; pHH = nan; cHH = nan; pHL = nan; lastcHL = nan; pLL = nan; cLL = nan; pLH = nan; lastcLH = nan; inpb = False; pg = nan; pr = nan; pb_cnt = 0
    for i in range(n):
        if daily and i > 0 and day[i] != day[i - 1]:
            up = True; pHH = nan; cHH = nan; pHL = nan; lastcHL = nan; pLL = nan; cLL = nan; pLH = nan; lastcLH = nan; inpb = False; pg = nan; pr = nan; pb_cnt = 0
        oo = o[i]; hh = h[i]; ll = l[i]; cl = c[i]; green = cl >= oo; red = cl < oo
        if up:
            if np.isnan(pHH) or hh > pHH: pHH = hh
            flip = (cl < lastcHL) if flip_close == 1 else (ll < lastcHL)
            if (not np.isnan(lastcHL)) and flip:
                zc[nz] = i; zs[nz] = 3; ztop[nz] = pHH; zbot[nz] = lastcHL; nz += 1
                up = False; lastcLH = nan; pLL = ll; cLL = nan; inpb = False; pb_cnt = 0
            else:
                if pb_mode == 0: start = red and (not np.isnan(pg)) and cl < pg
                elif pb_mode == 1: start = red
                else: start = i > 0 and cl < l[i - 1]
                if (not inpb) and start:
                    inpb = True; cHH = pHH; pHL = ll; pb_cnt = 0
                elif inpb:
                    pb_cnt += 1
                    if ll < pHL: pHL = ll
                    conf = (cl > cHH) if confirm_close == 1 else (hh > cHH)
                    if conf and pb_cnt >= pb_min:
                        inpb = False; lastcHL = pHL; zc[nz] = i; zs[nz] = 0; ztop[nz] = cHH; zbot[nz] = pHL; nz += 1; pHH = hh
        else:
            if np.isnan(pLL) or ll < pLL: pLL = ll
            flip = (cl > lastcLH) if flip_close == 1 else (hh > lastcLH)
            if (not np.isnan(lastcLH)) and flip:
                zc[nz] = i; zs[nz] = 2; ztop[nz] = lastcLH; zbot[nz] = pLL; nz += 1
                up = True; lastcHL = nan; pHH = hh; cHH = nan; inpb = False; pb_cnt = 0
            else:
                if pb_mode == 0: start = green and (not np.isnan(pr)) and cl > pr
                elif pb_mode == 1: start = green
                else: start = i > 0 and cl > h[i - 1]
                if (not inpb) and start:
                    inpb = True; cLL = pLL; pLH = hh; pb_cnt = 0
                elif inpb:
                    pb_cnt += 1
                    if hh > pLH: pLH = hh
                    conf = (cl < cLL) if confirm_close == 1 else (ll < cLL)
                    if conf and pb_cnt >= pb_min:
                        inpb = False; lastcLH = pLH; zc[nz] = i; zs[nz] = 1; ztop[nz] = pLH; zbot[nz] = cLL; nz += 1; pLL = ll
        if green: pg = oo
        elif red: pr = oo
    return zc[:nz], zs[:nz], ztop[:nz], zbot[:nz]

def build_zone_table(m1, p: ZoneParams, atr15=None):
    """zone events for one pair. Event = close of the TF bar that confirms the zone; simulation starts at M1 index `ev` (first M1 bar after the TF bar)."""
    tf = DD.resample(m1, p.tf); N = len(m1["t"]); daily = p.reset != "none"; day = DD.day_ids(tf["t"], "utc" if p.reset == "utc" else "ny") if daily else np.zeros(len(tf["t"]), np.int64)
    zc, zs, top, bot = zones_kernel(tf["o"], tf["h"], tf["l"], tf["c"], day, daily, p.pb_mode, p.confirm_close, p.flip_close, p.pb_min)
    keep = (top > bot) & (tf["e"][zc] < N); zc, zs, top, bot = zc[keep], zs[keep], top[keep], bot[keep]; ev = tf["e"][zc]
    m15 = DD.resample(m1, 15); a15 = DD.wilder_atr(m15["h"], m15["l"], m15["c"]) if atr15 is None else atr15
    j = np.searchsorted(m15["e"], ev, side="right") - 1; ok = j >= 20; zc, zs, top, bot, ev, j = zc[ok], zs[ok], top[ok], bot[ok], ev[ok], j[ok]; atr = a15[j]
    size = (top - bot) / atr; ok = (size >= p.min_size_atr) & (size <= p.max_size_atr) & (atr > 0)
    t_ev = m1["t"][ev]; hr, _ = DD.ny_parts(t_ev); ok &= (hr >= p.h0) & (hr < p.h1)
    if p.min_gap_bars > 0:
        last = -10 ** 9; k2 = np.zeros(len(zc), bool)
        for q in range(len(zc)):
            if ok[q] and zc[q] - last >= p.min_gap_bars: k2[q] = True; last = zc[q]
        ok &= k2
    zc, zs, top, bot, ev, atr, t_ev, hr = zc[ok], zs[ok], top[ok], bot[ok], ev[ok], atr[ok], t_ev[ok], hr[ok]
    d1 = DD.day_ids(m1["t"], "utc" if p.reset == "utc" else "ny") if daily else None
    if daily:
        chg = np.flatnonzero(d1[1:] != d1[:-1]) + 1; k = np.searchsorted(chg, ev, side="right"); ord_end = np.where(k < len(chg), chg[np.minimum(k, len(chg) - 1)], N)
    else: ord_end = np.minimum(ev + 2880, N)
    ts = pd.to_datetime(t_ev)
    return pd.DataFrame(dict(tf=p.tf, side=np.where((zs == 0) | (zs == 2), 1, -1).astype(np.int8), ztype=(zs >= 2).astype(np.int8), tfbar=zc, ev=ev, ord_end=ord_end, top=top, bot=bot, atr15=atr, t_ev=t_ev, year=ts.year.values, ny_hour=hr, dow=ts.dayofweek.values))
