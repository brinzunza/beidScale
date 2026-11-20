"""Causal event features added to every zone event (technical indicators, mean reversion, trend, volatility, session / zone context, cross-pair).
All values use only bars CLOSED before the event (M15/H1/H4 bars: last bar with end <= ev; M1: bars < ev). Directional features are mirrored to the long perspective
(sell zones: signs flipped, high/low distances swapped), like the price windows. Output: float32 matrix (n_events, N_FEATURES) + NAMES."""
import numpy as np, pandas as pd
import data as DD

TFS = ((15, "m15"), (60, "h1"), (240, "h4"))
BAR_SIGNED = ["ret1", "ret4", "ret16", "sma10_dist", "sma20_dist", "sma50_dist", "sma100_dist", "sma200_dist", "ema12_dist", "ema26_dist", "sma20_slope", "sma20_50_spread", "sma20_50_state",
              "rsi14", "rsi2", "macd", "macd_hist", "boll_pctb", "z50", "z100", "stoch_k", "stoch_d", "adx_di", "don20_pos", "don55_pos", "body"]
BAR_UNSIGNED = ["boll_bw", "boll_squeeze", "adx", "don20_hh_dist", "don20_ll_dist", "atr_ratio", "rv_ratio", "bar_range", "upper_wick", "lower_wick", "eff_ratio20", "ac1_50"]
BAR_SWAP = [("don20_hh_dist", "don20_ll_dist"), ("upper_wick", "lower_wick")]
M1_SIGNED = ["m1_r5", "m1_r15", "m1_r60", "m1_rsi14", "m1_z60"]; M1_UNSIGNED = ["m1_ac1_60", "m1_rv_ratio"]
SESS_SIGNED = ["day_open_dist", "day_pos", "prev_close_dist", "week_open_dist"]; SESS_UNSIGNED = ["day_hi_dist", "day_lo_dist", "prev_hi_dist", "prev_lo_dist", "asia_hi_dist", "asia_lo_dist", "asia_range", "day_range", "hrs_since_open"]
SESS_SWAP = [("day_hi_dist", "day_lo_dist"), ("prev_hi_dist", "prev_lo_dist"), ("asia_hi_dist", "asia_lo_dist")]
ZONE_NONE = ["dist_near", "dist_far"]; ZONE_UNSIGNED = ["zone_gap_log", "zones_today", "same_side_today"]
CROSS = ["ret4", "rsi14", "sma50_dist"]
NAMES, KIND = [], {}
for _, tag in TFS:
    for n in BAR_SIGNED: NAMES.append(f"{tag}_{n}"); KIND[NAMES[-1]] = "s"
    for n in BAR_UNSIGNED: NAMES.append(f"{tag}_{n}"); KIND[NAMES[-1]] = "u"
for n in M1_SIGNED: NAMES.append(n); KIND[n] = "s"
for n in M1_UNSIGNED: NAMES.append(n); KIND[n] = "u"
for n in SESS_SIGNED: NAMES.append(n); KIND[n] = "s"
for n in SESS_UNSIGNED: NAMES.append(n); KIND[n] = "u"
for n in ZONE_NONE: NAMES.append(n); KIND[n] = "n"
for n in ZONE_UNSIGNED: NAMES.append(n); KIND[n] = "u"
for p in DD.PAIRS:
    for n in CROSS: NAMES.append(f"x_{p}_{n}"); KIND[NAMES[-1]] = "s"
NAMES.append("side"); KIND["side"] = "n"
SWAPS = [(f"{tag}_{a}", f"{tag}_{b}") for _, tag in TFS for a, b in BAR_SWAP] + SESS_SWAP
N_FEATURES = len(NAMES)

def _s(x): return pd.Series(x)
def _sh(x, k): out = np.empty(len(x)); out[:k] = np.nan; out[k:] = x[:-k]; return out
def _ema(x, n): return _s(x).ewm(span=n, adjust=False).mean().values
def _wilder(x, n): return _s(x).ewm(alpha=1.0 / n, adjust=False).mean().values
def _div(a, b):
    with np.errstate(all="ignore"): return a / np.where(b > 0, b, np.nan)
def rsi(c, n):
    d = c - _sh(c, 1); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0); up[np.isnan(d)] = np.nan; dn[np.isnan(d)] = np.nan
    with np.errstate(all="ignore"): rs = _wilder(up, n) / _wilder(dn, n); return 100.0 - 100.0 / (1.0 + rs)

def bar_features(b):
    """indicator arrays per bar of one timeframe (value at bar i uses bars <= i only)."""
    o, h, l, c = b["o"], b["h"], b["l"], b["c"]; atr = DD.wilder_atr(h, l, c, 14); F = {}
    for k in (1, 4, 16): F[f"ret{k}"] = _div(c - _sh(c, k), atr)
    sma = {n: _s(c).rolling(n).mean().values for n in (10, 20, 50, 100, 200)}
    for n, v in sma.items(): F[f"sma{n}_dist"] = _div(c - v, atr)
    e12, e26 = _ema(c, 12), _ema(c, 26); F["ema12_dist"] = _div(c - e12, atr); F["ema26_dist"] = _div(c - e26, atr)
    F["sma20_slope"] = _div(sma[20] - _sh(sma[20], 5), atr); F["sma20_50_spread"] = _div(sma[20] - sma[50], atr); F["sma20_50_state"] = np.sign(sma[20] - sma[50])
    F["rsi14"] = (rsi(c, 14) - 50) / 50; F["rsi2"] = (rsi(c, 2) - 50) / 50
    macd = e12 - e26; F["macd"] = _div(macd, atr); F["macd_hist"] = _div(macd - _ema(macd, 9), atr)
    s20 = _s(c).rolling(20).std().values; F["boll_pctb"] = _div(c - (sma[20] - 2 * s20), 4 * s20) - 0.5; bw = 4 * s20; F["boll_bw"] = _div(bw, atr); F["boll_squeeze"] = _s(bw).rolling(100).rank(pct=True).values
    for n in (50, 100): F[f"z{n}"] = _div(c - sma[n], _s(c).rolling(n).std().values)
    hh14, ll14 = _s(h).rolling(14).max().values, _s(l).rolling(14).min().values; k = _div(c - ll14, hh14 - ll14) - 0.5; F["stoch_k"] = k; F["stoch_d"] = _s(k).rolling(3).mean().values
    pc = _sh(c, 1); tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)]); uh = h - _sh(h, 1); dl = _sh(l, 1) - l
    pdm = np.where((uh > dl) & (uh > 0), uh, 0.0); mdm = np.where((dl > uh) & (dl > 0), dl, 0.0); aw = _wilder(tr, 14); pdi = 100 * _div(_wilder(pdm, 14), aw); mdi = 100 * _div(_wilder(mdm, 14), aw)
    F["adx"] = _wilder(100 * _div(np.abs(pdi - mdi), pdi + mdi), 14) / 100; F["adx_di"] = _div(pdi - mdi, pdi + mdi)
    for n in (20, 55):
        hh, ll = _s(h).rolling(n).max().values, _s(l).rolling(n).min().values; F[f"don{n}_pos"] = _div(c - ll, hh - ll) - 0.5
        if n == 20: F["don20_hh_dist"] = _div(hh - c, atr); F["don20_ll_dist"] = _div(c - ll, atr)
    d = c - _sh(c, 1); F["atr_ratio"] = _div(_wilder(tr, 5), _wilder(tr, 50)); F["rv_ratio"] = _div(_s(d).rolling(20).std().values, _s(d).rolling(100).std().values); F["bar_range"] = _div(h - l, atr)
    F["body"] = _div(c - o, atr); F["upper_wick"] = _div(h - np.maximum(o, c), atr); F["lower_wick"] = _div(np.minimum(o, c) - l, atr)
    F["eff_ratio20"] = _div(np.abs(c - _sh(c, 20)), _s(np.abs(d)).rolling(20).sum().values); F["ac1_50"] = _s(d).rolling(50).corr(_s(_sh(d, 1))).values
    return F

def _at(a, j):
    out = np.full(len(j), np.nan); ok = j >= 0; out[ok] = a[j[ok]]; return out

_CACHE = {}
def _h1_cross(pair, start, end, m1=None):
    key = (pair, start, end)
    if m1 is None and key in _CACHE: return _CACHE[key]
    m = m1 if m1 is not None else DD.load_m1(pair, start, end); b = DD.resample(m, 60); atr = DD.wilder_atr(b["h"], b["l"], b["c"], 14); c = b["c"]
    F = np.stack([_div(c - _sh(c, 4), atr), (rsi(c, 14) - 50) / 50, _div(c - _s(c).rolling(50).mean().values, atr)], 1); res = (b["t"] + 3600 * 10 ** 9, F)
    if m1 is None: _CACHE[key] = res
    return res

def event_features(m1, zt, pair, start=None, end=None):
    """m1: bars dict of `pair`; zt: zone-event table (columns ev, side, top, bot, atr15, t_ev; sorted by event order within the shard). -> float32 (n, N_FEATURES)."""
    n = len(zt); ev = zt.ev.values.astype(np.int64); atr = zt.atr15.values; side = zt.side.values.astype(np.float64); out = {}; i1 = ev - 1
    for minutes, tag in TFS:                                                                    # --- indicators on closed M15 / H1 / H4 bars
        b = DD.resample(m1, minutes); F = bar_features(b); j = np.searchsorted(b["e"], ev, side="right") - 1
        for name, arr in F.items(): out[f"{tag}_{name}"] = _at(arr, j)
    c = m1["c"]; d = c - _sh(c, 1)                                                                # --- M1 micro structure
    for k in (5, 15, 60): out[f"m1_r{k}"] = _div(_at(c, i1) - _at(c, i1 - k), atr)
    out["m1_rsi14"] = _at((rsi(c, 14) - 50) / 50, i1); sc = _s(c); out["m1_z60"] = _at(_div(c - sc.rolling(60).mean().values, sc.rolling(60).std().values), i1)
    out["m1_ac1_60"] = _at(_s(d).rolling(60).corr(_s(_sh(d, 1))).values, i1); out["m1_rv_ratio"] = _at(_div(_s(d).rolling(15).std().values, _s(d).rolling(240).std().values), i1)
    t, o, h, l = m1["t"], m1["o"], m1["h"], m1["l"]; day = DD.day_ids(t, "ny"); u, st = np.unique(day, return_index=True); en = np.r_[st[1:], len(day)]       # --- session context
    dhi, dlo, dcl, dop = np.maximum.reduceat(h, st), np.minimum.reduceat(l, st), c[en - 1], o[st]; k = np.searchsorted(u, day[i1]); kp = k - 1; okp = kp >= 0; kp = np.maximum(kp, 0)
    chi = _s(h).groupby(day).cummax().values; clo = _s(l).groupby(day).cummin().values; pxe = c[i1]
    out["day_open_dist"] = _div(pxe - dop[k], atr); out["day_hi_dist"] = _div(chi[i1] - pxe, atr); out["day_lo_dist"] = _div(pxe - clo[i1], atr); out["day_pos"] = _div(pxe - clo[i1], chi[i1] - clo[i1]) - 0.5; out["day_range"] = _div(chi[i1] - clo[i1], atr)
    out["prev_hi_dist"] = _div(np.where(okp, dhi[kp] - pxe, np.nan), atr); out["prev_lo_dist"] = _div(np.where(okp, pxe - dlo[kp], np.nan), atr); out["prev_close_dist"] = _div(np.where(okp, pxe - dcl[kp], np.nan), atr)
    wk = (u + 3) // 7; _, wfirst = np.unique(wk, return_index=True); wopen = dop[wfirst[np.searchsorted(np.unique(wk), wk)]]; out["week_open_dist"] = _div(pxe - wopen[k], atr)
    out["hrs_since_open"] = (t[i1] - t[st][k]) / 3.6e12; hrs = (t // (3600 * 10 ** 9)) % 24; ud = t // (86400 * 10 ** 9)
    ah = _s(np.where(hrs < 7, h, -np.inf)).groupby(ud).cummax().values[i1]; al = _s(np.where(hrs < 7, l, np.inf)).groupby(ud).cummin().values[i1]; ok = np.isfinite(ah) & np.isfinite(al)
    out["asia_hi_dist"] = _div(np.where(ok, ah - pxe, np.nan), atr); out["asia_lo_dist"] = _div(np.where(ok, pxe - al, np.nan), atr); out["asia_range"] = _div(np.where(ok, ah - al, np.nan), atr)
    near = np.where(side > 0, zt.top.values, zt.bot.values); far = np.where(side > 0, zt.bot.values, zt.top.values)                                       # --- zone context
    out["dist_near"] = side * (pxe - near) / atr; out["dist_far"] = side * (pxe - far) / atr
    tev = zt.t_ev.values.astype(np.int64); gap = np.r_[np.nan, np.diff(tev) / 6e10]; out["zone_gap_log"] = np.log1p(np.maximum(gap, 0)); zday = pd.Series(day[i1])
    out["zones_today"] = zday.groupby(zday.values).cumcount().values.astype(float); out["same_side_today"] = pd.Series(np.arange(n)).groupby([zday.values, side]).cumcount().values.astype(float)
    for p in DD.PAIRS:                                                                          # --- cross-pair H1 context (fixed slot per pair)
        tc, F = _h1_cross(p, start, end, m1 if p == pair else None); j = np.searchsorted(tc, tev, side="right") - 1; fresh = (j >= 0) & (tev - tc[np.maximum(j, 0)] <= 2 * 3600 * 10 ** 9)
        for q, nm in enumerate(CROSS): out[f"x_{p}_{nm}"] = np.where(fresh, F[np.maximum(j, 0), q], np.nan)
    out["side"] = side
    X = np.stack([np.asarray(out[nm], dtype=np.float64) for nm in NAMES], 1); X[~np.isfinite(X)] = np.nan; sell = side < 0; ix = {nm: q for q, nm in enumerate(NAMES)}
    flip = [ix[nm] for nm in NAMES if KIND[nm] == "s"]; X[np.ix_(sell, flip)] *= -1
    for a, b2 in SWAPS: ia, ib = ix[a], ix[b2]; tmp = X[sell, ia].copy(); X[sell, ia] = X[sell, ib]; X[sell, ib] = tmp
    return X.astype(np.float32)

def groups():
    """feature inventory by family (for reporting)."""
    g = {}
    for nm in NAMES:
        if nm.startswith(("m15_", "h1_", "h4_")): fam = nm.split("_", 1)[0]; key = f"{fam} indicators"
        elif nm.startswith("m1_"): key = "M1 micro"
        elif nm.startswith("x_"): key = "cross-pair H1"
        elif nm in SESS_SIGNED + SESS_UNSIGNED: key = "session / day structure"
        elif nm in ZONE_NONE + ZONE_UNSIGNED + ["side"]: key = "zone context"
        else: key = "other"
        g.setdefault(key, []).append(nm)
    return g
