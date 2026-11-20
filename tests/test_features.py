"""Event features: shapes/names, causality (truncating the bars after the event leaves the features unchanged), RSI vs a plain loop, side mirroring."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import data as DD, zones as Z, features as FT

def setup(end="2012-05-01"):
    m = DD.load_m1("EURUSD", start="2012-01-01", end=end); zt = Z.build_zone_table(m, Z.ZoneParams(tf=15)); zt = zt[zt.t_ev >= np.datetime64("2012-03-01", "ns").astype(np.int64)].reset_index(drop=True); return m, zt

def test_shape_names_and_coverage():
    m, zt = setup(); X = FT.event_features(m, zt, "EURUSD", "2012-01-01", "2012-05-01")
    assert X.shape == (len(zt), FT.N_FEATURES) and len(set(FT.NAMES)) == FT.N_FEATURES and X.dtype == np.float32 and FT.N_FEATURES > 150
    fin = np.isfinite(X).mean(0); bad = [n for n, f in zip(FT.NAMES, fin) if f < (0.90 if n.startswith('asia_') else 0.97)]   # asia range is legitimately undefined for Sunday-evening events; assert not bad, bad

def test_causality_truncation():
    m, zt = setup(); X = FT.event_features(m, zt, "EURUSD", "2012-01-01", "2012-05-01"); k = np.flatnonzero(zt.ev.values < len(m["t"]) - 5000)[-300:]; cut = int(zt.ev.values[k].max()) + 3
    mt = {q: v[:cut] for q, v in m.items()}; zk = zt.iloc[k].reset_index(drop=True); a = FT.event_features(mt, zk, "EURUSD", "2012-01-01", "2012-05-01"); b = X[k]
    zdrop = [FT.NAMES.index(n) for n in ("zones_today", "same_side_today")]; keep = [q for q in range(FT.N_FEATURES) if q not in zdrop]   # per-day counters depend on which events are in the table, not on bars
    assert np.allclose(a[:, keep], b[:, keep], equal_nan=True, atol=1e-4), [FT.NAMES[q] for q in keep if not np.allclose(a[:, q], b[:, q], equal_nan=True, atol=1e-4)]

def test_rsi_matches_loop():
    rng = np.random.default_rng(0); c = 1 + np.cumsum(rng.normal(0, 1e-3, 500)); n = 14; d = np.diff(c); u, dn = np.maximum(d, 0), np.maximum(-d, 0); au = ad = None; out = []
    au, ad = u[0], dn[0]
    for q in range(1, len(d)): au = (1 - 1 / n) * au + u[q] / n; ad = (1 - 1 / n) * ad + dn[q] / n
    ref = 100 - 100 / (1 + au / ad); assert abs(FT.rsi(c, n)[-1] - ref) < 1e-6

def test_mirroring_flips_signed_and_keeps_unsigned():
    m, zt = setup("2012-04-10"); X = FT.event_features(m, zt, "EURUSD", "2012-01-01", "2012-04-10"); z2 = zt.copy(); z2["side"] = -z2["side"]; z2["top"], z2["bot"] = z2["bot"], z2["top"]
    # flipping every event's side (and swapping top/bot so near/far edges stay put) flips signed features and swaps hi/lo distances
    X2 = FT.event_features(m, z2, "EURUSD", "2012-01-01", "2012-04-10"); ix = {n: i for i, n in enumerate(FT.NAMES)}
    for n in ("h1_rsi14", "m15_ret4", "day_open_dist", "m1_r15"): assert np.allclose(X[:, ix[n]], -X2[:, ix[n]], equal_nan=True, atol=1e-5), n
    for n in ("h1_adx", "m15_boll_bw", "asia_range"): assert np.allclose(X[:, ix[n]], X2[:, ix[n]], equal_nan=True, atol=1e-5), n
    assert np.allclose(X[:, ix["prev_hi_dist"]], X2[:, ix["prev_lo_dist"]], equal_nan=True, atol=1e-5)
