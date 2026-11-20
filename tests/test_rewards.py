"""Reward kernel vs an independent brute-force per-bar reference, plus invariants."""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import data as DD, zones as Z, actions as A, rewards as RW

def ref_action(m1, row, a, pip, minr_pips=1.5, horizon=2880):
    e, j, k, d = A.decode(a); sg = int(row.side); N = len(m1["t"]); o, h, l, c = m1["o"], m1["h"], m1["l"], m1["c"]
    hi = (lambda i: h[i]) if sg > 0 else (lambda i: -l[i]); lo = (lambda i: l[i]) if sg > 0 else (lambda i: -h[i]); cl = (lambda i: c[i]) if sg > 0 else (lambda i: -c[i]); op = (lambda i: o[i]) if sg > 0 else (lambda i: -o[i])
    tp = row.top if sg > 0 else -row.bot; bt = row.bot if sg > 0 else -row.top; H = tp - bt; az = row.atr15; minr = minr_pips * pip
    ev = int(row.ev)
    if e < 5:
        E = tp - A.DEPTH[e] * H; f = None
        for i in range(ev, int(row.ord_end)):
            if lo(i) <= E: f = i; break
        if f is None: return None, None            # unfilled, validity checked separately
    else: E = op(ev); f = ev
    stops = [bt, bt - 0.25 * H, bt - 0.5 * H, E - az, E - 2 * az]; r = E - stops[j]
    if r < minr: return "invalid", None
    if k < 5: t = A.MULT[k] * r
    elif k == 5: t = tp - E
    else: t = 2 * H
    if k >= 5 and t < minr: return "invalid", None
    hend = min(f + horizon, N - 1)
    if d == 0: stop_lvl, tgt_lvl = E - r, E + t
    else: stop_lvl, tgt_lvl = E + r, E - t
    for i in range(f, hend + 1):
        if d == 0: hit_stop = lo(i) <= stop_lvl; hit_t = (i > f) and hi(i) >= tgt_lvl
        else: hit_stop = hi(i) >= stop_lvl; hit_t = (i > f) and lo(i) <= tgt_lvl
        if e == 5 and i == f:   # market entry: the first bar is a normal bar (fill at its open), target may hit on it too
            hit_t = (hi(i) >= tgt_lvl) if d == 0 else (lo(i) <= tgt_lvl)
        if hit_stop: return -1.0, f
        if hit_t: return t / r, f
    ce = cl(hend); return ((ce - E) / r if d == 0 else (E - ce) / r), f

def make():
    m1 = DD.load_m1("EURUSD", start="2012-03-01", end="2012-04-30"); zt = Z.build_zone_table(m1, Z.ZoneParams(tf=15)); return m1, zt

def test_kernel_matches_reference():
    m1, zt = make(); pip = DD.PIP["EURUSD"]; zs = zt.iloc[:120]; out = RW.build_rewards(m1, zs, pip); rng = np.random.default_rng(0); n_cmp = 0; bad = 0
    for zi in range(len(zs)):
        row = zs.iloc[zi]
        for a in rng.integers(0, A.NACT, 25):
            e = A.decode(a)[0]; res, f = ref_action(m1, row, int(a), pip)
            if res == "invalid": assert out["V"][zi, a] == 0; continue
            if res is None: assert out["F"][zi, e] == 0; continue
            assert out["V"][zi, a] == 1 and out["F"][zi, e] == 1
            n_cmp += 1
            if abs(float(out["R"][zi, a]) - res) > 5e-3 * max(1, abs(res)): bad += 1
    assert n_cmp > 800, n_cmp; assert bad == 0, (bad, n_cmp)

def test_invariants_and_spread_formula():
    m1, zt = make(); pip = DD.PIP["EURUSD"]; zs = zt.iloc[:150]; out = RW.build_rewards(m1, zs, pip); R, F, RISK, V = out["R"].astype(np.float32), out["F"], out["RISK"], out["V"]
    assert R.shape == (150, A.NACT) and (R[V == 0] == 0).all()
    ent = np.repeat(np.arange(6), 5 * 7 * 2); assert (R[(F[:, ent] == 0)] == 0).all()
    assert (R >= -1.0001).all()
    n0 = A.net_r(R, F, RISK, 0.0, V); n13 = A.net_r(R, F, RISK, 1.3, V); assert np.allclose(n0[V > 0], R[V > 0])
    a = np.flatnonzero((V[0] > 0) & (F[0, ent] > 0))
    if len(a): s = A.risk_slot(a[0]); assert abs((n0[0, a[0]] - n13[0, a[0]]) - 1.3 / RISK[0].reshape(-1)[s]) < 1e-5
    assert A.decode(A.aid(3, 2, 5, 1)) == (3, 2, 5, 1) and A.NACT == 420

def test_causality_of_rewards_truncation():
    """rewards for zones resolved before the cut do not change when later bars are removed (no look-ahead beyond the horizon)."""
    m1, zt = make(); pip = DD.PIP["EURUSD"]; cut = len(m1["t"]) * 2 // 3; zs = zt[zt.ord_end + 2880 < cut].iloc[:80]; full = RW.build_rewards(m1, zs, pip); part = RW.build_rewards({k: v[:cut] for k, v in m1.items()}, zs, pip)
    assert np.array_equal(full["R"], part["R"]) and np.array_equal(full["F"], part["F"])
