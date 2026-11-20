"""Table A reward builder: for every zone event and every one of the 420 actions, the gross R it would have produced on the M1 path. Run once (numba, parallel); training never re-simulates.
Conventions: limit entries fill on the first M1 bar (>= event, < order end) whose low touches the level, or trades through it by `fill_pen_pips` (fill price = level); the fill bar can only lose; a bar reaching both stop and target resolves as stop;
orders cancel at the order end; filled trades are closed at market after `horizon` bars; market entry fills at the open of the first bar after the event. Fade = opposite direction at the same entry price (stop-order semantics),
with stop/target distances mirrored. Actions with risk < min_risk pips are invalid, as are zone-top / 2H targets closer than min_risk pips (R-multiple targets scale with the risk and stay valid)."""
import os, numpy as np
from numba import njit, prange
import actions as A
INF = 1 << 60
MULTS = np.array(A.MULT); DEPTHS = np.array(A.DEPTH)

@njit(cache=True)
def _aid(e, j, k, d):
    return ((e * 5 + j) * 7 + k) * 2 + d

@njit(cache=True)
def _scan(sg, h, l, c, o, s_adv, s_nxt, hend, E, offs, order, nfin):
    """first-passage bar indices for 32 offsets above/below E. adv = counted from s_adv, nxt = counted from s_nxt. Returns up_adv, dn_adv, up_nxt, dn_nxt (int64 arrays, INF if never) and the close at hend."""
    up_adv = np.full(32, INF, np.int64); dn_adv = np.full(32, INF, np.int64); up_nxt = np.full(32, INF, np.int64); dn_nxt = np.full(32, INF, np.int64)
    pua = 0; pda = 0; pun = 0; pdn = 0; mxa = -1e300; mna = -1e300; mxn = -1e300; mnn = -1e300
    for i in range(s_adv, hend + 1):
        if sg > 0: hi = h[i]; lo = l[i]
        else: hi = -l[i]; lo = -h[i]
        u = hi - E; d = E - lo
        if u > mxa: mxa = u
        if d > mna: mna = d
        while pua < nfin and offs[order[pua]] <= mxa: up_adv[order[pua]] = i; pua += 1
        while pda < nfin and offs[order[pda]] <= mna: dn_adv[order[pda]] = i; pda += 1
        if i >= s_nxt:
            if u > mxn: mxn = u
            if d > mnn: mnn = d
            while pun < nfin and offs[order[pun]] <= mxn: up_nxt[order[pun]] = i; pun += 1
            while pdn < nfin and offs[order[pdn]] <= mnn: dn_nxt[order[pdn]] = i; pdn += 1
        if pua >= nfin and pda >= nfin and pun >= nfin and pdn >= nfin: break
    cend = c[hend] if sg > 0 else -c[hend]
    return up_adv, dn_adv, up_nxt, dn_nxt, cend

@njit(parallel=True, cache=True)
def reward_kernel(o, h, l, c, ev, ord_end, side, top, bot, atr, pip, min_risk_pips, horizon, fill_pen_pips):
    n = len(ev); N = len(o); R = np.zeros((n, A.NACT), np.float32); F = np.zeros((n, A.ENT), np.uint8); RISK = np.full((n, A.ENT, A.NSTOP), np.nan, np.float32); V = np.zeros((n, A.NACT), np.uint8); FIL = np.full((n, A.ENT), -1, np.int32)
    minr = min_risk_pips * pip; pen = fill_pen_pips * pip                    # limit orders fill only when price trades THROUGH the level by `pen` (0 = touch)
    for z in prange(n):
        sg = side[z]
        if sg > 0: tp = top[z]; bt = bot[z]
        else: tp = -bot[z]; bt = -top[z]
        H = tp - bt; az = atr[z]
        for e in range(A.ENT):
            if e < 5:
                E = tp - DEPTHS[e] * H; f = -1
                for i in range(ev[z], ord_end[z]):
                    lo = l[i] if sg > 0 else -h[i]
                    if lo <= E - pen: f = i; break
                s_adv = f; s_nxt = f + 1
            else:
                E = o[ev[z]] if sg > 0 else -o[ev[z]]; f = ev[z]; s_adv = f; s_nxt = f
            stops = np.empty(5); stops[0] = bt; stops[1] = bt - 0.25 * H; stops[2] = bt - 0.5 * H; stops[3] = E - az; stops[4] = E - 2.0 * az
            offs = np.full(32, 1e300); okj = np.zeros(5, np.bool_)
            for j in range(5):
                r = E - stops[j]
                if r >= minr: okj[j] = True; offs[j] = r; RISK[z, e, j] = r / pip
                for k in range(5):
                    if okj[j]: offs[5 + j * 5 + k] = MULTS[k] * r
            tz = tp - E
            if tz >= minr: offs[30] = tz
            if 2.0 * H >= minr: offs[31] = 2.0 * H
            for j in range(5):
                for k in range(7):
                    ok = okj[j]
                    if k == 5 and not (tz >= minr): ok = False
                    if k == 6 and not (2.0 * H >= minr): ok = False
                    if ok:
                        V[z, _aid(e, j, k, 0)] = 1; V[z, _aid(e, j, k, 1)] = 1
            if f < 0: continue
            F[z, e] = 1; FIL[z, e] = f - ev[z]; hend = min(f + horizon, N - 1)
            order = np.argsort(offs); nfin = 0
            for q in range(32):
                if offs[q] < 1e299: nfin += 1
            ua, da, un, dn, cend = _scan(sg, h, l, c, o, s_adv, s_nxt, hend, E, offs, order, nfin)
            for j in range(5):
                if not okj[j]: continue
                r = offs[j]
                for k in range(7):
                    if k < 5: ti = 5 + j * 5 + k
                    elif k == 5: ti = 30
                    else: ti = 31
                    if offs[ti] > 1e299: continue
                    t = offs[ti]
                    tS = da[j]; tT = un[ti]                                   # with the zone: stop below, target above
                    if tS <= tT and tS < INF: R[z, _aid(e, j, k, 0)] = -1.0
                    elif tT < INF: R[z, _aid(e, j, k, 0)] = t / r
                    else: R[z, _aid(e, j, k, 0)] = (cend - E) / r
                    tS = ua[j]; tT = dn[ti]                                   # fade: stop above, target below
                    if tS <= tT and tS < INF: R[z, _aid(e, j, k, 1)] = -1.0
                    elif tT < INF: R[z, _aid(e, j, k, 1)] = t / r
                    else: R[z, _aid(e, j, k, 1)] = (E - cend) / r
    return R, F, RISK, V, FIL

def build_rewards(m1, zt, pip, min_risk_pips=1.5, horizon=2880, chunk=50000, fill_pen_pips=None):
    """zt: zone table from zones.build_zone_table. Returns dict of arrays (float16 R, uint8 F, float32 RISK pips (n,6,5), uint8 V, int32 FIL)."""
    if fill_pen_pips is None: fill_pen_pips = float(os.environ.get("BEID_FILL_PEN", "0"))          # one knob for Table A, Table B and the verification backtests
    outs = {k: [] for k in ("R", "F", "RISK", "V", "FIL")}
    for a in range(0, len(zt), chunk):
        z = zt.iloc[a:a + chunk]
        R, F, RISK, V, FIL = reward_kernel(m1["o"], m1["h"], m1["l"], m1["c"], z.ev.values.astype(np.int64), z.ord_end.values.astype(np.int64), z.side.values.astype(np.int64), z.top.values, z.bot.values, z.atr15.values, pip, min_risk_pips, horizon, float(fill_pen_pips))
        for k, v in (("R", R.astype(np.float16)), ("F", F), ("RISK", RISK), ("V", V), ("FIL", FIL)): outs[k].append(v)
    return {k: np.concatenate(v) if v else None for k, v in outs.items()}
