"""Evaluation of learned policies on Table A test rows: realized gross/net R across the spread sweep, month-cluster t, trades/yr, Kelly K, positive-year share, usage mix. Baselines on the same rows."""
import os, numpy as np, pandas as pd
from pathlib import Path
import data as DD, actions as A, rewardview as RV
SWEEP = (0.0, 0.5, 1.0, 1.3, 1.6, 2.0)        # 2.0 = stress for the wider spreads of the earlier years

def cluster_t(x, months):
    x = np.asarray(x, float); u, inv = np.unique(months, return_inverse=True); S = np.bincount(inv, x); N = np.bincount(inv); m = S.sum() / len(x); se = np.sqrt(((S - m * N) ** 2).sum()) / len(x); return m, (m / se if se > 0 else np.nan)

class TableView:
    def __init__(self, d, rows, null_shift=0, plant=0.0):
        d = Path(d); zall = pd.read_parquet(d / "zones.parquet"); self.z = zall.iloc[rows].reset_index(drop=True); rows = np.asarray(rows)
        self.R, self.V, self.F, self.RISK = RV.RewardSource(d, zall, null_shift, plant).get(rows); self.rows = rows
    def realized(self, act, spread):
        """gross/net R and filled flag for chosen actions act (n,) with -1 = skip."""
        n = len(act); i = np.arange(n); a = np.where(act >= 0, act, 0); e = A.entry_of(a); filled = (self.F[i, e] > 0) & (act >= 0) & (self.V[i, a] > 0); gross = np.where(filled, self.R[i, a], 0.0)
        risk = self.RISK.reshape(n, -1)[i, A.risk_slot(a)]
        with np.errstate(divide="ignore", invalid="ignore"): cost = np.where(filled, spread / risk, 0.0)
        return gross, gross - np.nan_to_num(cost), filled, np.nan_to_num(risk)

def dedupe(z, mask):
    """one position per (pair, side, FX day): keep the earliest chosen event. With BEID_BLOCK_HOURS=h > 0 instead: after a chosen event no other event on the same (pair, side) is taken for h hours (set 48 to forbid overlapping holds, since trades close within 48 h)."""
    bh = float(os.environ.get("BEID_BLOCK_HOURS", "0"))
    if bh <= 0:
        _, fx = DD.ny_parts(z.t_ev.values); key = pd.DataFrame({"p": z.pair.values, "s": z.side.values, "d": fx, "i": np.arange(len(z))})[mask]; first = key.sort_values("i").drop_duplicates(["p", "s", "d"])["i"].values; out = np.zeros(len(z), bool); out[first] = True; return out
    t = z.t_ev.values.astype(np.int64); out = np.zeros(len(z), bool); blk = int(bh * 3600 * 10 ** 9); g = pd.DataFrame({"p": z.pair.values, "s": z.side.values, "i": np.arange(len(z))})[mask]
    for _, grp in g.groupby(["p", "s"]):
        last = -(1 << 62)
        for i in grp.i.values[np.argsort(t[grp.i.values], kind="stable")]:
            if t[i] - last >= blk: out[i] = True; last = t[i]
    return out

def summarize(tv, act, label, spreads=SWEEP, years=None):
    z = tv.z; chosen = act >= 0; keep = dedupe(z, chosen); act = np.where(keep, act, -1); rows = []
    for sp in spreads:
        g, nr, fl, risk = tv.realized(act, sp); m = fl; n = int(m.sum())
        if n < 30: rows.append(dict(rule=label, spread=sp, n_trades=n)); continue
        months = pd.to_datetime(z.t_ev.values[m]).to_period("M").astype(str).values; mean, t = cluster_t(nr[m], months); yr = pd.Series(nr[m]).groupby(z.year.values[m]).mean(); ny = max(z.year.nunique(), 1)
        st = A.stop_index_of(act[m]); tg = (act[m] // A.NDIR) % A.NTGT; var = nr[m].var()
        rows.append(dict(rule=label, spread=sp, n_orders=int(keep.sum()), n_trades=n, fill_rate=float(n / max(keep.sum(), 1)), mean_net_R=mean, t=t, mean_gross_R=float(g[m].mean()), win=float((nr[m] > 0).mean()), trades_per_year=n / ny,
                         pos_years=float((yr > 0).mean()), K=float((n / ny) * mean ** 2 / var) if (var > 0 and mean > 0) else 0.0, median_risk_pips=float(np.median(risk[m])), share_wide_stop=float((st >= 3).mean()), share_fade=float((act[m] % A.NDIR == 1).mean()), share_ge2R=float(np.isin(tg, [3, 4, 6]).mean())))
    return rows

def decisions(pred, rule, thr=None):
    q = pred["q"].astype(np.float32); pol = pred["pol"].astype(np.float32); skip = pol[:, A.NACT]; trade_best = pol[:, :A.NACT].argmax(1); margin = pol[:, :A.NACT].max(1) - skip
    if rule == "greedy": return np.where(pol.argmax(1) < A.NACT, pol.argmax(1), -1), margin
    if rule == "greedyQ": qq = np.where(pred["valid"] > 0, q, -1e9); a = qq.argmax(1); return np.where(qq.max(1) > 0, a, -1), qq.max(1)
    if rule.startswith("top"):
        return np.where(margin >= thr, trade_best, -1), margin
    raise ValueError(rule)

def evaluate_run(table_dir, pred_files, spreads=SWEEP, top_qs=(0.01, 0.03), null_shift=0, plant=0.0):
    """pred_files: chronological list of pred_<year>.npz. Top-q thresholds use the PREVIOUS file's margin quantile (first file: top-q rules skipped)."""
    out = []; prev_margin = None
    for f in pred_files:
        P = dict(np.load(f)); tv = TableView(table_dir, P["rows"], null_shift, plant); P["valid"] = tv.V; yr = int(tv.z.year.iloc[0])
        for rule in ("greedy", "greedyQ"):
            act, _ = decisions(P, rule); r = summarize(tv, act, rule, spreads); [x.update(year=yr, file=str(f)) for x in r]; out += r
        _, margin = decisions(P, "greedy")
        if prev_margin is not None:
            for qv in top_qs:
                thr = np.quantile(prev_margin, 1 - qv); act, _ = decisions(P, "top", thr); r = summarize(tv, act, f"top{int(round(qv*100))}pct", spreads); [x.update(year=yr, file=str(f)) for x in r]; out += r
        prev_margin = margin
    df = pd.DataFrame(out)
    if df.empty: return df
    # pooled-over-years summary (re-pool trades through a second pass)
    return df

def pooled(table_dir, pred_files, spreads=SWEEP, top_qs=(0.01, 0.03), null_shift=0, plant=0.0):
    """pool all test years' decisions per rule and summarize once (month-cluster t over the whole test period)."""
    parts = {}; prev = None
    for f in pred_files:
        P = dict(np.load(f)); tv = TableView(table_dir, P["rows"], null_shift, plant); P["valid"] = tv.V
        for rule in ("greedy", "greedyQ"): parts.setdefault(rule, []).append((tv, decisions(P, rule)[0]))
        _, margin = decisions(P, "greedy")
        if prev is not None:
            for qv in top_qs: parts.setdefault(f"top{int(round(qv*100))}pct", []).append((tv, decisions(P, "top", np.quantile(prev, 1 - qv))[0]))
        prev = margin
    rows = []
    for rule, lst in parts.items():
        class _TV: pass
        tv = _TV(); tv.z = pd.concat([t.z for t, _ in lst], ignore_index=True); tv.R = np.concatenate([t.R for t, _ in lst]); tv.V = np.concatenate([t.V for t, _ in lst]); tv.F = np.concatenate([t.F for t, _ in lst]); tv.RISK = np.concatenate([t.RISK for t, _ in lst])
        tv.realized = TableView.realized.__get__(tv); act = np.concatenate([a for _, a in lst]); rows += summarize(tv, act, rule, spreads)
    return pd.DataFrame(rows)

def baselines(table_dir, train_rows, test_rows, spread_select=1.0, spreads=SWEEP, seed=0):
    """best fixed action chosen on training rows (mean net R at spread_select over filled valid events), a random policy, and always-skip, evaluated on the test rows."""
    trv = TableView(table_dir, train_rows); n = len(train_rows); best, bestv = None, -1e9
    ent = np.repeat(np.arange(A.ENT), A.NSTOP * A.NTGT * A.NDIR); slot = A.risk_slot(np.arange(A.NACT)); risk = trv.RISK.reshape(n, -1)[:, slot]
    with np.errstate(divide="ignore", invalid="ignore"): net = trv.R - np.where(trv.F[:, ent] > 0, spread_select / risk, 0.0)
    ok = (trv.V > 0) & (trv.F[:, ent] > 0); s = np.where(ok, net, 0.0).sum(0); c = ok.sum(0); mean = np.where(c > 200, s / np.maximum(c, 1), -1e9); a = int(mean.argmax())
    tev = TableView(table_dir, test_rows); rows = summarize(tev, np.full(len(test_rows), a), f"best_fixed:{A.name(a)}", spreads)
    rng = np.random.default_rng(seed); ra = rng.integers(0, A.NACT, len(test_rows)); rows += summarize(tev, np.where(tev.V[np.arange(len(ra)), ra] > 0, ra, -1), "random_valid", spreads); return pd.DataFrame(rows)
