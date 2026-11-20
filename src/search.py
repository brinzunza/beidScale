"""Approach 2, search + verification. propose: rank UNSEEN sampled definitions x actions with the surrogate (lower-confidence bound) using data up to a selection year.
verify: run the real engine for the proposed (definition, action) candidates on the held-out years and report realized trade-level net R (month-cluster t) across the spread sweep.
  python search.py propose --corpus 'DIR/corpus_*.npz' --select_year 2017 --n_new 3000 --topk 40 --out cand.json
  python search.py verify --cand cand.json --years 2018-2026 --out verify.csv"""
import argparse, json, numpy as np, pandas as pd
import data as DD, zones as Z, rewards as RW, actions as A, definitions as DF, embed_surrogate as ES, evaluate as EV

def propose(a):
    ids, params, stats = ES.load_corpus(a.corpus); years = list(range(2005, 2027)); S = ES.fit(params, stats, years, a.select_year, np.arange(len(params)), seeds=3, epochs=12); new_ids = np.arange(10 ** 6, 10 ** 6 + a.n_new)
    P = np.stack([DF.param_vector(DF.sample_params(int(i))) for i in new_ids]); mean, sd = ES.predict(S, P, a.spread); score = mean - a.lcb * sd; top = np.argsort(-score.ravel())[:a.topk]; di, ai = np.unravel_index(top, score.shape)
    cand = [dict(def_id=int(new_ids[d]), action=int(x), action_name=A.name(int(x)), pred_net=float(mean[d, x]), pred_sd=float(sd[d, x]), params=DF.sample_params(int(new_ids[d])).__dict__) for d, x in zip(di, ai)]
    json.dump(dict(select_year=a.select_year, spread=a.spread, cand=cand), open(a.out, "w"), indent=1); print(f"{len(cand)} candidates, predicted net R {np.mean([c['pred_net'] for c in cand]):+.4f}")

def realized(def_id, action, y0, y1, pairs=DD.PAIRS, spreads=(0.0, 0.5, 1.0, 1.3, 1.6), start=None, end=None):
    p = DF.sample_params(def_id); parts = []
    for pair in pairs:
        m1 = DD.load_m1(pair, start, end); zt = Z.build_zone_table(m1, p); zt = zt[(zt.year >= y0) & (zt.year <= y1)].reset_index(drop=True)
        if len(zt) == 0: continue
        rw = RW.build_rewards(m1, zt, DD.PIP[pair]); e = A.entry_of(action); i = np.arange(len(zt)); ok = (rw["F"][:, e] > 0) & (rw["V"][:, action] > 0); risk = rw["RISK"].reshape(len(zt), -1)[:, A.risk_slot(action)]
        parts.append(pd.DataFrame(dict(t_ev=zt.t_ev.values[ok], year=zt.year.values[ok], pair=pair, R=rw["R"][:, action].astype(np.float32)[ok], risk=risk[ok])))
    if not parts: return None
    d = pd.concat(parts, ignore_index=True); rows = []
    for sp in spreads:
        net = d.R.values - sp / d.risk.values; mo = pd.to_datetime(d.t_ev.values).to_period("M").astype(str).values; m, t = EV.cluster_t(net, mo); rows.append(dict(spread=sp, n=len(d), mean_net_R=m, t=t, win=float((net > 0).mean()), pos_years=float((pd.Series(net).groupby(d.year.values).mean() > 0).mean())))
    return rows

def verify(a):
    c = json.load(open(a.cand)); y0, y1 = [int(x) for x in a.years.split("-")]; out = []
    for k, x in enumerate(c["cand"]):
        r = realized(x["def_id"], x["action"], y0, y1, start=a.start, end=a.end)
        if r: out += [dict(rank=k, def_id=x["def_id"], action=x["action_name"], pred_net=x["pred_net"], **q) for q in r]
        print(k, x["action_name"], "pred", round(x["pred_net"], 4), "realized@1.3", [round(q["mean_net_R"], 4) for q in (r or []) if q["spread"] == 1.3], flush=True)
    pd.DataFrame(out).to_csv(a.out, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--corpus"); ap.add_argument("--select_year", type=int, default=2017); ap.add_argument("--n_new", type=int, default=3000); ap.add_argument("--topk", type=int, default=40); ap.add_argument("--spread", type=float, default=1.3); ap.add_argument("--lcb", type=float, default=1.0)
    ap.add_argument("--cand"); ap.add_argument("--years", default="2018-2026"); ap.add_argument("--out", default="out.json"); ap.add_argument("--start"); ap.add_argument("--end"); a = ap.parse_args(); {"propose": propose, "verify": verify}[a.cmd](a)
