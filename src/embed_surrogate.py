"""Approach 2: strategy embeddings. A surrogate maps (definition parameters, action attributes) -> expected gross R and mean 1/risk (so net R at any spread), trained on the definition corpus (Table B) using only past years,
and used to rank UNSEEN definitions/actions. Evaluated walk-forward: select on years < Y, realize on year Y. Ensemble gives uncertainty (lower-confidence-bound selection)."""
import glob, numpy as np, pandas as pd, torch, torch.nn as nn
import actions as A, definitions as DF
torch.set_num_threads(4)

def load_corpus(pattern):
    ids, params, stats = [], [], []
    for f in sorted(glob.glob(pattern)): z = np.load(f); ids.append(z["ids"]); params.append(z["params"]); stats.append(z["stats"])
    return np.concatenate(ids), np.concatenate(params), np.concatenate(stats)

def action_attrs():
    e, j, k, d = zip(*[A.decode(a) for a in range(A.NACT)]); return np.array(e), np.array(j), np.array(k), np.array(d)

class Surrogate(nn.Module):
    def __init__(self, n_par=11, hidden=256, emb=16):
        super().__init__(); self.pm = nn.Sequential(nn.Linear(n_par, hidden), nn.GELU()); self.emb = nn.ModuleList([nn.Embedding(n, emb) for n in (A.ENT, A.NSTOP, A.NTGT, A.NDIR)])
        self.mlp = nn.Sequential(nn.Linear(hidden + 4 * emb, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
    def forward(self, par, act):
        a = torch.cat([e(act[:, i]) for i, e in enumerate(self.emb)], 1); return self.mlp(torch.cat([self.pm(par), a], 1))

def window_stats(stats, years, y0, y1):
    """sum the (D, Y, 420, 4) sufficient statistics over years in [y0, y1] -> n, mean R, mean inv risk, var R."""
    sel = [i for i, y in enumerate(years) if y0 <= y <= y1]; s = stats[:, sel].sum(1); n = s[..., 0]; mR = s[..., 1] / np.maximum(n, 1); mI = s[..., 2] / np.maximum(n, 1); v = s[..., 3] / np.maximum(n, 1) - mR ** 2; return n, mR, mI, np.maximum(v, 1e-6)

def scale_params(p, mu=None, sd=None):
    if mu is None: mu, sd = p.mean(0), p.std(0) + 1e-6
    return (p - mu) / sd, mu, sd

def fit(params, stats, years, y_train_end, train_defs, n_min=30, seeds=3, epochs=15, dev="cpu", y_train_start=2005):
    n, mR, mI, _ = window_stats(stats, years, y_train_start, y_train_end); Pn, mu, sd = scale_params(params[train_defs]); e, j, k, d = action_attrs(); models = []
    nn_, mr_, mi_ = n[train_defs], mR[train_defs], mI[train_defs]; di, ai = np.nonzero(nn_ >= n_min)
    X = torch.from_numpy(Pn[di].astype(np.float32)); Aa = torch.from_numpy(np.stack([e[ai], j[ai], k[ai], d[ai]], 1)).long(); Y = torch.from_numpy(np.stack([mr_[di, ai], mi_[di, ai]], 1).astype(np.float32)); Wt = torch.from_numpy(np.sqrt(nn_[di, ai]).astype(np.float32)); Wt = Wt / Wt.mean()
    ysd = Y.std(0) + 1e-6; ymu = Y.mean(0)
    for s in range(seeds):
        torch.manual_seed(s); m = Surrogate(params.shape[1]).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3); rng = np.random.default_rng(s)
        for ep in range(epochs):
            perm = rng.permutation(len(X))
            for b in range(0, len(X), 4096):
                ix = torch.from_numpy(perm[b:b + 4096]); out = m(X[ix].to(dev), Aa[ix].to(dev)); loss = (((out - (Y[ix] - ymu) / ysd).to(dev) ** 2).sum(1) * Wt[ix].to(dev)).mean(); opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); models.append(m)
    return dict(models=models, mu=mu, sd=sd, ymu=ymu, ysd=ysd)

def predict(S, params, spread=1.3, dev="cpu"):
    """-> predicted net R (ensemble mean, sd) of shape (D, 420) at the given spread."""
    P = torch.from_numpy(((params - S["mu"]) / S["sd"]).astype(np.float32)); e, j, k, d = action_attrs(); D = len(P); outs = []
    A_ = torch.from_numpy(np.stack([e, j, k, d], 1)).long()
    with torch.no_grad():
        for m in S["models"]:
            pr = []
            for i in range(D):
                o = m(P[i:i + 1].expand(A.NACT, -1).to(dev), A_.to(dev)).cpu() * S["ysd"] + S["ymu"]; pr.append((o[:, 0] - spread * o[:, 1]).numpy())
            outs.append(np.stack(pr))
    o = np.stack(outs); return o.mean(0), o.std(0)

def walk_eval(params, stats, years, test_years, spread=1.3, topk=50, holdout_frac=0.2, n_min_train=50, n_min_test=20, lcb=1.0, seeds=3, epochs=12):
    """for each test year Y: train on years < Y using non-held-out definitions; select top-K (definition, action) among HELD-OUT definitions by (a) the surrogate LCB, (b) in-sample train mean net R, (c) random;
    report pooled realized net R in year Y (n-weighted) and how many distinct definitions were picked."""
    rng = np.random.default_rng(0); D = len(params); hold = rng.random(D) < holdout_frac; tr = np.flatnonzero(~hold); ho = np.flatnonzero(hold); rows = []
    for Y in test_years:
        S = fit(params, stats, years, Y - 1, tr, seeds=seeds, epochs=epochs); mean, sd = predict(S, params[ho], spread)
        n_tr, mR_tr, mI_tr, _ = window_stats(stats, years, 2005, Y - 1); net_tr = (mR_tr - spread * mI_tr)[ho]; n_te, mR_te, mI_te, v_te = window_stats(stats, years, Y, Y); net_te = (mR_te - spread * mI_te)[ho]; elig = (n_tr[ho] >= n_min_train) & (n_te[ho] >= n_min_test)
        def realize(score, label):
            sc = np.where(elig, score, -1e9).ravel(); top = np.argsort(-sc)[:topk]; di, ai = np.unravel_index(top, score.shape); w = n_te[ho][di, ai]; r = (net_te[di, ai] * w).sum() / w.sum(); se = np.sqrt((v_te[ho][di, ai] * w).sum()) / w.sum()
            rows.append(dict(year=Y, method=label, realized_net_R=r, se_naive=se, n_fills=float(w.sum()), n_defs=len(set(di)), pred=float(score[di, ai].mean()))); return r
        realize(mean - lcb * sd, "surrogate_lcb"); realize(mean, "surrogate_mean"); realize(net_tr, "in_sample_best")
        rr = np.where(elig, rng.random(elig.shape), -1e9).ravel(); top = np.argsort(-rr)[:topk]; di, ai = np.unravel_index(top, elig.shape); w = n_te[ho][di, ai]; rows.append(dict(year=Y, method="random_eligible", realized_net_R=(net_te[di, ai] * w).sum() / w.sum(), se_naive=np.nan, n_fills=float(w.sum()), n_defs=len(set(di)), pred=np.nan))
        # surrogate quality: rank correlation of prediction vs realized year-Y net on eligible cells
        m = elig; a = mean[m]; b = net_te[m]; rows.append(dict(year=Y, method="surrogate_rank_corr", realized_net_R=float(pd.Series(a).corr(pd.Series(b), method="spearman")), se_naive=np.nan, n_fills=float(m.sum()), n_defs=int(hold.sum()), pred=np.nan))
        print("year", Y, {r["method"]: round(r["realized_net_R"], 4) for r in rows[-5:]}, flush=True)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--corpus", required=True, help="glob of corpus_*.npz from definitions.py"); ap.add_argument("--out", default="surrogate_walk.csv"); ap.add_argument("--test_years", default="2013-2025"); ap.add_argument("--topk", type=int, default=50); ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(); ids, params, stats = load_corpus(a.corpus); years = list(range(2005, 2027)); y0, y1 = [int(x) for x in a.test_years.split("-")]
    res = walk_eval(params, stats, years, list(range(y0, y1 + 1)), topk=a.topk, seeds=a.seeds); res.to_csv(a.out, index=False); print(res.groupby("method").realized_net_R.agg(["mean", "std", "count"]).round(4))
