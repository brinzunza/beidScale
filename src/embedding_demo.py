"""Embedding dataset example + diagnostics (small, runs locally in a few minutes): toy Table A (EURUSD+GBPUSD 15m zones 2010-2015), tiny BEIDNet trained on years < 2014,
embeddings of the 2015 events, k-means clusters, win rate of a fixed action (50% limit, far-edge stop, 1R target, with the zone) per cluster, and cosine similarity vs win-rate agreement.
Outputs in results/embedding_demo/: embedding_dataset_example.parquet, cluster_stats.csv, fig1_clusters.png, fig2_cluster_winrate_vs_cosine.png, fig3_cosine_vs_winrate.png"""
import sys, os, json, numpy as np, pandas as pd, torch
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as st
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import data as DD, build_table as BT, train as TR, windows as W, model as M, actions as A, rewardview as RV
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / "embedding_demo"; OUT.mkdir(parents=True, exist_ok=True); WORK = OUT / "work"; PAIRS = ["EURUSD", "GBPUSD"]; Y = 2015; ACT = A.aid(2, 0, 1, 0); K = 8

def build_and_train():
    tdir = WORK / "ds" / "tableA"
    if not (tdir / "zones.parquet").exists():
        for p in PAIRS: BT.shard(p, 15, "ny", str(WORK / "ds"), start="2010-01-01", end="2016-01-01")
        BT.merge(str(WORK / "ds"))
    if not (WORK / "run" / f"model_{Y}.pt").exists():
        args = TR.parser().parse_args(["--table", str(tdir), "--out", str(WORK / "run"), "--size", "tiny", "--test_year", str(Y), "--epochs", "5", "--batch", "128", "--bar_pairs", *PAIRS, "--bars_start", "2010-01-01", "--bars_end", "2016-01-01", "--log_every", "20"]); TR.run(args)
    return tdir

def extract(tdir):
    T = TR.Table(tdir); store = W.BarStore(pairs=tuple(PAIRS), start="2010-01-01", end="2016-01-01").to(torch.device("cpu")); T.set_store(store); ck = torch.load(WORK / "run" / f"model_{Y}.pt", weights_only=False); mu, sd = ck["mu"], ck["sd"]
    net = M.BEIDNet(ck["size"], n_ctx=T.geo.shape[1] + T.X.shape[1]); net.load_state_dict(ck["state"]); net.eval(); rows = np.flatnonzero(T.year == Y); E = []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            idx, w1, w2, x, cat, netR, V = T.batch(rows[i:i + 256], torch.device("cpu"), mu, sd, 1.0); E.append(net.encode(w1, w2, x, cat).numpy())
    E = np.concatenate(E); rows_sorted = np.sort(rows); R, V, F, RISK = RV.RewardSource(tdir, T.z).get(rows_sorted); filled = (F[:, 2] > 0) & (V[:, ACT] > 0); r = R[:, ACT]
    ctx = np.concatenate([T.geo[rows_sorted], np.nan_to_num((np.asarray(T.X[rows_sorted]) - mu) / sd).clip(-6, 6)], 1); z = T.z.iloc[rows_sorted].reset_index(drop=True)
    return z, E, ctx, filled, r

def wilson(k, n, zc=1.96):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + zc ** 2 / n; c = (p + zc ** 2 / (2 * n)) / d; h = zc * np.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n ** 2)) / d; return c - h, c + h

def analyze(name, Z, win, R, rng):
    """Z: embedding (n,d) for FILLED events. Returns dict with clusters, local win rate, neighbor pairs table, cluster stats."""
    Z = Z - Z.mean(0)   # mean-centre first: raw transformer embeddings share a large common component (cosine ~0.99 between any two events)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9); km = KMeans(K, n_init=10, random_state=0).fit(Zn); lab = km.labels_; cen = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True); cos_cen = (Zn * cen[lab]).sum(1)
    rows = []
    for c in range(K):
        m = lab == c; n = int(m.sum()); k = int(win[m].sum()); lo, hi = wilson(k, n); rows.append(dict(embedding=name, cluster=c, n=n, win_rate=k / n, ci_lo=lo, ci_hi=hi, mean_R=float(R[m].mean()), mean_cos_to_centroid=float(cos_cen[m].mean())))
    cs = pd.DataFrame(rows); chi = st.chi2_contingency(np.array([[win[lab == c].sum(), (lab == c).sum() - win[lab == c].sum()] for c in range(K)]))[1]
    nn = NearestNeighbors(n_neighbors=21, metric="cosine").fit(Zn); dist, ind = nn.kneighbors(Zn); sim = 1 - dist[:, 1:]; ind = ind[:, 1:]; same = (win[:, None] == win[ind]).astype(float)
    local = win[ind].mean(1); auc = roc_auc_score(win, local); rho = st.spearmanr(R, R[ind].mean(1)).statistic
    cc = cen @ cen.T; wr = cs.win_rate.values; iu = np.triu_indices(K, 1); pairs_c = pd.DataFrame(dict(cos=cc[iu], dwin=np.abs(wr[:, None] - wr[None, :])[iu]))
    return dict(name=name, Zn=Zn, lab=lab, cs=cs, chi_p=chi, sim=sim, same=same, local=local, auc=auc, rho=rho, ind=ind, pairs_c=pairs_c, cos_cen=cos_cen)

def agree_curve(res, win, nbins=10, nperm=100, rng=None):
    sim = res["sim"].ravel(); same = res["same"].ravel(); q = np.quantile(sim, np.linspace(0, 1, nbins + 1)); b = np.clip(np.digitize(sim, q[1:-1]), 0, nbins - 1); obs = np.array([same[b == i].mean() for i in range(nbins)]); mid = np.array([sim[b == i].mean() for i in range(nbins)])
    null = []
    for _ in range(nperm):
        w = rng.permutation(win); s2 = (w[:, None] == w[res["ind"]]).astype(float).ravel(); null.append([s2[b == i].mean() for i in range(nbins)])
    null = np.array(null); return mid, obs, np.quantile(null, 0.025, 0), np.quantile(null, 0.975, 0)

def main():
    tdir = build_and_train(); z, E, ctx, filled, r = extract(tdir); rng = np.random.default_rng(0); f = np.flatnonzero(filled); win = (r[f] > 0).astype(int); R = r[f]; zf = z.iloc[f].reset_index(drop=True)
    print(f"2015 events {len(z)}  filled+valid {len(f)}  win rate {win.mean():.3f}  mean gross R {R.mean():+.3f}")
    # baseline embedding: PCA(16) of the same context features the model saw (geometry + 46 day factors), standardized with 2015 scale
    pca = PCA(16, random_state=0).fit_transform((ctx[f] - ctx[f].mean(0)) / (ctx[f].std(0) + 1e-6)); res = {"BEIDNet embedding (96-d, centred)": analyze("BEIDNet embedding (96-d, centred)", E[f], win, R, rng), "PCA-16 of hand features": analyze("PCA-16 of hand features", pca, win, R, rng)}
    names = list(res); cs = pd.concat([res[n]["cs"] for n in names]); cs.to_csv(OUT / "cluster_stats.csv", index=False)
    # dataset example
    ds = zf[["pair", "t_ev", "side", "ztype", "ny_hour", "dow", "top", "bot", "atr15"]].copy(); ds["size_atr"] = (ds.top - ds.bot) / ds.atr15; ds["pair"] = ds.pair.map({0: "EURUSD", 1: "GBPUSD"}); ds["t_ev"] = pd.to_datetime(ds.t_ev); ds["win"] = win; ds["gross_R"] = R
    for n, key in zip(names, ("model", "pca")): ds[f"cluster_{key}"] = res[n]["lab"]
    emb = pd.DataFrame(E[f].astype(np.float32), columns=[f"emb_{i:03d}" for i in range(E.shape[1])]); pd.concat([ds.reset_index(drop=True), emb], axis=1).to_parquet(OUT / "embedding_dataset_example.parquet", index=False)
    cmap = plt.get_cmap("tab10")
    # ---- fig 1: t-SNE
    fig, ax = plt.subplots(2, 2, figsize=(13, 11)); base = win.mean()
    for i, n in enumerate(names):
        Zn = res[n]["Zn"]; xy = TSNE(2, perplexity=35, init="pca", random_state=0, metric="cosine").fit_transform(Zn); a = ax[i, 0]; a.scatter(xy[:, 0], xy[:, 1], c=[cmap(c) for c in res[n]["lab"]], s=9, alpha=0.8); a.set_title(f"{n}\nk-means clusters (k={K}, cosine geometry)", fontsize=10)
        for c in range(K): m = res[n]["lab"] == c; a.text(*np.median(xy[m], 0), str(c), fontsize=13, weight="bold", ha="center", bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        b = ax[i, 1]; sc = b.scatter(xy[:, 0], xy[:, 1], c=res[n]["local"], cmap="RdBu_r", vmin=base - 0.2, vmax=base + 0.2, s=9); b.set_title(f"{n}\nlocal win rate of the 20 nearest neighbours (cosine); base {base:.2f}", fontsize=10); plt.colorbar(sc, ax=b, fraction=0.046)
        for a_ in (a, b): a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("BEID zone events, 2015 test year: 15m zones, EURUSD+GBPUSD, action = 50% limit / far-edge stop / 1R target (win = gross R > 0)", fontsize=11); fig.tight_layout(); fig.savefig(OUT / "fig1_clusters.png", dpi=130); plt.close(fig)
    # ---- fig 2: cluster win rate and intra-cluster cosine
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, n in enumerate(names):
        c = res[n]["cs"].sort_values("n", ascending=False).reset_index(drop=True); a = ax[i]; x = np.arange(K); a.bar(x, c.win_rate, color=[cmap(k) for k in c.cluster], alpha=0.85); a.errorbar(x, c.win_rate, yerr=[c.win_rate - c.ci_lo, c.ci_hi - c.win_rate], fmt="none", ecolor="k", capsize=3); a.axhline(base, color="k", ls="--", lw=1)
        a.set_xticks(x); a.set_xticklabels([f"c{k}\nn={m}" for k, m in zip(c.cluster, c.n)], fontsize=8); a.set_ylim(0.2, 0.8); a.set_ylabel("win rate (95% Wilson CI)"); a2 = a.twinx(); a2.plot(x, c.mean_cos_to_centroid, "o-", color="crimson"); a2.set_ylabel("mean cosine similarity to cluster centroid", color="crimson"); a2.set_ylim(0, 1)
        a.set_title(f"{n}\nchi-square test of equal win rates across clusters: p = {res[n]['chi_p']:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig2_cluster_winrate_vs_cosine.png", dpi=130); plt.close(fig)
    # ---- fig 3: cosine similarity vs win-rate agreement
    fig, ax = plt.subplots(1, 3, figsize=(17, 5)); exp = base ** 2 + (1 - base) ** 2
    for i, n in enumerate(names):
        mid, obs, lo, hi = agree_curve(res[n], win, rng=rng); a = ax[i]; a.fill_between(mid, lo, hi, color="grey", alpha=0.3, label="95% band if win/loss were unrelated to embedding"); a.plot(mid, obs, "o-", label="observed"); a.axhline(exp, color="k", ls="--", lw=1, label=f"chance agreement {exp:.3f}")
        a.set_xlabel("cosine similarity between an event and its neighbour (binned)"); a.set_ylabel("P(neighbour has the same win/loss outcome)"); a.set_title(f"{n}\nkNN win-rate AUC {res[n]['auc']:.3f}, Spearman(R, neighbours' mean R) {res[n]['rho']:+.3f}", fontsize=10); a.legend(fontsize=7)
    a = ax[2]
    for n, col in zip(names, ("tab:blue", "tab:orange")):
        p = res[n]["pairs_c"]; rr = st.spearmanr(p.cos, p.dwin).statistic; a.scatter(p.cos, p.dwin, s=28, alpha=0.8, color=col, label=f"{n}: Spearman {rr:+.2f} (28 cluster pairs)")
    a.set_xlabel("cosine similarity between cluster centroids"); a.set_ylabel("|win rate difference| between the two clusters"); a.set_title("clusters: similar embeddings, similar win rate?", fontsize=10); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig3_cosine_vs_winrate.png", dpi=130); plt.close(fig)
    print(cs.round(3).to_string(index=False)); print({n: dict(chi_p=round(res[n]["chi_p"], 3), knn_auc=round(res[n]["auc"], 3), spearman=round(res[n]["rho"], 3)) for n in names})
if __name__ == "__main__":
    main()
