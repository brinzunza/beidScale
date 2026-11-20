import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import actions as A, definitions as DF, embed_surrogate as ES

def synth(D=300, seed=0):
    rng = np.random.default_rng(seed); params = np.stack([DF.param_vector(DF.sample_params(i)) for i in range(D)]); years = np.arange(2005, 2027); Y = len(years); e, j, k, d = ES.action_attrs()
    # planted truth: mean gross R depends on (min_size_atr, pb_min) x (entry depth, target): a hidden smooth function; noise shrinks with n
    f = 0.08 * (params[:, 6:7] > 0.2) * (e[None, :] == 2) * (k[None, :] == 1) - 0.03 * (params[:, 5:6] > 3) * (d[None, :] == 1) - 0.01
    stats = np.zeros((D, Y, A.NACT, 4)); n = rng.integers(30, 200, (D, Y, A.NACT)).astype(float)
    mR = f[:, None, :] + rng.standard_normal((D, Y, A.NACT)) / np.sqrt(n) * 0.9; stats[..., 0] = n; stats[..., 1] = n * mR; stats[..., 2] = n * 0.02; stats[..., 3] = n * (0.9 ** 2 + mR ** 2); return params, stats, years

def test_surrogate_recovers_planted_structure_and_beats_random():
    params, stats, years = synth(); res = ES.walk_eval(params, stats, years, [2016, 2020], topk=30, seeds=2, epochs=8)
    for y in (2016, 2020):
        r = res[res.year == y].set_index("method"); assert r.loc["surrogate_rank_corr", "realized_net_R"] > 0.03, r      # sparse planted structure: only a few cells carry signal, so the all-cell rank correlation is small
        assert r.loc["surrogate_lcb", "realized_net_R"] > r.loc["random_eligible", "realized_net_R"] + 0.04, r
