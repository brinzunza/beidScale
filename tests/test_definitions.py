import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import data as DD, zones as Z, rewards as RW, actions as A, definitions as DF

def test_aggregate_matches_direct_net_r():
    m1 = DD.load_m1("EURUSD", start="2012-03-01", end="2012-05-01"); zt = Z.build_zone_table(m1, Z.ZoneParams(tf=15)); rw = RW.build_rewards(m1, zt, DD.PIP["EURUSD"]); agg = DF.aggregate(zt, rw)
    yi = 2012 - DF.Y0; n = agg[yi, :, 0]; mean_r = agg[yi, :, 1] / np.maximum(n, 1); mean_inv = agg[yi, :, 2] / np.maximum(n, 1)
    net = A.net_r(rw["R"].astype(np.float32), rw["F"], rw["RISK"], 1.3, rw["V"]); ent = np.repeat(np.arange(6), 70); ok = (rw["V"] > 0) & (rw["F"][:, ent] > 0); direct = np.where(ok, net, 0).sum(0) / np.maximum(ok.sum(0), 1)
    assert np.allclose(n, ok.sum(0)); assert np.allclose(mean_r - 1.3 * mean_inv, direct, atol=2e-3)

def test_params_sampling_is_deterministic_and_varied():
    a = [DF.sample_params(i) for i in range(50)]; b = [DF.sample_params(i) for i in range(50)]; assert [x.key() for x in a] == [x.key() for x in b]; assert len({x.key() for x in a}) > 40
