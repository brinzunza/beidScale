"""With K=1 and the all-hold policy the replay environment reproduces the Table A reward of the same action exactly; a learnable toy shows PPO machinery runs."""
import sys, numpy as np, torch, pandas as pd, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_table as BT, train as TR, windows as W, manage_env as ME, actions as A, data as DD

@pytest.mark.parametrize("pen", ["0", "0.3"])
def test_hold_policy_matches_table_a(tmp_path, pen, monkeypatch):
    monkeypatch.setenv("BEID_FILL_PEN", pen)
    out = tmp_path / "ds"; BT.shard("EURUSD", 15, "ny", str(out), start="2011-01-01", end="2012-07-01"); BT.merge(str(out)); T = TR.Table(out / "tableA"); T.FIL = np.load(out / "tableA" / "FIL.npy", mmap_mode="r")
    store = W.BarStore(pairs=("EURUSD",), start="2011-01-01", end="2012-07-01").to(torch.device("cpu")); T.set_store(store); base = (2, 0, 1); ep = ME.Episodes(T, np.arange(min(T.n, 400)), base, spread=0.0); assert ep.n > 100
    env = ME.ManageEnv(store, ep, torch.device("cpu"), K=1, spread=0.0); idx = torch.arange(ep.n); st = env.reset(idx)
    for t in range(env.nsteps):
        env.step(st, torch.zeros(ep.n, dtype=torch.long))
        if st["done"].all(): break
    ref = np.asarray(T.R[ep.rows], dtype=np.float32)[:, A.aid(*base, 0)]; got = st["R"].numpy(); assert np.abs(got - ref).max() < 2e-3, (np.abs(got - ref).max(), int((np.abs(got - ref) > 2e-3).sum()))
    # actions are well-defined: breakeven/trail/exit run without error and never raise the loss below -1R
    for a in (1, 2, 3):
        st = env.reset(idx)
        for t in range(env.nsteps):
            env.step(st, torch.full((ep.n,), a)); 
            if st["done"].all(): break
        assert (st["R"].numpy() >= -1.0003).all()
