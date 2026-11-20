import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import search as S, actions as A

def test_realized_runs_on_toy_slice():
    r = S.realized(3, A.aid(2, 0, 1, 0), 2012, 2012, pairs=("EURUSD",), start="2012-01-01", end="2012-06-01"); assert r is not None and {q["spread"] for q in r} == {0.0, 0.5, 1.0, 1.3, 1.6}
    assert r[0]["n"] > 5 and np.isfinite(r[0]["mean_net_R"]) and r[0]["mean_net_R"] >= r[-1]["mean_net_R"]          # more spread, lower net
