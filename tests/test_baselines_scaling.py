import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_table as BT, baselines as BL, controls as CT, scaling as SC, evaluate as EV

def test_baselines_and_planners(tmp_path, capsys):
    out = tmp_path / "ds"; BT.shard("EURUSD", 15, "ny", str(out), start="2010-01-01", end="2012-07-01"); BT.merge(str(out)); d = out / "tableA"; z = pd.read_parquet(d / "zones.parquet"); tr, te = BL.split_rows(z, 2012)
    assert len(tr) > 100 and len(te) > 50 and z.t_ev.values[tr].max() < np.datetime64("2012-01-01", "ns").astype(np.int64)
    b = EV.baselines(d, tr, te); assert {"random_valid"} <= set(b.rule) and any(r.startswith("best_fixed") for r in b.rule)
    class A_: table = str(d); runs = str(tmp_path / "r"); years = "2012-2012"; size = "tiny"; shifts = 2; gpus = 2; epochs = 1; plant = 0.1
    CT.plan(A_); text = capsys.readouterr().out; assert "--null_shift" in text and "--plant 0.1" in text and text.count("torchrun") == 1 + 2 + 1
    class S_: table = str(d); runs = str(tmp_path / "r"); years = "2012"; gpus = 1; epochs = 1
    SC.plan(S_); assert capsys.readouterr().out.count("torchrun") == 12
