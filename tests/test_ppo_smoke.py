"""PPO trade-management smoke: runs end to end on a toy table (CPU), checks the output file and that evaluation policies produce finite numbers."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_table as BT, ppo_manage as PM

def test_ppo_manage_smoke(tmp_path, monkeypatch):
    out = tmp_path / "ds"; BT.shard("EURUSD", 15, "ny", str(out), start="2010-01-01", end="2012-07-01"); BT.merge(str(out))
    monkeypatch.setattr(sys, "argv", ["ppo_manage.py", "--table", str(out / "tableA"), "--out", str(tmp_path / "run"), "--test_year", "2012", "--iters", "3", "--episodes", "256", "--K", "30", "--bar_pairs", "EURUSD", "--bars_start", "2010-01-01", "--bars_end", "2012-07-01"])
    PM.main(); r = json.load(open(tmp_path / "run" / "manage_2012.json"))["res"]; assert set(r) == {"hold", "random", "learned"} and all(np.isfinite(v["mean_net_R"]) for v in r.values()) and r["hold"]["n"] > 30
