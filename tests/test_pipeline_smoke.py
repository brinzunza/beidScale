"""Tiny end-to-end smoke on CPU: build a toy Table A (1 pair, 15m zones, 2010-2012), train 1 epoch with the tiny model, predict, evaluate."""
import sys, numpy as np, pytest, tempfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_table as BT, train as TR, evaluate as EV

def test_end_to_end_smoke(tmp_path):
    out = tmp_path / "ds"; BT.shard("EURUSD", 15, "ny", str(out), start="2010-01-01", end="2013-01-01"); BT.merge(str(out)); tdir = out / "tableA"
    args = TR.parser().parse_args(["--table", str(tdir), "--out", str(tmp_path / "run"), "--size", "tiny", "--test_year", "2012", "--epochs", "1", "--batch", "64", "--max_rows", "1200", "--bar_pairs", "EURUSD", "--bars_start", "2010-01-01", "--bars_end", "2013-01-01", "--log_every", "5"])
    TR.run(args); pred = np.load(tmp_path / "run" / "pred_2012.npz"); assert pred["q"].shape[1] == 420 and pred["pol"].shape[1] == 421 and len(pred["rows"]) > 50 and np.isfinite(pred["q"].astype(np.float32)).all()
    res = EV.evaluate_run(str(tdir), [str(tmp_path / "run" / "pred_2012.npz")], spreads=(0.0, 1.3)); assert {"greedy", "greedyQ"} <= set(res.rule) and len(res) >= 4   # top-q rules need a previous year, so they appear from the second prediction file on

def test_new_training_options_and_ensemble(tmp_path, monkeypatch):
    import ensemble as EN, rewardview as RV
    out = tmp_path / "ds"; BT.shard("EURUSD", 15, "ny", str(out), start="2010-01-01", end="2013-01-01"); BT.merge(str(out)); tdir = out / "tableA"; names = json.load(open(out / "shards" / "daycols.json"))["feat_names"]; X = np.load(tdir / "X.npy"); assert X.shape[1] == 46 + len(names) and names[-1] == "day_missing"
    assert X[:, -1].min() == 0 and X[:, -1].max() == 0                                   # 2010-12 events all have day factors
    monkeypatch.setenv("BEID_NO_FADE", "1"); base = ["--table", str(tdir), "--size", "tiny", "--test_year", "2012", "--epochs", "1", "--batch", "64", "--max_rows", "1200", "--bar_pairs", "EURUSD", "--bars_start", "2010-01-01", "--bars_end", "2013-01-01", "--log_every", "50"]
    for k in range(2): TR.run(TR.parser().parse_args(base + ["--out", str(tmp_path / f"r{k}"), "--seed", str(k), "--ema", "0.99", "--tf_pow", "0.5", "--min_year", "2010"]))
    p = np.load(tmp_path / "r0" / "pred_2012.npz"); assert (p["pol"].astype(np.float32)[:, 1:420:2] <= -1e3).all()          # fade actions masked out of the policy
    e = EN.average([tmp_path / "r0" / "pred_2012.npz", tmp_path / "r1" / "pred_2012.npz"]); assert e["q"].shape == p["q"].shape and np.array_equal(e["rows"], p["rows"]) and (e["pol"].astype(np.float32)[:, 1:420:2] <= -1e3).all()
