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
