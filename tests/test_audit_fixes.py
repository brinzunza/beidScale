"""Audit fixes: fill penetration, overlap blocking, epoch selection / refit / era options."""
import sys, json, subprocess, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as DD, zones as Z, rewards as RW, evaluate as EV, build_table as BT, train as TR

def test_fill_penetration_is_stricter():
    m = DD.load_m1("EURUSD", start="2012-03-01", end="2012-04-20"); zt = Z.build_zone_table(m, Z.ZoneParams(tf=15)).iloc[:150]; pip = DD.PIP["EURUSD"]
    a = RW.build_rewards(m, zt, pip, fill_pen_pips=0.0); b = RW.build_rewards(m, zt, pip, fill_pen_pips=0.5)
    assert (b["F"] <= a["F"]).all() and b["F"][:, :5].sum() < a["F"][:, :5].sum() and np.array_equal(b["F"][:, 5], a["F"][:, 5])           # market entries unchanged; limit fills only drop
    both = (a["F"][:, :5] > 0) & (b["F"][:, :5] > 0); assert (b["FIL"][:, :5][both] >= a["FIL"][:, :5][both]).all()                                   # and happen no earlier

def test_block_hours_blocks_overlap(monkeypatch):
    t0 = np.datetime64("2015-03-02T10:00", "ns").astype(np.int64); h = 3600 * 10 ** 9; z = pd.DataFrame(dict(t_ev=[t0, t0 + 5 * h, t0 + 30 * h, t0 + 60 * h, t0 + 5 * h], pair=[0, 0, 0, 0, 1], side=[1, 1, 1, 1, 1])); mask = np.ones(5, bool)
    monkeypatch.delenv("BEID_BLOCK_HOURS", raising=False); day = EV.dedupe(z, mask); assert day.tolist() == [True, False, True, True, True]          # same FX day -> one; next days free
    monkeypatch.setenv("BEID_BLOCK_HOURS", "48"); blk = EV.dedupe(z, mask); assert blk.tolist() == [True, False, False, True, True]                   # +30h still inside 48h of the first; +60h free

def test_selection_refit_and_era_options(tmp_path):
    out = tmp_path / "ds"; BT.shard("EURUSD", 15, "ny", str(out), start="2010-01-01", end="2013-01-01"); BT.merge(str(out)); tdir = out / "tableA"
    base = ["--table", str(tdir), "--size", "tiny", "--test_year", "2012", "--epochs", "2", "--batch", "64", "--max_rows", "1200", "--bar_pairs", "EURUSD", "--bars_start", "2010-01-01", "--bars_end", "2013-01-01", "--log_every", "50"]
    for sel in ("ev", "q"): TR.run(TR.parser().parse_args(base + ["--out", str(tmp_path / sel), "--select", sel, "--no_atr_pips"]))
    log = pd.read_csv(tmp_path / "ev" / "log_2012.csv"); assert len(log) == 2 and {"ev_val", "q_val"} <= set(log.columns)
    TR.run(TR.parser().parse_args(base + ["--out", str(tmp_path / "rf"), "--include_val"])); assert (tmp_path / "rf" / "pred_2012.npz").exists()
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "src" / "refit.py"), "--runs", str(tmp_path / "ev"), "--years", "2012-2012", "--table", "T", "--out", "O", "--gpus", "2"], capture_output=True, text=True); assert "--include_val" in r.stdout and "--test_year 2012" in r.stdout, r.stderr
