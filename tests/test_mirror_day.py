"""Day-factor mirroring: sell rows become the long-perspective view; buy rows untouched; applying it twice restores the original; semantic checks per rule."""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import data as DD, build_table as BT

def setup():
    d, cols = BT.days_table(); x = d[d.pair == "EURUSD"][cols].dropna().values[:400].astype(np.float32); return cols, x

def test_buy_untouched_and_involution():
    cols, x = setup(); side = np.where(np.arange(len(x)) % 2 == 0, 1, -1); m = BT.mirror_day(x, cols, side)
    assert np.array_equal(m[side > 0], x[side > 0]) and not np.array_equal(m[side < 0], x[side < 0]); assert np.allclose(BT.mirror_day(m, cols, side), x, atol=1e-6)

def test_rules_and_unmirrored_columns():
    cols, x = setup(); side = -np.ones(len(x), int); m = BT.mirror_day(x, cols, side); ix = {c: i for i, c in enumerate(cols)}
    assert np.allclose(m[:, ix["ret"]], -x[:, ix["ret"]]) and np.allclose(m[:, ix["cloc"]], 1 - x[:, ix["cloc"]]) and np.allclose(m[:, ix["upw"]], x[:, ix["dnw"]])
    assert np.allclose(m[:, ix["d_prev_hi"]], -x[:, ix["d_prev_lo"]]) and np.allclose(m[:, ix["zBUY_size"]], x[:, ix["zSELL_size"]]) and np.allclose(m[:, ix["hi_hr"]], x[:, ix["lo_hr"]])
    for c in ("rng", "rv5", "rv20", "atr_pct", "dow_s", "mon_c", "sA_pbfrac", "sA_inpb", "beid_missing", "asia_rng", "volvol"): assert np.array_equal(m[:, ix[c]], x[:, ix[c]]), c
    touched = set(BT.DAY_FLIP) | set(BT.DAY_COMPLEMENT) | {c for p in BT.DAY_SWAP + BT.DAY_SWAPNEG for c in p}; assert set(cols) - touched == {c for c in cols if np.array_equal(m[:, ix[c]], x[:, ix[c]])} - touched
