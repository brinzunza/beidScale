"""Zone engine: defaults reproduce the goal335 kernel exactly; wick/close params change zones; event is causal (uses only bars <= confirmation bar)."""
import sys, numpy as np, pytest
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import data as DD, zones as Z
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

def sample():
    m = DD.load_m1("EURUSD", start="2012-03-01", end="2012-04-15"); return m

def test_parity_with_goal335():
    import ref_goal335_zones as RB
    m = sample(); day = DD.day_ids(m["t"], "ny")
    a = Z.zones_kernel(m["o"], m["h"], m["l"], m["c"], day, True, 0, 1, 1, 1); b = RB.zones_kernel(m["o"], m["h"], m["l"], m["c"], day, True)
    for x, y in zip(a, b): assert np.array_equal(x, y)
    assert len(a[0]) > 100

def test_params_change_zones():
    m = sample(); day = DD.day_ids(m["t"], "ny"); base = Z.zones_kernel(m["o"], m["h"], m["l"], m["c"], day, True, 0, 1, 1, 1)
    for args in ((1, 1, 1, 1), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 3)):
        z = Z.zones_kernel(m["o"], m["h"], m["l"], m["c"], day, True, *args); assert len(z[0]) != len(base[0]) or not np.array_equal(z[2], base[2])

def test_causality_truncation():
    m = sample(); cut = len(m["t"]) // 2; full = Z.build_zone_table(m, Z.ZoneParams(tf=5)); tr = {k: v[:cut] for k, v in m.items()}; part = Z.build_zone_table(tr, Z.ZoneParams(tf=5))
    f = full[full.ev < cut - 600]; p = part[part.ev < cut - 600]
    assert len(f) == len(p) and np.allclose(f.top.values, p.top.values) and np.array_equal(f.ev.values, p.ev.values)

def test_event_after_bar_close_and_table_sane():
    m = sample()
    for tf in (1, 5, 15, 60):
        t = Z.build_zone_table(m, Z.ZoneParams(tf=tf)); assert len(t) > 20 and (t.top > t.bot).all() and (t.ord_end > t.ev).all() and (t.ev < len(m["t"])).all()
        assert set(t.side.unique()) <= {-1, 1}
