import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_table as BT, rewardview as RV, actions as A, evaluate as EV, pandas as pd

def test_null_shift_and_plant(tmp_path):
    out = tmp_path / "ds"; BT.shard("EURUSD", 15, "ny", str(out), start="2010-01-01", end="2012-07-01"); BT.merge(str(out)); d = out / "tableA"; z = pd.read_parquet(d / "zones.parquet")
    base = RV.RewardSource(d, z); shifted = RV.RewardSource(d, z, null_shift=500); rows = np.arange(100, 160); R0, V0, F0, K0 = base.get(rows); R1, V1, F1, K1 = shifted.get(rows); R2, _, _, _ = base.get(rows + 500)
    assert np.array_equal(R1, R2) and not np.array_equal(R0, R1)                                  # shifted rewards are those of the event 500 rows later (same pair)
    plant = RV.RewardSource(d, z, plant_delta=0.25); reg = RV.region_mask(z); allrows = np.flatnonzero(reg)[:80]; Rp, Vp, Fp, _ = plant.get(allrows); Rb, _, _, _ = base.get(allrows)
    m = (Fp[:, 2] > 0) & (Vp[:, RV.PLANT_ACTION] > 0); assert abs(float((Rp - Rb)[m, RV.PLANT_ACTION].mean()) - 0.25) < 1e-3 and (np.delete(Rp - Rb, RV.PLANT_ACTION, 1) == 0).all()
    assert reg.sum() > 20
