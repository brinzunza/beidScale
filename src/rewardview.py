"""One place that reads reward arrays for a set of event rows, optionally applying (a) a circular-shift NULL (rewards of other events within the same pair, large offset) or (b) a PLANTED edge
(+delta net R on one hidden action in a hidden state region). Used identically by training and evaluation so controls are consistent."""
import numpy as np, pandas as pd
from pathlib import Path
import actions as A
PLANT_ACTION = A.aid(2, 0, 1, 0)       # 50% limit, far-edge stop, 1R target, with the zone

def region_mask(z):
    """hidden state region for the planted edge: 15-minute zones created 07-11 NY with height above 1 ATR15."""
    return ((z.ztf.values == 15) & (z.ny_hour.values >= 7) & (z.ny_hour.values < 11) & (((z.top - z.bot) / z.atr15).values > 1.0))

class RewardSource:
    def __init__(self, table_dir, zones, null_shift=0, plant_delta=0.0):
        d = Path(table_dir); self.R = np.load(d / "R.npy", mmap_mode="r"); self.V = np.load(d / "V.npy", mmap_mode="r"); self.F = np.load(d / "F.npy", mmap_mode="r"); self.RISK = np.load(d / "RISK.npy", mmap_mode="r"); self.n = len(zones)
        self.rew = np.arange(self.n)
        if null_shift:
            pairs = zones.pair.values
            for p in np.unique(pairs):
                ix = np.flatnonzero(pairs == p); self.rew[ix] = np.roll(ix, -int(null_shift))      # table is time-sorted, so ix is in time order
        self.plant = plant_delta; self.region = region_mask(zones) if plant_delta else None
    def get(self, rows):
        rows = np.asarray(rows); rr = self.rew[rows]; o = np.argsort(rr); inv = np.argsort(o); s = rr[o]
        R = np.asarray(self.R[s], dtype=np.float32)[inv]; V = np.asarray(self.V[s])[inv]; F = np.asarray(self.F[s])[inv]; RISK = np.asarray(self.RISK[s])[inv]
        if self.plant:
            m = self.region[rows] & (F[:, 2] > 0) & (V[:, PLANT_ACTION] > 0)           # planted edge sits on the STATE's region (rows), delivered through the reward row
            R[m, PLANT_ACTION] += np.float32(self.plant)
        return R, V, F, RISK
