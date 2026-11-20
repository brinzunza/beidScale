"""GPU window gather: last 256 M1 bars + last 96 M15 bars before each zone event, mirrored to the long perspective, normalized by ATR15 around the zone mid. Causal: all indices < event."""
import numpy as np, torch
import data as DD
L1, L2 = 256, 96

class BarStore:
    """concatenated per-pair bar arrays (float32) + time-of-day channels; offsets keyed by pair order DD.PAIRS."""
    def __init__(self, bars_m1=None, pairs=DD.PAIRS, start=None, end=None):
        self.pairs = pairs; self.off1 = {}; self.off15 = {}; a1, a15, t1, t15 = [], [], [], []; n1 = n15 = 0
        for p in pairs:
            m1 = DD.load_m1(p, start, end) if bars_m1 is None else bars_m1[p]; m15 = DD.resample(m1, 15)
            self.off1[p], self.off15[p] = n1, n15
            for src, A, T in ((m1, a1, t1), (m15, a15, t15)):
                A.append(np.stack([src["o"], src["h"], src["l"], src["c"]], 1).astype(np.float32)); hr, _ = DD.ny_parts(src["t"]); ang = 2 * np.pi * (hr + (src["t"] // (60 * 10 ** 9)) % 60 / 60.0) / 24.0; T.append(np.stack([np.sin(ang), np.cos(ang)], 1).astype(np.float32))
            n1 += len(m1["t"]); n15 += len(m15["t"])
        self.m1 = np.concatenate(a1); self.m15 = np.concatenate(a15); self.t1 = np.concatenate(t1); self.t15 = np.concatenate(t15); self.dev = None
    def to(self, device):
        self.dev = device; self.tm1 = torch.from_numpy(self.m1).to(device); self.tm15 = torch.from_numpy(self.m15).to(device); self.tt1 = torch.from_numpy(self.t1).to(device); self.tt15 = torch.from_numpy(self.t15).to(device); return self

def _gather(arr, tod, idx, L, side, mid, atr):
    B = idx.shape[0]; ar = torch.arange(-L, 0, device=idx.device); ii = (idx[:, None] + ar[None, :]).clamp(min=0); x = arr[ii]                  # (B, L, 4) o h l c
    sg = side.to(x.dtype)[:, None, None]; o, h, l, c = x[..., 0], x[..., 1], x[..., 2], x[..., 3]; pos = (sg[..., 0] > 0)
    oo = torch.where(pos, o, -o); hh = torch.where(pos, h, -l); ll = torch.where(pos, l, -h); cc = torch.where(pos, c, -c)
    z = torch.stack([oo, hh, ll, cc], -1); midm = torch.where(side > 0, mid, -mid); z = (z - midm[:, None, None]) / atr[:, None, None]; return torch.cat([z.clamp(-12, 12), tod[ii]], -1)   # (B, L, 6)

def gather_windows(store, m1_idx, m15_idx, side, mid, atr):
    """m1_idx: global index of the FIRST M1 bar after the event (window = the L1 bars before it). m15_idx: global index of the last CLOSED M15 bar (window ends at it, inclusive)."""
    w1 = _gather(store.tm1, store.tt1, m1_idx, L1, side, mid, atr); w2 = _gather(store.tm15, store.tt15, m15_idx + 1, L2, side, mid, atr); return w1, w2
