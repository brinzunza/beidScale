"""Optional self-supervised pretraining: masked-patch reconstruction of M1/M15 windows at random times (years < end_year only) with the BEIDNet encoder; saves encoder weights for `train.py --init`.
  torchrun --nproc_per_node=6 pretrain.py --out RUNS/pre --size small --steps 20000 --end_year 2017"""
import argparse, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
import data as DD, windows as W, model as M

class MaskedPre(nn.Module):
    def __init__(self, size):
        super().__init__(); self.net = M.BEIDNet(size, n_ctx=1); d = self.net.d; self.mask_tok = nn.Parameter(torch.zeros(1, 1, d)); self.rec1 = nn.Linear(d, 8 * 4); self.rec2 = nn.Linear(d, 4 * 4)
    def forward(self, w1, w2, ratio=0.3):
        n = self.net; B = w1.shape[0]; t1 = n.stem1(w1.transpose(1, 2)).transpose(1, 2); t2 = n.stem2(w2.transpose(1, 2)).transpose(1, 2); m1 = torch.rand(B, t1.shape[1], device=w1.device) < ratio; m2 = torch.rand(B, t2.shape[1], device=w1.device) < ratio
        t1 = torch.where(m1[..., None], self.mask_tok.expand_as(t1), t1) + n.pos1 + n.tok_type[0]; t2 = torch.where(m2[..., None], self.mask_tok.expand_as(t2), t2) + n.pos2 + n.tok_type[1]
        x = torch.cat([n.cls.expand(B, -1, -1) + n.tok_type[3], torch.zeros(B, 1, n.d, device=w1.device) + n.tok_type[2], t1, t2], 1); x = n.norm(n.enc(x)); h1 = x[:, 2:2 + t1.shape[1]]; h2 = x[:, 2 + t1.shape[1]:]
        tg1 = w1[..., :4].reshape(B, t1.shape[1], 8 * 4); tg2 = w2[..., :4].reshape(B, t2.shape[1], 4 * 4); l1 = ((self.rec1(h1) - tg1) ** 2).mean(-1); l2 = ((self.rec2(h2) - tg2) ** 2).mean(-1)
        return ((l1 * m1).sum() + (l2 * m2).sum()) / (m1.sum() + m2.sum()).clamp(min=1)

def random_windows(store, rng, B, end_year, dev):
    """random M1/M15 windows (long perspective, normalized by a crude 0.1% price scale); only the first part of each pair's history (years < end_year, by proportion) is used. The M15 window is
    aligned proportionally (approximate), which is fine for self-supervised pretraining that never touches labels."""
    pairs = list(store.off1); starts1 = [store.off1[p] for p in pairs] + [len(store.m1)]; starts15 = [store.off15[p] for p in pairs] + [len(store.m15)]; frac = min(1.0, max(0.1, (end_year - 2005) / 21.7)); idx1, idx15 = [], []
    for _ in range(B):
        k = int(rng.integers(len(pairs))); lo = starts1[k] + W.L1 + 10; hi = starts1[k + 1] - 10; hi = lo + int((hi - lo) * frac); i = int(rng.integers(lo, hi))
        r = (i - starts1[k]) / (starts1[k + 1] - starts1[k]); j = starts15[k] + int(r * (starts15[k + 1] - starts15[k])); j = min(max(j, starts15[k] + W.L2 + 5), starts15[k + 1] - 2); idx1.append(i); idx15.append(j)
    i1 = torch.tensor(idx1, device=dev); i15 = torch.tensor(idx15, device=dev); mid = store.tm1[i1 - 1, 3]
    return W.gather_windows(store, i1, i15, torch.ones(B, dtype=torch.long, device=dev), mid, mid * 1e-3)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--size", default="small"); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--end_year", type=int, default=2017)
    ap.add_argument("--bar_pairs", nargs="*"); ap.add_argument("--bars_start"); ap.add_argument("--bars_end"); a = ap.parse_args(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rng = np.random.default_rng(0); torch.manual_seed(0)
    store = W.BarStore(pairs=tuple(a.bar_pairs or DD.PAIRS), start=a.bars_start, end=a.bars_end).to(dev); m = MaskedPre(a.size).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05); t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for s in range(a.steps):
        w1, w2 = random_windows(store, rng, a.batch, a.end_year, dev); w1 = w1.clamp(-6, 6); w2 = w2.clamp(-6, 6); loss = m(w1, w2); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 100 == 0: print(f"step {s} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if s % 2000 == 0 or s == a.steps - 1: torch.save(m.net.state_dict(), out / "encoder.pt")
if __name__ == "__main__":
    main()
