import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as DD, windows as W, pretrain as PT

def test_masked_pretraining_runs_and_loss_drops():
    m1 = DD.load_m1("EURUSD", start="2012-03-01", end="2012-04-15"); store = W.BarStore(bars_m1={"EURUSD": m1}, pairs=("EURUSD",)).to(torch.device("cpu")); torch.manual_seed(0); net = PT.MaskedPre("tiny"); opt = torch.optim.AdamW(net.parameters(), lr=1e-3); rng = np.random.default_rng(0); l0 = None
    for s in range(25):
        w1, w2 = PT.random_windows(store, rng, 32, 2030, torch.device("cpu")); w1 = w1.clamp(-6, 6); w2 = w2.clamp(-6, 6); loss = net(w1, w2); opt.zero_grad(); loss.backward(); opt.step(); l0 = l0 or loss.item()
    assert np.isfinite(loss.item()) and loss.item() < l0
