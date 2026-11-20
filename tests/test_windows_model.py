import sys, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import data as DD, zones as Z, windows as W, model as M, actions as A

def test_window_causality_and_mirror():
    m1 = DD.load_m1("EURUSD", start="2012-03-01", end="2012-03-20"); store = W.BarStore(bars_m1={"EURUSD": m1}, pairs=("EURUSD",)).to(torch.device("cpu")); zt = Z.build_zone_table(m1, Z.ZoneParams(tf=15)); m15 = DD.resample(m1, 15)
    z = zt[zt.ev > 400].iloc[:6]; j15 = np.searchsorted(m15["e"], z.ev.values, side="right") - 1; mid = ((z.top + z.bot) / 2).values; atr = z.atr15.values
    w1, w2 = W.gather_windows(store, torch.from_numpy(z.ev.values), torch.from_numpy(j15), torch.from_numpy(z.side.values.astype(np.int64)), torch.from_numpy(mid).float(), torch.from_numpy(atr).float())
    assert w1.shape == (6, W.L1, 6) and w2.shape == (6, W.L2, 6)
    for q in range(6):
        ev = int(z.ev.iloc[q]); sg = int(z.side.iloc[q]); c_last = m1["c"][ev - 1]; exp = (sg * c_last - (mid[q] if sg > 0 else -mid[q])) / atr[q]
        assert abs(float(w1[q, -1, 3]) - exp) < 1e-3, (float(w1[q, -1, 3]), exp)         # last window bar = bar ev-1 (causal), mirrored for sells
        # the window never contains bar ev or later: changing m1[ev:] must not change the window
    m1b = {k: v.copy() for k, v in m1.items()}; ev0 = int(z.ev.iloc[0]); m1b["c"][ev0:] += 1.0; m1b["h"][ev0:] += 1.0; m1b["l"][ev0:] += 1.0; m1b["o"][ev0:] += 1.0
    store2 = W.BarStore(bars_m1={"EURUSD": m1b}, pairs=("EURUSD",)).to(torch.device("cpu")); w1b, w2b = W.gather_windows(store2, torch.from_numpy(z.ev.values[:1]), torch.from_numpy(j15[:1]), torch.from_numpy(z.side.values[:1].astype(np.int64)), torch.from_numpy(mid[:1]).float(), torch.from_numpy(atr[:1]).float())
    assert torch.allclose(w1[:1, :, :4], w1b[:, :, :4], atol=1e-5)
    assert torch.allclose(w2[:1, :-1, :4], w2b[:, :-1, :4], atol=1e-5)

def test_model_shapes_and_planted_learning():
    torch.manual_seed(0); net = M.BEIDNet("tiny", n_ctx=10); B = 256; w1 = torch.randn(B, 256, 6); w2 = torch.randn(B, 96, 6); ctx = torch.randn(B, 10); cat = torch.zeros(B, 4, dtype=torch.long)
    q, pol = net(w1, w2, ctx, cat); assert q.shape == (B, A.NACT) and pol.shape == (B, A.NACT + 1)
    V = torch.ones(B, A.NACT, dtype=torch.uint8); netR = torch.zeros(B, A.NACT) - 0.05; good = (ctx[:, 0] > 0).long(); netR[torch.arange(B), 7 + 20 * good] = 0.6     # planted: action 7 if ctx0<=0 else 27 earns +0.6
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(150):
        q, pol = net(w1, w2, ctx, cat); lp, ev, ent = M.policy_loss(pol, V, netR, 0.001); lq = M.q_loss(q, netR, V); loss = lp + lq; opt.zero_grad(); loss.backward(); opt.step()
    assert float(ev) > 0.4, float(ev)
    am = M.masked_policy_logits(pol, V).argmax(1); assert (am == (7 + 20 * good)).float().mean() > 0.9

def test_invalid_actions_never_chosen_and_skip_available():
    B = 8; pol = torch.randn(B, A.NACT + 1); V = torch.zeros(B, A.NACT, dtype=torch.uint8); V[:, 5] = 1; lg = M.masked_policy_logits(pol, V); p = torch.softmax(lg, 1); assert (p[:, [i for i in range(A.NACT) if i != 5]] < 1e-6).all() and (p[:, 5] + p[:, A.NACT]).min() > 0.999
