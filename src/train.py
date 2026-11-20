"""Walk-forward training of BEIDNet on Table A (memory-mapped). Single GPU/CPU or torchrun DDP. Usage:
  torchrun --nproc_per_node=6 train.py --table DIR/tableA --out RUNS/x --size small --test_year 2018 --epochs 6
Train rows: events with year < test_year - 1 (and resolution horizon before the boundary); validation = the year test_year-1 (epoch selection by validation loss); prediction on test_year rows saved."""
import os, sys, time, json, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
import data as DD, actions as A, windows as W, model as M, rewardview as RV
ENT_OF = torch.tensor(np.repeat(np.arange(A.ENT), A.NSTOP * A.NTGT * A.NDIR)); SLOT_OF = torch.tensor(A.risk_slot(np.arange(A.NACT)))

class Table:
    def __init__(self, d, store_args=None, null_shift=0, plant=0.0):
        d = Path(d); self.z = pd.read_parquet(d / "zones.parquet"); self.R = np.load(d / "R.npy", mmap_mode="r"); self.V = np.load(d / "V.npy", mmap_mode="r"); self.F = np.load(d / "F.npy", mmap_mode="r"); self.RISK = np.load(d / "RISK.npy", mmap_mode="r"); self.X = np.load(d / "X.npy", mmap_mode="r"); self.src = RV.RewardSource(d, self.z, null_shift, plant)
        self.n = len(self.z); self.year = self.z.year.values; self.t_ev = self.z.t_ev.values.astype(np.int64); self.pair = self.z.pair.values; self.side = self.z.side.values; self.mid = ((self.z.top + self.z.bot) / 2).values; self.atr = self.z.atr15.values
        self.off1 = None; self.ctx_cat = np.stack([self.pair, pd.Series(self.z.ztf).map({1: 0, 5: 1, 15: 2, 60: 3}).values, self.z.dow.values, self.z.ztype.values], 1).astype(np.int64)
        H = ((self.z.top - self.z.bot) / self.z.atr15).values; hr = self.z.ny_hour.values; self.geo = np.stack([np.log1p(H), np.log1p(self.z.atr15.values / np.array([DD.PIP[DD.PAIRS[p]] for p in self.pair]) / 10.0), np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)], 1).astype(np.float32)
    def set_store(self, store):
        self.store = store; self.m1_idx = np.array([store.off1[DD.PAIRS[p]] for p in self.pair]) + self.z.ev.values; self.m15_idx = np.array([store.off15[DD.PAIRS[p]] for p in self.pair]) + self.z.j15.values
    def batch(self, idx, device, mu, sd, spread):
        idx = np.sort(idx); R_, V_, F_, RISK_ = self.src.get(idx); R = torch.from_numpy(R_); V = torch.from_numpy(V_); Fm = torch.from_numpy(F_); RISK = torch.from_numpy(RISK_).reshape(len(idx), -1)
        with torch.no_grad():
            risk = RISK[:, SLOT_OF]; cost = torch.where(Fm[:, ENT_OF] > 0, spread / risk, torch.zeros_like(risk)); net = torch.where(V > 0, R - torch.nan_to_num(cost), torch.zeros_like(R))
        x = np.concatenate([self.geo[idx], np.nan_to_num((np.asarray(self.X[idx]) - mu) / sd).clip(-6, 6)], 1).astype(np.float32)
        w1, w2 = W.gather_windows(self.store, torch.from_numpy(self.m1_idx[idx]).to(device), torch.from_numpy(self.m15_idx[idx]).to(device), torch.from_numpy(self.side[idx]).to(device), torch.from_numpy(self.mid[idx]).to(device).float(), torch.from_numpy(self.atr[idx]).to(device).float())
        return idx, w1, w2, torch.from_numpy(x).to(device), torch.from_numpy(self.ctx_cat[idx]).to(device), net.to(device), V.to(device)

def setup_dist():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo"); r = int(os.environ["RANK"]); lr = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available(): torch.cuda.set_device(lr)
        return r, int(os.environ["WORLD_SIZE"]), lr
    return 0, 1, 0

def run(a):
    rank, world, lrank = setup_dist(); dev = torch.device(f"cuda:{lrank}" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() and a.mps else "cpu")); torch.manual_seed(a.seed + rank)
    T = Table(a.table, null_shift=a.null_shift, plant=a.plant); store = W.BarStore(pairs=tuple(a.bar_pairs or DD.PAIRS), start=a.bars_start, end=a.bars_end).to(dev); T.set_store(store); Y = a.test_year; boundary = np.datetime64(f"{Y-1}-01-01", "ns").astype(np.int64); horizon_ns = 72 * 3600 * 10 ** 9
    var_ok = np.ones(T.n, bool)
    if a.tf: var_ok &= T.z.ztf.isin(a.tf).values
    if a.reset: var_ok &= (T.z.reset == a.reset).values
    if a.pairs: var_ok &= T.z.pairname.isin(a.pairs).values
    if a.min_year: var_ok &= T.year >= a.min_year                      # drop early, lower-quality feed years (and events without day factors)
    tr = np.flatnonzero(var_ok & (T.year < Y - 1) & (T.t_ev + horizon_ns < boundary)); va = np.flatnonzero(var_ok & (T.year == Y - 1) & (T.t_ev + horizon_ns < np.datetime64(f"{Y}-01-01", "ns").astype(np.int64))); te = np.flatnonzero(var_ok & (T.year == Y))
    if a.include_val: tr = np.flatnonzero(var_ok & (T.year < Y) & (T.t_ev + horizon_ns < np.datetime64(f"{Y}-01-01", "ns").astype(np.int64)))     # refit: train through the validation year with a fixed epoch count (no selection)
    if a.no_atr_pips: T.geo[:, 1] = 0.0                                                   # remove the pips-scale feature that reveals the data era
    if a.frac < 1: tr = np.sort(np.random.default_rng(a.seed).choice(tr, int(len(tr) * a.frac), replace=False))
    if a.max_rows: tr = tr[-a.max_rows:]; va = va[:a.max_rows // 4]; te = te[:a.max_rows // 4]
    Xtr = np.asarray(T.X[np.sort(tr)[:: max(1, len(tr) // 50000)]]); mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-6
    net = M.BEIDNet(a.size, n_ctx=T.geo.shape[1] + T.X.shape[1], checkpoint=a.grad_ckpt).to(dev); n_par = net.n_params()
    if rank == 0: print(f"rows train {len(tr)} val {len(va)} test {len(te)} | params {n_par/1e6:.2f}M | device {dev} | world {world}", flush=True)
    if a.init: print('init from', a.init, net.load_state_dict({k: v for k, v in torch.load(a.init, map_location='cpu').items() if k in net.state_dict() and net.state_dict()[k].shape == v.shape}, strict=False))
    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[lrank] if dev.type == "cuda" else None) if world > 1 else net
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95)); steps_per_epoch = max(1, len(tr) // (a.batch * world)); total = steps_per_epoch * a.epochs; warm = max(10, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(total, 1))))); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); log = []; best, best_state = 1e18, None; step = 0
    use_amp = dev.type == "cuda"; params = [q for q in net.parameters()]
    ema_p = [q.detach().clone() for q in params] if a.ema > 0 else None
    def ema_swap():
        for q, e in zip(params, ema_p): t = q.detach().clone(); q.data.copy_(e); e.copy_(t)           # exchange live and EMA weights (call twice to restore)
    tfw = None
    if a.tf_pow > 0: ztf = T.z.ztf.values; cnt = pd.Series(ztf[tr]).value_counts(); w_tf = (1.0 / cnt) ** a.tf_pow; tfw = pd.Series(ztf[tr]).map(w_tf).values; tfw = tfw / tfw.sum()
    def fwd(batch):
        idx, w1, w2, x, cat, netR, V = batch
        with torch.autocast(device_type=dev.type if dev.type != "mps" else "cpu", dtype=torch.bfloat16, enabled=use_amp): q, pol = model(w1, w2, x, cat)
        lq = M.q_loss(q, netR, V); lp, ev, ent = M.policy_loss(pol, V, netR, a.ent, a.objective); return lq, lp, ev, ent, q, pol
    for ep in range(a.epochs):
        model.train(); rng = np.random.default_rng(a.seed * 1000 + ep); perm = (rng.permutation(tr) if tfw is None else rng.choice(tr, len(tr), replace=True, p=tfw))[rank::world]; t0 = time.time(); agg = np.zeros(4)
        for b in range(steps_per_epoch):
            idx = perm[b * a.batch:(b + 1) * a.batch]
            if len(idx) < 8: break
            lq, lp, ev, ent, _, _ = fwd(T.batch(idx, dev, mu, sd, a.train_spread)); loss = a.wq * lq + a.wp * lp; opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if ema_p is not None: torch._foreach_mul_(ema_p, a.ema); torch._foreach_add_(ema_p, [q.detach() for q in params], alpha=1 - a.ema)
            agg += np.array([lq.item(), lp.item(), ev.item(), ent.item()])
            if rank == 0 and step % a.log_every == 0: print(f"ep {ep} step {step}/{total} q {agg[0]/(b+1):.4f} pol {agg[1]/(b+1):.4f} E[net] {agg[2]/(b+1):+.4f} ent {agg[3]/(b+1):.3f} {time.time()-t0:.0f}s", flush=True)
        if ema_p is not None: ema_swap()                                  # validate / select epochs with the EMA weights
        model.eval(); vl = []
        with torch.no_grad():
            for i in range(0, len(va), a.batch):
                lq, lp, ev, ent, _, _ = fwd(T.batch(va[i:i + a.batch], dev, mu, sd, a.train_spread)); vl.append([lq.item(), ev.item()])
        vl = np.mean(vl, 0) if vl else np.array([np.nan, np.nan]); log.append(dict(epoch=ep, q_train=agg[0] / max(b + 1, 1), ev_train=agg[2] / max(b + 1, 1), q_val=vl[0], ev_val=vl[1], secs=time.time() - t0))
        if rank == 0:
            print("EPOCH", log[-1], flush=True); pd.DataFrame(log).to_csv(out / f"log_{Y}.csv", index=False)
            crit = vl[0] if a.select == "q" else -vl[1]
            if a.include_val or crit < best: best = crit; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if ema_p is not None: ema_swap()
    if rank == 0:
        if best_state is not None: net.load_state_dict(best_state)
        torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, size=a.size), out / f"model_{Y}.pt"); net.eval(); Q, P = [], []
        with torch.no_grad():
            for i in range(0, len(te), a.batch):
                idx, w1, w2, x, cat, netR, V = T.batch(te[i:i + a.batch], dev, mu, sd, a.train_spread); q, pol = net(w1, w2, x, cat); Q.append(q.float().cpu().numpy().astype(np.float16)); P.append(M.masked_policy_logits(pol.float(), V).cpu().numpy().astype(np.float16))
        te_sorted = np.sort(te); np.savez(out / f"pred_{Y}.npz", rows=te_sorted, q=np.concatenate(Q) if Q else np.zeros((0, A.NACT), np.float16), pol=np.concatenate(P) if P else np.zeros((0, A.NACT + 1), np.float16))
        print("saved predictions", len(te_sorted), flush=True)
    if world > 1: torch.distributed.destroy_process_group()

def parser():
    ap = argparse.ArgumentParser(); ap.add_argument("--table", required=True); ap.add_argument("--out", required=True); ap.add_argument("--size", default="small"); ap.add_argument("--test_year", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch", type=int, default=512); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ent", type=float, default=0.01); ap.add_argument("--wq", type=float, default=1.0); ap.add_argument("--wp", type=float, default=1.0); ap.add_argument("--objective", default="net"); ap.add_argument("--train_spread", type=float, default=1.0)
    ap.add_argument("--frac", type=float, default=1.0); ap.add_argument("--max_rows", type=int, default=0); ap.add_argument("--tf", type=int, nargs="*"); ap.add_argument("--reset"); ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--null_shift", type=int, default=0); ap.add_argument("--plant", type=float, default=0.0); ap.add_argument("--init"); ap.add_argument("--bar_pairs", nargs="*"); ap.add_argument("--bars_start"); ap.add_argument("--bars_end"); ap.add_argument("--log_every", type=int, default=50); ap.add_argument("--grad_ckpt", action="store_true"); ap.add_argument("--mps", action="store_true")
    ap.add_argument("--min_year", type=int, default=0, help="train/val/test only on events from this year on (the day-factor table starts 2008; 2005-07 M1 data is coarse)"); ap.add_argument("--tf_pow", type=float, default=0.0, help=">0: sample training rows with weight count^-tf_pow per zone timeframe (1 = equal share per timeframe)")
    ap.add_argument("--select", default="ev", choices=["ev", "q"], help="epoch selection on the validation year: policy expected net R (ev) or Q loss (q)"); ap.add_argument("--include_val", action="store_true", help="refit mode: also train on the validation year, keep the LAST epoch (use refit.py to pick the epoch count)"); ap.add_argument("--no_atr_pips", action="store_true")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights used for validation/prediction (e.g. 0.999); 0 = off"); return ap

if __name__ == "__main__":
    run(parser().parse_args())
