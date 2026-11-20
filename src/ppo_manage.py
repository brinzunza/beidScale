"""PPO for dynamic trade management on ManageEnv (own compact implementation, GPU-vectorized episodes). Train on events from years < Y, evaluate greedy on year Y vs static management.
Usage: python ppo_manage.py --table DIR/tableA --test_year 2020 --out DIR --iters 300 --episodes 4096"""
import argparse, time, json, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
import data as DD, windows as W, train as TR, manage_env as ME

class AC(nn.Module):
    def __init__(self, d_in=ME.OBS_DIM, h=256):
        super().__init__(); self.body = nn.Sequential(nn.Linear(d_in, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh()); self.pi = nn.Linear(h, ME.N_ACT); self.v = nn.Linear(h, 1)
    def forward(self, x): h = self.body(x); return self.pi(h), self.v(h)[:, 0]

def rollout(env, net, idx, greedy=False):
    st = env.reset(idx); obs_l, act_l, lp_l, v_l, r_l, d_l = [], [], [], [], [], []
    for t in range(env.nsteps):
        o = env.observe(st); live = ~st["done"]
        with torch.no_grad(): lg, v = net(o); dist = torch.distributions.Categorical(logits=lg); a = lg.argmax(1) if greedy else dist.sample(); lp = dist.log_prob(a)
        r = env.step(st, a); obs_l.append(o); act_l.append(a); lp_l.append(lp); v_l.append(v); r_l.append(r); d_l.append(live)
        if st["done"].all(): break
    return st, torch.stack(obs_l), torch.stack(act_l), torch.stack(lp_l), torch.stack(v_l), torch.stack(r_l), torch.stack(d_l)

def gae(r, v, live, lam=0.95):
    T, B = r.shape; adv = torch.zeros_like(r); last = torch.zeros(B, device=r.device)
    for t in reversed(range(T)):
        nxt = v[t + 1] if t + 1 < T else torch.zeros(B, device=r.device); nxt_live = live[t + 1] if t + 1 < T else torch.zeros(B, dtype=torch.bool, device=r.device); delta = r[t] + nxt * nxt_live - v[t]; last = delta + lam * nxt_live * last; adv[t] = last
    return adv, adv + v

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--table", required=True); ap.add_argument("--out", required=True); ap.add_argument("--test_year", type=int, required=True); ap.add_argument("--iters", type=int, default=200); ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--K", type=int, default=5); ap.add_argument("--spread", type=float, default=1.0); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--base", type=int, nargs=3, default=[2, 0, 1]); ap.add_argument("--bar_pairs", nargs="*"); ap.add_argument("--bars_start"); ap.add_argument("--bars_end"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--mps", action="store_true"); a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if a.mps and torch.backends.mps.is_available() else "cpu")); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    T = TR.Table(a.table); T.FIL = np.load(Path(a.table) / "FIL.npy", mmap_mode="r"); store = W.BarStore(pairs=tuple(a.bar_pairs or DD.PAIRS), start=a.bars_start, end=a.bars_end).to(dev); T.set_store(store)
    Y = a.test_year; rows = np.flatnonzero((T.year < Y - 0) | (T.year == Y)); ep = ME.Episodes(T, rows, tuple(a.base), a.spread); tr = (ep.year < Y) & (T.t_ev[ep.rows] + 72 * 3600 * 10 ** 9 < np.datetime64(f"{Y}-01-01", "ns").astype(np.int64)); te = ep.year == Y;  # (train episodes whose 48h path reaches into the test year are purged)
    ept, epe = ep.sub(tr), ep.sub(te)
    envt = ME.ManageEnv(store, ept, dev, a.K, a.spread); enve = ME.ManageEnv(store, epe, dev, a.K, a.spread); print(f"episodes train {ept.n} test {epe.n} device {dev}", flush=True)
    net = AC().to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr); log = []
    def evaluate(policy):
        idx = torch.arange(epe.n, device=dev); Rs = []
        for b in range(0, epe.n, 8192):
            ii = idx[b:b + 8192]; st = enve.reset(ii)
            for t in range(enve.nsteps):
                o = enve.observe(st)
                if policy == "learned":
                    with torch.no_grad(): act = net(o)[0].argmax(1)
                elif policy == "hold": act = torch.zeros(len(ii), dtype=torch.long, device=dev)
                else: act = torch.randint(0, ME.N_ACT, (len(ii),), device=dev)
                enve.step(st, act)
                if st["done"].all(): break
            Rs.append((st["R"] - enve.spread / enve.risk_pips[ii].double()).cpu().numpy())
        return np.concatenate(Rs)
    for it in range(a.iters):
        idx = torch.from_numpy(rng.choice(ept.n, a.episodes, replace=ept.n < a.episodes)).to(dev); st, obs, act, lp, v, r, live = rollout(envt, net, idx); adv, ret = gae(r, v, live)
        m = live.reshape(-1); O = obs.reshape(-1, obs.shape[-1])[m]; Ac = act.reshape(-1)[m]; LP = lp.reshape(-1)[m]; AD = adv.reshape(-1)[m]; RT = ret.reshape(-1)[m]; AD = (AD - AD.mean()) / (AD.std() + 1e-8)
        for ep_ in range(4):
            perm = torch.randperm(len(O), device=dev)
            for b in range(0, len(O), 16384):
                j = perm[b:b + 16384]; lg, vv = net(O[j]); dist = torch.distributions.Categorical(logits=lg); ratio = (dist.log_prob(Ac[j]) - LP[j]).exp(); l1 = ratio * AD[j]; l2 = ratio.clamp(0.8, 1.2) * AD[j]
                loss = -torch.min(l1, l2).mean() + 0.5 * ((vv - RT[j]) ** 2).mean() - 0.01 * dist.entropy().mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net_r = (st["R"] - envt.spread / envt.risk_pips[idx].double()).mean().item()
        if it % 10 == 0: log.append(dict(iter=it, train_net_R=net_r)); print(f"iter {it} train net R/episode {net_r:+.4f} entropy {dist.entropy().mean().item():.3f}", flush=True)
    res = {}
    for pol in ("hold", "random", "learned"):
        R = evaluate(pol); mo = pd.to_datetime(T.z.t_ev.values[epe.rows]).to_period("M").astype(str).values; import evaluate as EV; m, t = EV.cluster_t(R, mo); res[pol] = dict(n=len(R), mean_net_R=float(m), t=float(t), win=float((R > 0).mean()))
    print(json.dumps(res, indent=1)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); json.dump(dict(res=res, log=log, args=vars(a)), open(out / f"manage_{Y}.json", "w")); torch.save(net.state_dict(), out / f"manage_{Y}.pt")

if __name__ == "__main__":
    main()
