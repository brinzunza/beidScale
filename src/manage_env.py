"""Approach 3 environment: dynamic trade management on GPU-replayed M1 paths. Episodes start at the fill of a base BEID entry (from Table A); each step advances K minutes and the agent chooses
0 hold | 1 exit at market | 2 move stop to breakeven | 3 trail the stop 0.75R behind the close. Stop/target resolution is conservative (stop first within a step). Reward = net R at termination.
With K=1 and the all-hold policy the realized R reproduces Table A exactly (test)."""
import numpy as np, torch
import data as DD, actions as A, windows as W
HORIZON = 2880; N_ACT = 4; OBS_DIM = 27

class Episodes:
    """per-episode constants built from Table A rows for a base action (entry e, stop j, target k, direction with)."""
    def __init__(self, T, rows, base=(2, 0, 1), spread=1.0):
        e, j, k = base; rows = np.asarray(rows); z = T.z.iloc[rows]; sg = z.side.values.astype(np.float64); tp = np.where(sg > 0, z.top.values, -z.bot.values); bt = np.where(sg > 0, z.bot.values, -z.top.values); H = tp - bt; az = z.atr15.values
        E = tp - A.DEPTH[e] * H; stops = [bt, bt - 0.25 * H, bt - 0.5 * H, E - az, E - 2 * az]; r = E - stops[j]
        tgt = A.MULT[k] * r if k < 5 else ((tp - E) if k == 5 else 2 * H); pip = np.array([DD.PIP[DD.PAIRS[p]] for p in z.pair.values])
        fil = np.asarray(T.FIL[rows])[:, e].astype(np.int64); ok = (np.asarray(T.F[rows])[:, e] > 0) & (np.asarray(T.V[rows])[:, A.aid(e, j, k, 0)] > 0) & (fil >= 0)
        self.rows = rows[ok]; self.n = int(ok.sum()); sel = np.flatnonzero(ok)
        self.sg = sg[sel]; self.E = E[sel]; self.r = r[sel]; self.stop0 = E[sel] - r[sel]; self.tgt = E[sel] + tgt[sel]; self.fill = (np.array([T.store.off1[DD.PAIRS[p]] for p in z.pair.values[sel]]) + z.ev.values[sel] + fil[sel]); self.risk_pips = r[sel] / pip[sel]
        self.H = (H / az)[sel]; self.hr = z.ny_hour.values[sel]; self.spread = spread; self.year = z.year.values[sel]
    def sub(self, mask):
        s = Episodes.__new__(Episodes)
        for k, v in self.__dict__.items(): setattr(s, k, v[mask] if isinstance(v, np.ndarray) and len(v) == self.n else v)
        s.n = int(mask.sum()); return s

class ManageEnv:
    def __init__(self, store, ep: Episodes, device, K=5, spread=1.0):
        self.s, self.K, self.dev = store, K, device; self.ep = ep; self.spread = spread; self.nsteps = int(np.ceil(HORIZON / K)); fdt = torch.float32 if device.type == 'mps' else torch.float64; t = lambda x, dt=None: torch.from_numpy(np.asarray(x)).to(device=device, dtype=dt or fdt)
        self.sg, self.E, self.r, self.stop0, self.tgt = t(ep.sg), t(ep.E), t(ep.r), t(ep.stop0), t(ep.tgt); self.fill = t(ep.fill, torch.long); self.risk_pips = t(ep.risk_pips); self.H = t(ep.H); self.hr = t(ep.hr)
        self.Nbar = store.tm1.shape[0]
    def _bars(self, idx, start, k):
        """mirrored (o,h,l,c) for bars start..start+k-1 of the selected episodes -> (B,k,4) in price units of the mirrored series."""
        ii = (start[:, None] + torch.arange(k, device=self.dev)[None, :]).clamp(max=self.Nbar - 1); x = self.s.tm1[ii].double(); sg = self.sg[idx][:, None]; pos = sg > 0
        return torch.stack([torch.where(pos, x[..., 0], -x[..., 0]), torch.where(pos, x[..., 1], -x[..., 2]), torch.where(pos, x[..., 2], -x[..., 1]), torch.where(pos, x[..., 3], -x[..., 3])], -1)
    def reset(self, idx):
        """idx: LongTensor of episode indices. Returns state dict; episodes whose fill bar already hit the stop are terminal at reset."""
        B = len(idx); st = dict(idx=idx, stop=self.stop0[idx].double().clone(), t=torch.zeros(B, device=self.dev, dtype=torch.long), done=torch.zeros(B, dtype=torch.bool, device=self.dev), R=torch.zeros(B, device=self.dev, dtype=torch.float64),
                                mfe=torch.zeros(B, device=self.dev, dtype=torch.float64), mae=torch.zeros(B, device=self.dev, dtype=torch.float64), hist=torch.zeros(B, 16, device=self.dev, dtype=torch.float64), last=torch.zeros(B, device=self.dev, dtype=torch.float64), be=torch.zeros(B, device=self.dev, dtype=torch.float64))
        fb = self._bars(idx, self.fill[idx], 1)[:, 0]; hit = fb[:, 2] <= st["stop"]; st["done"] |= hit; st["R"] = torch.where(hit, torch.full_like(st["R"], -1.0), st["R"]); st["last"] = ((fb[:, 3] - self.E[idx]) / self.r[idx]).double()
        self.hend = torch.minimum(self.fill[idx] + HORIZON, torch.full_like(self.fill[idx], self.Nbar - 1)); return st
    def _obs(self, st):
        idx = st["idx"]; r = self.r[idx].double(); E = self.E[idx].double()
        o = torch.cat([st["last"][:, None], st["mfe"][:, None], st["mae"][:, None], ((st["stop"] - E) / r)[:, None], ((self.tgt[idx].double() - E) / r)[:, None], (st["t"].double() / self.nsteps)[:, None], st["hist"], self.H[idx].double()[:, None],
                       torch.sin(2 * np.pi * self.hr[idx].double() / 24)[:, None], torch.cos(2 * np.pi * self.hr[idx].double() / 24)[:, None], st["be"][:, None], torch.ones(len(idx), 1, device=self.dev, dtype=torch.float64)], 1)
        return o.float()
    def observe(self, st): return self._obs(st)
    def step(self, st, action):
        """advance K bars. action (B,) long. Returns reward tensor (net R at termination else 0) and updates st in place."""
        idx = st["idx"]; B = len(idx); K = self.K; act = action.clone(); act[st["done"]] = 0
        start = self.fill[idx] + 1 + st["t"] * K; nb = torch.minimum(torch.full_like(start, K), self.hend + 1 - start).clamp(min=0); bars = self._bars(idx, start, K); valid = torch.arange(K, device=self.dev)[None, :] < nb[:, None]
        lo = torch.where(valid, bars[..., 2], torch.full_like(bars[..., 2], 1e18)).min(1).values; hi = torch.where(valid, bars[..., 1], torch.full_like(bars[..., 1], -1e18)).max(1).values; E = self.E[idx].double(); r = self.r[idx].double()
        # within a step: stop first (checked on the minimum low), then target on the maximum high; per-bar ordering inside the step is conservative
        stopped = (~st["done"]) & (lo <= st["stop"]); won = (~st["done"]) & (~stopped) & (hi >= self.tgt[idx].double())
        R = torch.zeros(B, device=self.dev, dtype=torch.float64); R = torch.where(stopped, (st["stop"] - E) / r, R); R = torch.where(won, (self.tgt[idx].double() - E) / r, R); ended = stopped | won
        close = torch.where(nb > 0, bars[torch.arange(B), (nb - 1).clamp(min=0), 3], st["last"] * r + E); cr = (close - E) / r
        exit_now = (~st["done"]) & (~ended) & (act == 1); R = torch.where(exit_now, cr, R); ended = ended | exit_now
        # end of horizon: forced market exit
        st["t"] = st["t"] + (~st["done"]).long(); at_end = (~st["done"]) & (~ended) & ((self.fill[idx] + 1 + st["t"] * K) > self.hend); R = torch.where(at_end, cr, R); ended = ended | at_end
        live = (~st["done"]) & (~ended)
        be = live & (act == 2) & (cr > 0); st["stop"] = torch.where(be, torch.maximum(st["stop"], E), st["stop"]); st["be"] = torch.where(be, torch.ones_like(st["be"]), st["be"])
        tr = live & (act == 3); st["stop"] = torch.where(tr, torch.maximum(st["stop"], close - 0.75 * r), st["stop"])
        st["mfe"] = torch.maximum(st["mfe"], torch.where(nb > 0, (hi - E) / r, st["mfe"])); st["mae"] = torch.maximum(st["mae"], torch.where(nb > 0, (E - lo) / r, st["mae"]))
        new_last = torch.where(nb > 0, cr, st["last"]); st["hist"] = torch.cat([st["hist"][:, 1:], (new_last - st["last"])[:, None]], 1); st["last"] = new_last
        cost = self.spread / self.risk_pips[idx].double(); newly = ended & (~st["done"]); st["R"] = torch.where(newly, R, st["R"]); st["done"] = st["done"] | ended
        return torch.where(newly, R - cost, torch.zeros_like(R)).float()
