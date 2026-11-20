"""Action grid for a BEID zone event (long perspective; sell zones are mirrored). 6 entries x 5 stops x 7 targets x 2 directions = 420 actions (+ skip)."""
import numpy as np
DEPTH = (0.0, 0.25, 0.5, 0.75, 1.0)               # limit depth from the near (top) edge to the far edge; entry index 5 = market at the event
ENT, NSTOP, NTGT, NDIR = 6, 5, 7, 2; NACT = ENT * NSTOP * NTGT * NDIR; MULT = (0.5, 1.0, 1.5, 2.0, 3.0)
STOP_NAMES = ("far_edge", "far-0.25H", "far-0.5H", "1ATR15", "2ATR15"); TGT_NAMES = ("0.5R", "1R", "1.5R", "2R", "3R", "zone_top", "2H"); ENT_NAMES = ("top", "25%", "50%", "75%", "far", "market"); DIR_NAMES = ("with", "fade")
def aid(e, j, k, d): return ((e * NSTOP + j) * NTGT + k) * NDIR + d
def decode(a):
    d = a % NDIR; r = a // NDIR; k = r % NTGT; r //= NTGT; j = r % NSTOP; e = r // NSTOP; return e, j, k, d
def name(a):
    e, j, k, d = decode(a); return f"{ENT_NAMES[e]}|{STOP_NAMES[j]}|{TGT_NAMES[k]}|{DIR_NAMES[d]}"
def all_names(): return [name(a) for a in range(NACT)]
def entry_of(a): return np.asarray(a) // (NSTOP * NTGT * NDIR)
def stop_index_of(a): return (np.asarray(a) // (NTGT * NDIR)) % NSTOP
def risk_slot(a):
    """flat (entry, stop) slot index 0..29 for risk arrays of shape (n, 6, 5) flattened."""
    return entry_of(a) * NSTOP + stop_index_of(a)
def net_r(R, F, RISKP, spread_pips, valid=None):
    """R (n,420) gross, F (n,6) filled, RISKP (n,6,5) risk in pips -> net R at a round-trip spread. cost applies only to filled valid actions."""
    n = R.shape[0]; ent = np.repeat(np.arange(ENT), NSTOP * NTGT * NDIR); slot = risk_slot(np.arange(NACT)); risk = RISKP.reshape(n, -1)[:, slot]
    with np.errstate(divide="ignore", invalid="ignore"): cost = np.where(F[:, ent] > 0, spread_pips / risk, 0.0)
    out = R - cost
    if valid is not None: out = np.where(valid > 0, out, 0.0)
    return out
