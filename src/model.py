"""Sequence model for BEID zone events: conv/patch stems over the M1 and M15 windows + a context token (zone geometry, day vector, categorical embeddings) -> transformer -> Q head (420 actions) and policy head (420 + skip)."""
import torch, torch.nn as nn, torch.nn.functional as F
import actions as A
CONFIGS = {"tiny": dict(d=96, layers=3, heads=4), "small": dict(d=256, layers=8, heads=8), "base": dict(d=512, layers=12, heads=8), "large": dict(d=768, layers=16, heads=12)}

class BEIDNet(nn.Module):
    def __init__(self, size="small", n_ctx=50, n_cat=(4, 5, 7, 2), dropout=0.1, d=None, layers=None, heads=None, checkpoint=False):
        super().__init__(); c = CONFIGS[size]; d = d or c["d"]; layers = layers or c["layers"]; heads = heads or c["heads"]; self.d = d; self.checkpoint = checkpoint
        self.stem1 = nn.Conv1d(6, d, 8, 8); self.stem2 = nn.Conv1d(6, d, 4, 4); self.pos1 = nn.Parameter(torch.zeros(1, 32, d)); self.pos2 = nn.Parameter(torch.zeros(1, 24, d)); self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.ctx = nn.Sequential(nn.Linear(n_ctx, d), nn.GELU(), nn.Linear(d, d)); self.embs = nn.ModuleList([nn.Embedding(k, d) for k in n_cat]); self.tok_type = nn.Parameter(torch.zeros(4, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True); self.enc = nn.TransformerEncoder(layer, layers); self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, A.NACT); self.pol = nn.Linear(d, A.NACT + 1); nn.init.normal_(self.pos1, std=0.02); nn.init.normal_(self.pos2, std=0.02); nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.tok_type, std=0.02)
    def encode(self, w1, w2, ctx, cat):
        B = w1.shape[0]; t1 = self.stem1(w1.transpose(1, 2)).transpose(1, 2) + self.pos1 + self.tok_type[0]; t2 = self.stem2(w2.transpose(1, 2)).transpose(1, 2) + self.pos2 + self.tok_type[1]
        tc = (self.ctx(ctx) + sum(e(cat[:, i]) for i, e in enumerate(self.embs)))[:, None, :] + self.tok_type[2]; x = torch.cat([self.cls.expand(B, -1, -1) + self.tok_type[3], tc, t1, t2], 1)
        x = self.enc(x); return self.norm(x[:, 0])
    def forward(self, w1, w2, ctx, cat):
        h = self.encode(w1, w2, ctx, cat); return self.q(h), self.pol(h)
    def n_params(self): return sum(p.numel() for p in self.parameters())

def masked_policy_logits(pol, V):
    """invalid actions get -inf; the skip action (index NACT) is always available."""
    mask = torch.cat([V > 0, torch.ones(V.shape[0], 1, dtype=torch.bool, device=V.device)], 1); return pol.masked_fill(~mask, -1e4)

def policy_loss(pol, V, net, ent_coef=0.01, objective="net"):
    """exact expected reward under the softmax policy over valid actions + skip (reward 0). net: (B, 420) net R with invalid = 0."""
    lg = masked_policy_logits(pol.float(), V); p = F.softmax(lg, 1); net0 = torch.cat([net, torch.zeros(net.shape[0], 1, device=net.device)], 1)
    r = torch.log1p(0.01 * net0) if objective == "logg" else net0; ev = (p * r).sum(1).mean(); ent = -(p * torch.log(p + 1e-9)).sum(1).mean(); return -ev - ent_coef * ent, ev.detach(), ent.detach()

def q_loss(q, net, V):
    m = (V > 0).float(); l = F.huber_loss(q.float(), net, reduction="none", delta=1.0); return (l * m).sum() / m.sum().clamp(min=1)
