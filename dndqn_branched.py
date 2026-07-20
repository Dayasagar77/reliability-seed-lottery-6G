#!/usr/bin/env python3
"""dndqn_branched.py — Branching Dueling Q-Network (BDQ) for the Paper 1 novel mechanism:
a cliff-aware, physics-informed JOINT controller. Two action branches emitted simultaneously:

  Branch R (reliability): {STAY_PROTECT, STAY_SAVE, HANDOVER}
       -> transmit power collapsed to a binary protect/save actuator, paired with handover
  Branch S (spectrum):    {RB_HOLD, RB_DOWN, RB_UP}
       -> the continuous efficiency lever, kept fine-grained

Shared dueling encoder + one advantage stream per branch (Tavakoli et al., 2017). Compatible with
Double-DQN, N-step returns, and PER. The reward is a single scalar for the joint (R,S) action and is
shared across branches; per-branch Double-DQN targets are averaged (BDQ) for a low-variance update.

This module is the *substrate*; the novelty is the cliff-driven action geometry realized in
env_6g_branched.py (binary power + handover pairing + continuous spectrum).
"""
import torch, torch.nn as nn, numpy as np

# Branch layout — names, sizes, and human-readable action labels
R_STAY_PROTECT, R_STAY_SAVE, R_HANDOVER = 0, 1, 2
S_HOLD, S_DOWN, S_UP = 0, 1, 2
BRANCH_NAMES = ['reliability', 'spectrum']
BRANCH_SIZES = [3, 3]
RELIABILITY_LABELS = ['STAY_PROTECT', 'STAY_SAVE', 'HANDOVER']
SPECTRUM_LABELS = ['RB_HOLD', 'RB_DOWN', 'RB_UP']


class BranchedDuelingQNet(nn.Module):
    """Shared encoder -> one state-value head + one advantage head per action branch."""
    def __init__(self, obs_dim, branch_sizes=BRANCH_SIZES, hidden=(128, 128, 64)):
        super().__init__()
        layers = []; d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        self.shared = nn.Sequential(*layers)
        self.value = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, n)) for n in branch_sizes
        ])
        self.branch_sizes = list(branch_sizes)

    def forward(self, x):
        z = self.shared(x)
        v = self.value(z)                                   # (B,1)
        out = []
        for head in self.adv:
            a = head(z)                                     # (B, n_b)
            out.append(v + a - a.mean(dim=1, keepdim=True)) # dueling combine, per branch
        return out                                          # list of (B, n_b)

    def act(self, x):
        """Greedy per-branch action for a single obs (1D tensor/array)."""
        if not torch.is_tensor(x):
            x = torch.FloatTensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            qs = self.forward(x)
        return [int(q.argmax(dim=-1).item()) for q in qs]   # e.g. [R_action, S_action]


def build_branched_networks(obs_dim, branch_sizes=BRANCH_SIZES, device='cpu'):
    online = BranchedDuelingQNet(obs_dim, branch_sizes).to(device)
    target = BranchedDuelingQNet(obs_dim, branch_sizes).to(device)
    target.load_state_dict(online.state_dict())
    return online, target


def branched_td_loss(online, target, ob, ab, rb, nb, db, wb, gamma_n):
    """Branched Double-DQN loss with PER importance weights.
      ob,nb : (B, obs_dim) obs / next-obs
      ab    : (B, n_branches) int actions taken per branch
      rb,db : (B,) shared scalar reward / done
      wb    : (B,) PER importance weights
      gamma_n : gamma ** n_step
    Returns (loss_tensor, per_sample_abs_td) — abs TD averaged over branches (for PER priorities)."""
    ot = torch.FloatTensor(ob); nt = torch.FloatTensor(nb)
    at = torch.LongTensor(ab);  rt = torch.FloatTensor(rb)
    dt = torch.FloatTensor(db); wt = torch.FloatTensor(wb)
    q_online = online(ot)
    with torch.no_grad():
        q_next_online = online(nt)
        q_next_target = target(nt)
    loss = 0.0
    td_accum = np.zeros(len(rb), dtype=np.float64)
    n_branches = len(q_online)
    for bi in range(n_branches):
        cq = q_online[bi].gather(1, at[:, bi:bi + 1]).squeeze(1)
        with torch.no_grad():
            na = q_next_online[bi].argmax(1, keepdim=True)          # Double-DQN: online picks
            nq = q_next_target[bi].gather(1, na).squeeze(1)         # target evaluates
            tq = rt + gamma_n * nq * (1.0 - dt)
        loss = loss + (wt * nn.functional.huber_loss(cq, tq, reduction='none')).mean()
        td_accum += (cq - tq).detach().cpu().numpy()
    return loss / n_branches, np.abs(td_accum / n_branches)


def save_branched(net, path, extra=None):
    d = {'q_net_state': net.state_dict(), 'branch_sizes': net.branch_sizes,
         'branch_names': BRANCH_NAMES, 'algorithm': 'BDQ-cliff-aware'}
    if extra: d.update(extra)
    torch.save(d, path)


def load_branched(path, obs_dim, device='cpu'):
    ck = torch.load(path, map_location=device)
    net = BranchedDuelingQNet(obs_dim, ck.get('branch_sizes', BRANCH_SIZES)).to(device)
    net.load_state_dict(ck['q_net_state']); net.eval()
    return net, ck


if __name__ == "__main__":
    # self-test: shapes + a single forward/act + one loss step
    obs_dim = 6
    online, target = build_branched_networks(obs_dim)
    x = torch.randn(4, obs_dim)
    qs = online(x)
    assert len(qs) == 2 and qs[0].shape == (4, 3) and qs[1].shape == (4, 3), [q.shape for q in qs]
    a = online.act(torch.randn(obs_dim))
    assert len(a) == 2 and all(0 <= ai < 3 for ai in a), a
    ob = np.random.randn(8, obs_dim); nb = np.random.randn(8, obs_dim)
    ab = np.random.randint(0, 3, size=(8, 2)); rb = np.random.randn(8)
    db = np.zeros(8); wb = np.ones(8)
    loss, td = branched_td_loss(online, target, ob, ab, rb, nb, db, wb, gamma_n=0.99 ** 3)
    assert td.shape == (8,), td.shape
    print(f"self-test OK | branches={online.branch_sizes} | forward {[tuple(q.shape) for q in qs]} | "
          f"act={a} | loss={float(loss):.4f} | td[0]={td[0]:.4f}")
