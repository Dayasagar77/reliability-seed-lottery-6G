#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q4_qr_branched.py - the algorithmic change: measurement-keyed risk asymmetry.

WHAT IS ACTUALLY NEW HERE, AND WHAT IS NOT
------------------------------------------
Not new: distributional value learning (QR-DQN, Dabney et al. 2018), CVaR-greedy control from
quantiles, branched action spaces (BDQ, Tavakoli et al. 2018). All prior art, all cited.

New: WHICH STATISTIC EACH BRANCH ACTS ON, AND WHY IT WAS CHOSEN.

    reliability branch  -> measured cliff-coupled  -> acts on CVaR(alpha=0.25)
    spectrum branch     -> measured decoupled      -> acts on the mean

The assignment is read off the same policy-free, pre-training sweep that decided the action-space
geometry, not tuned. On a cliff the downside tail is the quantity that matters; on the decoupled
axis there is no cliff to fall off and the expectation is the right statistic.

THE OBJECT
----------
Shared FC128-FC128 encoder, exactly the published trunk. One state-value head emitting N quantiles,
one advantage head per branch emitting (n_actions x N) quantiles, combined per branch in quantile
space with the same dueling identity the published network uses:

    Z_b(s, a) = V(s) + A_b(s, a) - mean_a' A_b(s, a')          (elementwise over the N quantiles)

Greedy action per branch is argmax over actions of that branch's RISK OPERATOR applied across the
quantile axis.

TRAINING
--------
Quantile Huber loss (kappa = 1.0) at midpoint quantile levels tau_i = (i - 0.5)/N, with per-branch
Double-DQN action selection - the online network chooses the next action under ITS OWN branch risk
operator, the target network supplies the quantiles. Per-branch losses are averaged, and the
per-sample absolute TD returned for PER priorities is averaged over branches, matching
dndqn_branched.branched_td_loss.

WHAT THIS DOES NOT CLAIM - a convergence caveat that must travel with every result
---------------------------------------------------------------------------------
The reliability branch performs CVaR-GREEDY ACTION SELECTION under the STANDARD distributional
Bellman operator. That is not the same thing as optimising CVaR, and the difference is published:

    Lim, S. H. & Malalur, I. "Distributional Reinforcement Learning for Risk-Sensitive Policies."
    Advances in Neural Information Processing Systems 35 (2022).

They show that "the standard action-selection strategy when applying the distributional Bellman
optimality operator can result in convergence to neither the dynamic, Markovian CVaR nor the
static, non-Markovian CVaR", and propose a modified operator.

Consequences, stated so they cannot be forgotten later:
  * this file implements the standard strategy, deliberately, because it is the one that keeps the
    published trainer and the published evaluator untouched;
  * the ablation remains valid - qr_sym, qr_asym and qr_inv differ ONLY in the risk operator, so a
    difference between them is attributable to the operator whatever it converges to;
  * but no result from this code may be written as "the branch optimises CVaR". The admissible
    wording is "the branch selects actions by the CVaR of its predicted return distribution under
    the standard distributional Bellman operator", with the Lim & Malalur caveat cited.

PRIOR ART this design sits inside, established by the gate of 10 Aug 2026 - none of it is claimed
as novel here:
  * distributional value learning: Dabney et al. 2018 (QR-DQN); Bellemare et al. 2017 (C51)
  * CVaR-greedy and risk-sensitive control: Lim & Malalur 2022; Ying et al. 2022 (CPPO)
  * branched/factored action spaces: Tavakoli et al. 2018 (BDQ)
  * CVaR in wireless resource management: Zhang et al., IEEE Trans. Wireless Commun. 2024
    (eMBB/URLLC scheduling with CVaR as the risk criterion) - so no wireless-CVaR novelty exists
  * distributional RL for RRM: arXiv 2409.16764 (QR-DQN + conservative Q-learning, factored actions)
  * heterogeneous risk ACROSS AGENTS: Shen et al. 2023 (RiskQ). Its ablation reports that mixing
    risk measures across agents "performs poorly", because the RIGM principle requires agents to
    share one measure. That constraint is about reconstructing a centralised argmax from
    decentralised ones; this controller is a single agent with one shared scalar reward and no
    decentralised execution, so RIGM does not apply. The manuscript must say this explicitly.

What the 10 Aug gate found unclaimed: heterogeneous risk statistics across ACTION BRANCHES WITHIN
ONE AGENT, assigned by a pre-training sensitivity measurement.

Interface is deliberately identical to the published pair
    build_branched_networks(obs_dim) -> (online, target)
    branched_td_loss(online, target, ob, ab, rb, nb, db, wb, gamma_n) -> (loss, abs_td)
    save_branched(net, path, extra)
so the published trainer can be used unmodified by rebinding those three names.
"""
import torch
import torch.nn as nn
import numpy as np

N_QUANTILES = 51            # the value of the original QR-DQN paper; not tuned here
CVAR_ALPHA = 0.25           # standard default in the risk-sensitive RL literature; not tuned here
HUBER_KAPPA = 1.0
BRANCH_SIZES = [3, 3]
BRANCH_NAMES = ['reliability', 'spectrum']

# the measurement-keyed assignment: which risk operator each branch acts on
RISK_ASYM = ['cvar', 'mean']       # reliability -> tail, spectrum -> expectation   (PROPOSED)
RISK_SYM = ['mean', 'mean']        # both -> expectation                            (baseline)
RISK_INV = ['mean', 'cvar']        # inverted                                       (falsifier)


def taus(n=N_QUANTILES, device='cpu'):
    """Midpoint quantile levels tau_i = (i - 0.5)/n."""
    return (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / n


def reduce_risk(z, mode, alpha=CVAR_ALPHA):
    """Collapse the quantile axis (last) to a scalar score.

    mode 'mean' -> ordinary expectation.
    mode 'cvar' -> mean of the lowest ceil(alpha * n) quantiles, i.e. the CVaR of the PREDICTED
                   return distribution at level alpha. At alpha = 1.0 this is exactly the mean,
                   which is the reduction check in q4_smoke.py.

    This is a statistic OF THE PREDICTION, used for action selection. It is not a claim that the
    learned policy optimises CVaR - see the convergence caveat in this module's docstring
    (Lim & Malalur, NeurIPS 2022).
    """
    if mode == 'mean':
        return z.mean(dim=-1)
    if mode != 'cvar':
        raise ValueError('unknown risk mode %r' % mode)
    n = z.shape[-1]
    k = max(1, int(np.ceil(alpha * n)))
    if k >= n:
        return z.mean(dim=-1)
    lo, _ = torch.sort(z, dim=-1)
    return lo[..., :k].mean(dim=-1)


class BranchedQRNet(nn.Module):
    """Shared dueling trunk; per-branch quantile advantage heads; per-branch risk operator."""

    def __init__(self, obs_dim, branch_sizes=BRANCH_SIZES, hidden=(128, 128, 64),
                 n_quantiles=N_QUANTILES, risk=RISK_ASYM, alpha=CVAR_ALPHA):
        super().__init__()
        h1, h2, h3 = hidden
        self.encoder = nn.Sequential(nn.Linear(obs_dim, h1), nn.ReLU(),
                                     nn.Linear(h1, h2), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(h2, h3), nn.ReLU(), nn.Linear(h3, n_quantiles))
        self.adv = nn.ModuleList([
            nn.Sequential(nn.Linear(h2, h3), nn.ReLU(), nn.Linear(h3, n * n_quantiles))
            for n in branch_sizes])
        self.branch_sizes = list(branch_sizes)
        self.n_quantiles = int(n_quantiles)
        self.risk = list(risk)
        self.alpha = float(alpha)
        if len(self.risk) != len(self.branch_sizes):
            raise ValueError('one risk mode per branch is required')

    def forward(self, x):
        """-> list of (batch, n_actions_b, n_quantiles), one per branch."""
        f = self.encoder(x)
        v = self.value(f).unsqueeze(1)                       # (B, 1, N)
        out = []
        for bi, n in enumerate(self.branch_sizes):
            a = self.adv[bi](f).view(-1, n, self.n_quantiles)
            out.append(v + a - a.mean(dim=1, keepdim=True))  # dueling combine in quantile space
        return out

    def scores(self, x):
        """-> list of (batch, n_actions_b): each branch's quantiles reduced by ITS risk operator."""
        return [reduce_risk(z, self.risk[bi], self.alpha)
                for bi, z in enumerate(self.forward(x))]

    @torch.no_grad()
    def act(self, obs):
        """Greedy per-branch action for a single observation. Same signature as the published net."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        return [int(s.argmax(dim=1).item()) for s in self.scores(x)]


def build_qr_networks(obs_dim, branch_sizes=BRANCH_SIZES, device='cpu',
                      risk=RISK_ASYM, alpha=CVAR_ALPHA, n_quantiles=N_QUANTILES):
    online = BranchedQRNet(obs_dim, branch_sizes, n_quantiles=n_quantiles,
                           risk=risk, alpha=alpha).to(device)
    target = BranchedQRNet(obs_dim, branch_sizes, n_quantiles=n_quantiles,
                           risk=risk, alpha=alpha).to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    return online, target


def quantile_huber(td, tau, kappa=HUBER_KAPPA):
    """td: (B, Nt, N) with target quantiles on dim 1 and predicted on dim 2. tau: (N,)."""
    absd = td.abs()
    huber = torch.where(absd <= kappa, 0.5 * td.pow(2), kappa * (absd - 0.5 * kappa))
    w = (tau.view(1, 1, -1) - (td.detach() < 0).float()).abs()
    return (w * huber / kappa).sum(dim=2).mean(dim=1)        # (B,)


def qr_branched_td_loss(online, target, ob, ab, rb, nb, db, wb, gamma_n):
    """Drop-in replacement for dndqn_branched.branched_td_loss.

    ab : (B, n_branches) int actions taken per branch
    Returns (loss_tensor, per_sample_abs_td) with abs TD averaged over branches, as PER expects.

    The bootstrap uses the STANDARD distributional Bellman operator: the online net selects the
    next action under its own per-branch risk operator, the target net supplies that action's
    quantiles. Deliberate, and caveated in the module docstring - this converges to neither the
    Markovian nor the static CVaR (Lim & Malalur, NeurIPS 2022). The ablation is unaffected
    because every arm shares this operator and differs only in the risk statistic.
    """
    dev = next(online.parameters()).device
    ob = torch.as_tensor(np.asarray(ob, dtype=np.float32), device=dev)
    nb_ = torch.as_tensor(np.asarray(nb, dtype=np.float32), device=dev)
    ab = torch.as_tensor(np.asarray(ab, dtype=np.int64), device=dev)
    rb = torch.as_tensor(np.asarray(rb, dtype=np.float32), device=dev).view(-1, 1)
    db = torch.as_tensor(np.asarray(db, dtype=np.float32), device=dev).view(-1, 1)
    wb = torch.as_tensor(np.asarray(wb, dtype=np.float32), device=dev).view(-1)

    tau = taus(online.n_quantiles, dev)
    z_online = online(ob)
    with torch.no_grad():
        z_next_online = online(nb_)                 # Double-DQN: online selects
        z_next_target = target(nb_)                 # target evaluates
    loss = 0.0
    td_accum = np.zeros(len(wb), dtype=np.float64)
    nbranch = len(online.branch_sizes)
    for bi in range(nbranch):
        n_q = online.n_quantiles
        a_taken = ab[:, bi].view(-1, 1, 1).expand(-1, 1, n_q)
        z_pred = z_online[bi].gather(1, a_taken).squeeze(1)               # (B, N)
        sel = reduce_risk(z_next_online[bi], online.risk[bi], online.alpha).argmax(dim=1)
        a_next = sel.view(-1, 1, 1).expand(-1, 1, n_q)
        z_next = z_next_target[bi].gather(1, a_next).squeeze(1)           # (B, N)
        y = rb + gamma_n * (1.0 - db) * z_next                            # (B, N) target quantiles
        td = y.detach().unsqueeze(2) - z_pred.unsqueeze(1)                # (B, Nt, N)
        per_sample = quantile_huber(td, tau)                              # (B,)
        loss = loss + (wb * per_sample).mean()
        td_accum += td.detach().mean(dim=(1, 2)).abs().cpu().numpy().astype(np.float64)
    return loss / nbranch, np.abs(td_accum / nbranch)


def save_qr(net, path, extra=None):
    d = {'q_net_state': net.state_dict(), 'branch_sizes': net.branch_sizes,
         'branch_names': BRANCH_NAMES, 'algorithm': 'QR-BDQ-risk-asymmetric',
         'n_quantiles': net.n_quantiles, 'risk': net.risk, 'alpha': net.alpha}
    if extra:
        d.update(extra)
    torch.save(d, path)


def load_qr(path, obs_dim, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = BranchedQRNet(obs_dim, ck.get('branch_sizes', BRANCH_SIZES),
                        n_quantiles=ck.get('n_quantiles', N_QUANTILES),
                        risk=ck.get('risk', RISK_ASYM),
                        alpha=ck.get('alpha', CVAR_ALPHA)).to(device)
    net.load_state_dict(ck['q_net_state'])
    net.eval()
    return net, ck
