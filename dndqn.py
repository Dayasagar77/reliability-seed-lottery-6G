#!/usr/bin/env python3
"""
dndqn.py  —  Dueling N-step DQN (Dn-DQN)
====================================================================
Shared module imported by all Dn-DQN training, ablation, and eval scripts.

ALGORITHM: Dn-DQN
  A novel Deep Q-Network architecture specifically designed for joint
  handover and radio resource management in Sub-THz vehicular networks.

COMPONENTS:
  1. Dueling architecture  — V(s) + A(s,a) decomposition
  2. Double Q-learning      — online net selects action, target net evaluates
  3. N-step PER buffer      — n=3 with episode boundary handling
  4. Prioritized Experience Replay — PER-α=0.6

MATHEMATICAL DEFINITION:
  Shared encoder: h = ReLU(FC_128(ReLU(FC_128(obs))))
  Value stream:   V(s)   = FC_1(ReLU(FC_64(h)))
  Advantage stream: A(s,a) = FC_6(ReLU(FC_64(h)))
  Q-value:  Q(s,a) = V(s) + (A(s,a) − mean_a' A(s,a'))

N-STEP RETURNS WITH EPISODE BOUNDARY HANDLING:
  Naive n-step implementations silently corrupt Q-values at episode boundaries
  by bootstrapping from the wrong episode. This implementation tracks `done`
  flags per step and truncates the n-step return at episode termination:
    y = Σ_{k=0}^{K-1} γ^k r_{t+k} + γ^K × Q_target(s_{t+K}, a*)
  where K = min(n, steps_until_done).

ACADEMIC INTEGRITY:
  All components are standard implementations with cited sources:
  - Dueling DQN: Wang et al. (2016) "Dueling Network Architectures"
  - PER: Schaul et al. (2016) "Prioritized Experience Replay"
  - Double DQN: van Hasselt et al. (2016) "Deep RL with Double Q-learning"
  - N-step: Watkins (1989), Sutton & Barto (2018) Ch.7
"""

import random
import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

OBS_DIM  = 6
N_ACTIONS = 6
NET_ARCH  = [128, 128, 64]   # shared → (value_head, advantage_head)


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDARD QNetwork (for ablation config 6b — architecture comparison)
# ═══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    Standard DQN Q-Network.
    Used ONLY in ablation config 6b (Standard DQN vs Dn-DQN comparison).
    All other configs use DnDQN.
    Architecture: obs → FC128 → ReLU → FC128 → ReLU → FC64 → ReLU → FC6 (Q-values)
    """
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, net_arch=NET_ARCH):
        super().__init__()
        layers, in_dim = [], obs_dim
        for h in net_arch:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  Dn-DQN NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class DnDQN(nn.Module):
    """
    Dueling N-step DQN Network.

    Architecture:
      Shared encoder: FC128 → ReLU → FC128 → ReLU
      Value stream:   FC64 → ReLU → FC1  →  V(s)
      Advantage stream: FC64 → ReLU → FC6  →  A(s,a)
      Q-value: Q(s,a) = V(s) + (A(s,a) - mean_a' A(s,a'))


    Args:
        obs_dim:   Input observation dimension (default 6)
        n_actions: Number of discrete actions (default 6)
        net_arch:  [shared_h1, shared_h2, head_h] (default [128, 128, 64])
    """

    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS,
                 net_arch=NET_ARCH):
        super().__init__()
        assert len(net_arch) == 3, "net_arch must be [shared_h1, shared_h2, head_h]"

        shared_h1, shared_h2, head_h = net_arch
        self.n_actions = n_actions

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, shared_h1), nn.ReLU(),
            nn.Linear(shared_h1, shared_h2), nn.ReLU(),
        )

        # Value stream: scalar V(s)
        self.value = nn.Sequential(
            nn.Linear(shared_h2, head_h), nn.ReLU(),
            nn.Linear(head_h, 1),
        )

        # Advantage stream: A(s,a) for all actions
        self.advantage = nn.Sequential(
            nn.Linear(shared_h2, head_h), nn.ReLU(),
            nn.Linear(head_h, n_actions),
        )

        # Weight initialisation — orthogonal init for stable gradients
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, obs_dim) or (obs_dim,)
        Returns:
            Q: tensor of shape (batch, n_actions) — dueling Q-values
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)   # handle single observation (no batch dim)

        # Shared encoding
        h    = self.shared(x)                          # (batch, shared_h2)

        # Value and advantage heads
        v    = self.value(h)                           # (batch, 1)
        adv  = self.advantage(h)                       # (batch, n_actions)

        # Mean-subtracted advantage (identifiability fix — Wang et al. 2016)
        adv_mean   = adv.mean(dim=1, keepdim=True)      # (batch, 1)
        adv_center = adv - adv_mean                     # (batch, n_actions)

        # Dueling Q-values
        Q = v + adv_center                              # (batch, n_actions)

        return Q

    def get_value(self, x):
        """Return V(s) only — used for logging and analysis."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.shared(x)
        return self.value(h)

    def get_advantage(self, x):
        """Return raw A(s,a) — used for analysis and debugging."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.shared(x)
        return self.advantage(h)


# ═══════════════════════════════════════════════════════════════════════════════
#  N-STEP PRIORITIZED EXPERIENCE REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class SumTree:
    """
    Binary sum tree for O(log N) priority sampling.
    Leaf nodes store priorities; internal nodes store subtree sums.
    """
    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity, dtype=np.float64)
        self.data      = [None] * capacity
        self.write_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx, s):
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):  return idx
        if right >= len(self.tree): return left
        return self._retrieve(left, s) if s <= self.tree[left] \
               else self._retrieve(right, s - self.tree[left])

    @property
    def total(self): return float(self.tree[0])

    def add(self, priority, data):
        leaf = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(leaf, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf, priority):
        self._propagate(leaf, priority - self.tree[leaf])
        self.tree[leaf] = priority

    def get(self, s):
        s    = float(np.clip(s, 0.0, self.total - 1e-6))
        leaf = self._retrieve(0, s)
        leaf = int(np.clip(leaf, self.capacity - 1, len(self.tree) - 1))
        data_idx = int(np.clip(leaf - self.capacity + 1, 0, self.capacity - 1))
        return leaf, self.tree[leaf], self.data[data_idx]


class NStepPERBuffer:
    """
    N-step Prioritized Experience Replay Buffer.

    Combines:
      1. N-step return accumulation with episode boundary handling
      2. Prioritized sampling via SumTree

    N-STEP LOGIC:
      Maintains a rolling n-step buffer per interaction.
      When a transition (obs, action, reward, next_obs, done) arrives:
        - It is added to the n-step queue.
        - If queue length == n: compute n-step return and store in SumTree.
        - If done=True: flush the remaining queue with truncated returns.
      This guarantees no cross-episode contamination.

    STORED TRANSITION:
      (obs_t, action_t, n_step_return, obs_{t+n}, done_at_step_n)
      where:
        n_step_return = Σ_{k=0}^{K-1} γ^k r_{t+k}
        K = min(n, steps_until_done)
        done_at_step_n = True if episode ended before step t+n

    Args:
        capacity:   Maximum number of transitions in buffer
        n_steps:    Number of steps for n-step return (default 3)
        gamma:      Discount factor (default 0.99)
        alpha:      PER priority exponent (default 0.6)
        eps:        Priority epsilon for numerical stability (default 1e-6)
    """

    def __init__(self, capacity, n_steps=3, gamma=0.99,
                 alpha=0.6, eps=1e-6):
        self.tree      = SumTree(capacity)
        self.n_steps   = n_steps
        self.gamma     = gamma
        self.alpha     = alpha
        self.eps       = eps
        self.max_prio  = 1.0

        # Rolling n-step queue: stores (obs, action, reward, done) tuples
        self._nstep_queue = []

    def _compute_nstep_return(self, queue):
        """
        Compute n-step discounted return from queue.
        queue: list of (obs, action, reward, next_obs, done) tuples
        Returns: (obs_t, action_t, n_step_return, obs_last, done_last)
        """
        obs_t    = queue[0][0]
        action_t = queue[0][1]
        obs_last = queue[-1][3]
        done_last = queue[-1][4]

        nstep_return = 0.0
        for k, (_, _, r, _, d) in enumerate(queue):
            nstep_return += (self.gamma ** k) * r
            if d:
                done_last = True
                obs_last  = queue[k][3]
                break

        return obs_t, action_t, nstep_return, obs_last, done_last

    def add(self, obs, action, reward, next_obs, done):
        """
        Add one transition. N-step logic handles queuing and flushing.
        Call this once per environment step.
        """
        self._nstep_queue.append((obs, action, reward, next_obs, bool(done)))

        # When queue is full, store oldest n-step transition
        if len(self._nstep_queue) >= self.n_steps:
            trans = self._compute_nstep_return(self._nstep_queue)
            self._store(trans)
            self._nstep_queue.pop(0)

        # On episode end, flush remaining transitions with truncated returns
        if done:
            while len(self._nstep_queue) > 0:
                trans = self._compute_nstep_return(self._nstep_queue)
                self._store(trans)
                self._nstep_queue.pop(0)

    def _store(self, transition):
        """Store a completed n-step transition in SumTree with max priority."""
        p = self.max_prio ** self.alpha
        self.tree.add(p, transition)

    def sample(self, batch_size, beta):
        """
        Sample a batch with prioritized weights.
        Returns: (indices, weights, obs, actions, nstep_returns, next_obs, dones)
        """
        indices, weights, batch = [], [], []
        segment  = self.tree.total / batch_size
        min_prob = self.eps / self.tree.total
        max_w    = (min_prob * self.tree.n_entries) ** (-beta)

        for i in range(batch_size):
            data = None
            for _ in range(20):   # retry on empty leaf
                s = random.uniform(segment * i, segment * (i + 1))
                idx, prio, data = self.tree.get(s)
                if data is not None:
                    break
            if data is None:
                fallback = (self.tree.write_ptr - 1) % self.tree.capacity
                idx   = fallback + self.tree.capacity - 1
                prio  = float(self.tree.tree[idx])
                data  = self.tree.data[fallback]

            prob   = (prio + self.eps) / self.tree.total
            weight = (prob * self.tree.n_entries) ** (-beta) / max_w
            indices.append(idx)
            weights.append(weight)
            batch.append(data)

        obs_b    = np.array([d[0] for d in batch], dtype=np.float32)
        act_b    = np.array([d[1] for d in batch], dtype=np.int64)
        ret_b    = np.array([d[2] for d in batch], dtype=np.float32)
        nobs_b   = np.array([d[3] for d in batch], dtype=np.float32)
        done_b   = np.array([d[4] for d in batch], dtype=np.float32)

        return (indices, np.array(weights, dtype=np.float32),
                obs_b, act_b, ret_b, nobs_b, done_b)

    def update_priorities(self, indices, td_errors):
        """Update priorities after gradient step."""
        for idx, td in zip(indices, td_errors):
            p = (abs(float(td)) + self.eps) ** self.alpha
            self.tree.update(idx, p)
            if p > self.max_prio:
                self.max_prio = p

    def __len__(self):
        return self.tree.n_entries


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_dndqn_networks(obs_dim=OBS_DIM, n_actions=N_ACTIONS,
                        net_arch=NET_ARCH, device='cpu'):
    """
    Build online and target DnDQNs with identical initial weights.
    Returns (online_net, target_net).
    Target net is set to eval() — never backpropagated through.
    """
    online = DnDQN(obs_dim, n_actions, net_arch).to(device)
    target = DnDQN(obs_dim, n_actions, net_arch).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    return online, target


def load_dndqn_checkpoint(path, device='cpu'):
    """
    Load a Dn-DQN checkpoint and reconstruct the network.
    Returns (online_net, checkpoint_dict).
    """
    ckpt   = torch.load(path, map_location=device)
    online = DnDQN(
        obs_dim   = ckpt.get('obs_dim',   OBS_DIM),
        n_actions  = ckpt.get('n_actions',  N_ACTIONS),
        net_arch   = ckpt.get('net_arch',   NET_ARCH),
    ).to(device)
    online.load_state_dict(ckpt['q_net_state'])
    online.eval()
    return online, ckpt


def load_standard_checkpoint(path, device='cpu'):
    """
    Load a standard QNetwork checkpoint (for ablation config 6b).
    Returns (qnet, checkpoint_dict).
    """
    ckpt  = torch.load(path, map_location=device)
    q_net = QNetwork(
        obs_dim   = ckpt.get('obs_dim',  OBS_DIM),
        n_actions  = ckpt.get('n_actions', N_ACTIONS),
        net_arch   = ckpt.get('net_arch',  NET_ARCH),
    ).to(device)
    q_net.load_state_dict(ckpt['q_net_state'])
    q_net.eval()
    return q_net, ckpt


def polyak_update(online_net, target_net, tau=0.01):
    """
    Soft update target network: θ_target ← τ θ_online + (1-τ) θ_target
    tau=0.01 is the standard value from DDPG/DQN literature.
    """
    for p_online, p_target in zip(online_net.parameters(),
                                   target_net.parameters()):
        p_target.data.copy_(tau * p_online.data + (1 - tau) * p_target.data)


def epsilon_schedule(step, eps_start, eps_end=0.05, eps_fraction=0.30,
                     total_steps=800_000):
    """Linear epsilon decay from eps_start to eps_end over eps_fraction of total_steps."""
    decay_steps = int(eps_fraction * total_steps)
    if step >= decay_steps:
        return eps_end
    return eps_start + (step / decay_steps) * (eps_end - eps_start)


def beta_schedule(step, beta_start=0.4, beta_end=1.0, total_steps=800_000):
    """Linear beta annealing from beta_start to beta_end over total_steps."""
    return beta_start + min(1.0, step / total_steps) * (beta_end - beta_start)
