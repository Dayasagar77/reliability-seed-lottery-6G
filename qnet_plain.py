#!/usr/bin/env python3
"""qnet_plain.py — non-dueling Q-network for the architecture ablation
(Standard DQN, DDQN+PER). Matches ablation_standard_dqn_full.pt exactly:
Sequential(Linear6->128, ReLU, 128->128, ReLU, 128->64, ReLU, 64->6)."""
import torch, torch.nn as nn

class PlainQNet(nn.Module):
    def __init__(self, obs_dim=6, n_actions=6, arch=(128, 128, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, arch[0]), nn.ReLU(),
            nn.Linear(arch[0], arch[1]), nn.ReLU(),
            nn.Linear(arch[1], arch[2]), nn.ReLU(),
            nn.Linear(arch[2], n_actions),
        )
    def forward(self, x):
        return self.net(x)

def load_plain(path):
    ck = torch.load(path, map_location='cpu')
    net = PlainQNet(ck.get('obs_dim', 6), ck.get('n_actions', 6),
                    tuple(ck.get('net_arch', (128, 128, 64))))
    net.load_state_dict(ck['q_net_state']); net.eval()
    return net, ck
