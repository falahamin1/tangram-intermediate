"""
Graph Neural Network Actor-Critic for the Intermediate Tangram (4 pieces).

4 pieces × 5 constraints = 20 nodes in a block-diagonal 20×20 adjacency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = torch.bmm(adj, x)
        return F.relu(self.projection(support))


class GNNActorCritic(nn.Module):
    """
    Inputs to forward():
        h_rep : [Batch, 4, 5, 3]  — 4 pieces, 5 constraints, 3 params
        adj   : [Batch, 4, 5, 5]  — per-piece constraint adjacency
    """

    def __init__(self, node_dim: int = 3, hidden_dim: int = 128, num_actions: int = 16):
        super().__init__()
        self.num_pieces      = 4
        self.constraints_per = 5
        self.total_nodes     = self.num_pieces * self.constraints_per  # 20

        self.gcn1 = GCNLayer(node_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_rep, adj):
        batch_size = h_rep.shape[0]
        N = self.total_nodes
        C = self.constraints_per

        x = h_rep.view(batch_size, N, 3)   # [B, 20, 3]

        big_adj = torch.zeros(batch_size, N, N, device=h_rep.device)
        for i in range(self.num_pieces):
            big_adj[:, i*C:(i+1)*C, i*C:(i+1)*C] = adj[:, i, :, :]
        big_adj += torch.eye(N, device=h_rep.device).unsqueeze(0)

        h = self.gcn1(x, big_adj)
        h = self.gcn2(h, big_adj)

        global_pool = torch.max(h, dim=1)[0]
        latent      = self.rho(global_pool)

        return self.actor(latent), self.critic(latent)
