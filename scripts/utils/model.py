'''
SAC (Soft Actor Critic)
- continous action
'''
import numpy as np
import torch
import torch.nn as nn

## Parameters
N_INPUTS1   = 15 #9
N_INPUTS2   = 20 #9
N_OUTPUT    = 4

class InteractionNetwork(nn.Module):
    def __init__(self, obs_dim, state_dim, hidden_dim):
        super().__init__()
        hidden_hidden_dim = int(hidden_dim / 2)
        self.edge_mlp = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, state, obs):
        """
        state: [B, 9]
        obs: [B, D, K]
        mask: [B, K]
        """
        obs = obs.permute(0, 2, 1)  # [B, K, D]

        B, K, D = obs.shape
        Dg = state.shape[-1]

        mask = (obs.abs().sum(dim=2) != 0)  # 실제 장애물 여부
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [batch, 1]

        # Pairwise combinations (i, j)
        obs_i = obs.unsqueeze(2).repeat(1, 1, K, 1)   # [B, K, K, D]
        obs_j = obs.unsqueeze(1).repeat(1, K, 1, 1)   # [B, K, K, D]
        edge_input = torch.cat([obs_i, obs_j], dim=-1)  # [B, K, K, 2D]

        # Edge MLP
        edge_output = self.edge_mlp(edge_input)  # [B, K, K, H]

        # Edge mask (mask_i & mask_j)
        mask_i = mask.unsqueeze(2)  # [B, K, 1]
        mask_j = mask.unsqueeze(1)  # [B, 1, K]
        edge_mask = mask_i * mask_j  # [B, K, K]
        edge_output = edge_output * edge_mask.unsqueeze(-1)  # [B, K, K, H]

        # Aggregate incoming messages to each node i
        valid_counts = edge_mask.sum(dim=2, keepdim=True).clamp(min=1e-6)
        aggregated = edge_output.sum(dim=2) / valid_counts  # [B, K+1, H]
        aggregated = self.norm(aggregated)

        # State1 broadcast
        state1_expanded = state.unsqueeze(1).repeat(1, K, 1)  # [B, K, S]
        
        # Node update
        node_input = torch.cat([obs, aggregated, state1_expanded], dim=-1)
        node_output = self.node_mlp(node_input)  # [B, K, H]

        # Pooling
        mean_obs_masked = node_output.masked_fill(~mask.unsqueeze(-1), 0.0)  # 패딩된 부분을 0으로 만듦
        mean_pool = mean_obs_masked.sum(dim=1) / counts  # [batch, dim]

        max_obs_masked = node_output.masked_fill(~mask.unsqueeze(-1), -1e9)  # 패딩된 부분을 -1e9으로 만듦
        max_pool = max_obs_masked.max(dim=1)[0]  # [batch, dim]

        return torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2H]

class ActorNetwork(nn.Module):
    def __init__(self, n_state:int = N_INPUTS1, n_obs:int = N_INPUTS2, n_action:int = N_OUTPUT):
        super(ActorNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_state, 512),
            nn.ReLU(),
        )

        self.self_attention = nn.ModuleList([
            InteractionNetwork(state_dim=n_state, obs_dim=n_obs, hidden_dim=512)
            for _ in range(2)
            ])

        self.mu = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + 512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_action),
            ) for _ in range(2)
        ])

        self.std = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + 512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_action),
                nn.Softplus(),
            ) for _ in range(2)
        ])

    def forward(self, state, obs, mode):
        mode = mode.long().view(-1, 1)
        mode_onehot = torch.zeros(state.size(0), 2, device=state.device)
        mode_onehot.scatter_(1, mode, 1.0)
        # State branch
        _state = self.layer(state)  # [batch, 1024]


        mode_idx = mode.item()
        # Obs self attention
        _obs = self.self_attention[mode_idx](state, obs)  # [batch, 1024]
        _state = torch.cat([_state, _obs], dim=1)
        mu = self.mu[mode_idx](_state)

        return torch.tanh(mu).squeeze().cpu().numpy()