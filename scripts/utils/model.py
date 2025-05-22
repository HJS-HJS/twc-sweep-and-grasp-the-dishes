'''
SAC (Soft Actor Critic)
- continous action
'''
import os
import sys
import numpy as np
import torch
import torch.nn as nn

## Parameters
FILE_NAME = None
N_INPUTS1   = 21 #9
N_INPUTS2   = 19 #9
N_OUTPUT    = 4

def mask_attention_output(attn_output, mask):
    """
    attn_output: [batch, k, hidden_dim]
    mask: [batch, k] (True: 패딩된 부분, False: 실제 장애물)
    
    패딩된 부분을 강제로 0으로 변환
    """
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)  # [batch, k] → [batch, k, 1]
        attn_output = attn_output.masked_fill(mask_expanded, 0.0)
    return attn_output

class SelfAttentionObstacle(nn.Module):
    def __init__(self, obs_dim=10, hidden_dim=1024):
        super(SelfAttentionObstacle, self).__init__()
        hidden_dim = int(hidden_dim / 2)
        self.mean_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.max_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs, mask=None):
        obs = obs.permute(0, 2, 1)  # [batch, k, 10]

        valid_mask = (obs.abs().sum(dim=2) != 0)  # 실제 장애물 여부
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [batch, 1]

        mean_obs = self.mean_layer(obs)
        max_obs = self.max_layer(obs)

        # 패딩은 무시하고 평균 계산
        mean_obs_masked = mean_obs.masked_fill(~valid_mask.unsqueeze(-1), 0.0)  # 패딩된 부분을 0으로 만듦
        mean_obs = mean_obs_masked.sum(dim=1) / valid_counts  # [batch, dim]

        max_obs_masked = max_obs.masked_fill(~valid_mask.unsqueeze(-1), -1e9)  # 패딩된 부분을 -1e9으로 만듦
        max_obs = max_obs_masked.max(dim=1)[0] / valid_counts  # [batch, dim]

        return torch.cat([mean_obs, max_obs], dim=1)


class ActorNetwork(nn.Module):
    def __init__(self, n_state:int = N_INPUTS1, n_obs:int = N_INPUTS2, n_action:int = N_OUTPUT):
        super(ActorNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_state, 256),
            nn.ReLU(),
        )

        self.self_attention = SelfAttentionObstacle(obs_dim=n_obs, hidden_dim=1024)

        self.mu = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024 + 256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_action),
            ) for _ in range(2)
        ])

        self.std = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024 + 256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_action),
                nn.Softplus(),
            ) for _ in range(2)
        ])

    def forward(self, state, obs, mode):
        mode = mode.long().view(-1, 1)

        # State branch
        _state = self.layer(state)  # [batch, 1024]

        # Mask
        # Mask if obstacle not exist in each k
        valid_mask = (obs.abs().sum(dim=1) != 0)  # [batch, k]
        mask = (obs.abs().sum(dim=1) == 0)
        # Mask if obstacle not existd in every k

        # Obs self attention
        _obs = self.self_attention(obs, mask)  # [batch, 1024]
        _state = torch.cat([_state, _obs], dim=1)

        mode_idx = mode.item()
        mu = self.mu[mode_idx](_state)
        std = self.std[mode_idx](_state)

        # sample
        distribution = torch.distributions.Normal(mu, std)
        u = distribution.rsample()

        # Enforce action bounds [-1., 1.]
        action = torch.tanh(u)

        # return action, logprob
        return action.squeeze().cpu().numpy()