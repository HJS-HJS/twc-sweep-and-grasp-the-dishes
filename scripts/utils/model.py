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
N_INPUTS1   = 11
N_INPUTS2   = 5
N_OUTPUT    = 4

class ActorNetwork(nn.Module):
    def __init__(self, device, n_state:int = N_INPUTS1, n_obs:int = N_INPUTS2, n_action:int = N_OUTPUT):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.layer = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.obs1_layer = nn.Sequential(
            nn.Conv1d(n_obs,64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.t_net1 = nn.Sequential(
            nn.Conv1d(64,64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.t_net2 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256, 64*64),
            nn.ReLU(),
        )
            
        self.obs2_layer = nn.Sequential(
            nn.Conv1d(64,128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.obs3_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.last_layer = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
        )

        self.mu = nn.Sequential(
            nn.Linear(256,n_action),
        )
        self.std = nn.Sequential(
            nn.Linear(256,n_action),
            nn.Softplus(),
        )

    def forward(self, state):
        x = torch.tensor(state[0], dtype=torch.float32, device=self.device).unsqueeze(0)
        obs   = torch.tensor(state[1].T, dtype=torch.float32, device=self.device).unsqueeze(0)

        x = self.layer(x)
        obs = self.obs1_layer(obs)
        _t = self.t_net1(obs)
        _t = torch.max(_t, 2, keepdim=True)[0]
        _t = self.t_net2(_t)
        _t = _t.view(-1, 64, 64)
        obs = torch.bmm(_t, obs)
        obs = self.obs2_layer(obs)
        obs = torch.max(obs, 2, keepdim=True)[0]
        obs = self.obs3_layer(obs)

        x=self.last_layer(torch.cat([x, obs], dim=1))

        mu = self.mu(x)
        std = self.std(x)

        # sample
        distribution = torch.distributions.Normal(mu, std)
        u = distribution.rsample()

        # Enforce action bounds [-1., 1.]
        action = torch.tanh(u)

        # return action, logprob
        return action[0].tolist()