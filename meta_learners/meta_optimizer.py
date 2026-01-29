# meta_learner/meta_optimizer.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaOptimizer(nn.Module):
    """
    Learns a learning-rate scaling policy.
    """

    def __init__(self, input_dim=3, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return torch.sigmoid(self.fc2(x))
