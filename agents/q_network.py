import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_actions)
        )

    def forward(self, x):

        return self.model(x)