import torch.nn as nn


class RL_MODEL(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, state):
        actions = self.linear(state)

        return actions