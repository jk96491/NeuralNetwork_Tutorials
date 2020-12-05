import torch.nn as nn


class RL_MODEL(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, output_size)

    def forward(self, state):
        x = self.linear(state)
        actions = self.linear2(x)

        return actions