import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden_size, input_size, ounput_size, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        cur_dim = input_size
        for i in range(hidden_size):
            self.layers.append(nn.Linear(cur_dim, hidden_dim))
            cur_dim = hidden_dim

        self.layers.append(nn.Linear(cur_dim, ounput_size))

        self.Relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.Relu(layer(x))
        out = self.layers[-1](x)
        return out
