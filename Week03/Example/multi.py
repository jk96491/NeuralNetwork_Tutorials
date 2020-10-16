import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 1)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        return x

#0.25    0.75     0.5
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70],
                             [80, 75, 97],
                             [45, 64, 84],
                             [21, 85, 24],
                             [32, 15, 84],
                             [86, 21, 57]])
y_train = torch.FloatTensor([[115.75], [135.75], [135], [147.5], [102.75],
                             [124.75], [101.25], [81], [61.25], [65.75]])

model = Model()
optimizer = optim.Adam(model.parameters(), lr=0.01)

nb_epochs = 50000
for epoch in range(nb_epochs + 1):
    prediction = model(x_train)

    cost = torch.mean((prediction - y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Cost: {:.6f} : weight {}'.format(epoch, nb_epochs, cost.item(), model.fc1.weight))