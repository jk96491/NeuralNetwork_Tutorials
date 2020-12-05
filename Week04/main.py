import torch
import torch.nn as nn
import torch.optim as optim
from Utils import Tensorboard_Writer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = Model()

optimizer = optim.Adam(model.parameters(), lr=0.0000452)

tensorboard = Tensorboard_Writer("Test_tensorboard_Adam")

nb_epochs = 50000
for epoch in range(nb_epochs + 1):
    prediction = model(x_train)

    cost = torch.mean((prediction - y_train)**2)
    cost = torch.nn.CrossEntropyLoss(prediction, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

    tensorboard.WriteScalar("loss", epoch, cost)

tensorboard.close()