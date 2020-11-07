import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Utils import Tensorboard_Writer
from Utils import SuffleData

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(27, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 7)
        self.Relu = nn.ReLU()

    def forward(self, x):
        x = self.Relu(self.linear(x))
        x = self.Relu(self.linear2(x))
        x = self.Relu(self.linear3(x))

        x = self.linear4(x)

        return x


tensorboard = Tensorboard_Writer("Test3")

model = BinaryClassifier()

xy = np.loadtxt('faults.csv', delimiter=',')
x_data = xy[:, :-7]
y_data = xy[:, -7:]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

optimizer = optim.Adam(model.parameters(), lr=0.001)

nb_epochs = 10000
for epoch in range(nb_epochs + 1):
    x_train, y_train = SuffleData(x_train, y_train, 5)

    hypothesis = model(x_train)
    output = torch.max(y_train, 1)[1]
    cost = F.cross_entropy(hypothesis, output)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 1 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f} '.format(
            epoch, nb_epochs, cost.item(),
        ))

        tensorboard.WriteScalar("loss", epoch, cost)

tensorboard.close()