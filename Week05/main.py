import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Utils import Tensorboard_Writer

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 1)
        self.Relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.Relu(self.linear(x))
        x = self.Relu(self.linear2(x))
        x = self.Relu(self.linear3(x))

        x = self.sigmoid(self.linear4(x))

        return x


tensorboard = Tensorboard_Writer("Test3")

model = BinaryClassifier()

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

optimizer = optim.Adam(model.parameters(), lr=0.001)

nb_epochs = 10000
for epoch in range(nb_epochs + 1):
    hypothesis = model(x_train)
    #cost = F.mse_loss(hypothesis, y_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 20 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))

        tensorboard.WriteScalar("loss", epoch, cost)
        tensorboard.WriteScalar("accuracy", epoch, accuracy)

tensorboard.close()