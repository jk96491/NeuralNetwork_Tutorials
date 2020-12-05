import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Utils import Tensorboard_Writer
from Utils import SuffleData

training_len = 1900

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(27, 128),
                                    nn.ReLU(),
                                    nn.Dropout())
        self.layer2 = nn.Sequential(nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Dropout())
        self.layer3 = nn.Sequential(nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Dropout())
        self.layer4 = nn.Sequential(nn.Linear(32, 7))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


tensorboard = Tensorboard_Writer("Test3")

model = Classifier()

xy = np.loadtxt('faults.csv', delimiter=',') #1941
x_data = xy[:, :-7]
y_data = xy[:, -7:]

x_data = torch.FloatTensor(x_data)
y_data = torch.FloatTensor(y_data)

x_data, y_data = SuffleData(x_data, y_data, len(xy))

x_train = x_data[:training_len]
y_train = y_data[:training_len]

x_test = x_data[training_len:]
y_test = y_data[training_len:]
test_len = len(x_test)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

nb_epochs = 50000


for epoch in range(nb_epochs + 1):
    x_train, y_train = SuffleData(x_train, y_train, 5)

    hypothesis = model(x_train)
    output = torch.max(y_train, 1)[1]
    cost = F.cross_entropy(hypothesis, output)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 20 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f} '.format(
            epoch, nb_epochs, cost.item(),
        ))

        tensorboard.WriteScalar("loss", epoch, cost)

    # 100번 마다 Test
    if epoch % 1000 == 0 and epoch is not 0:
        correct_count = 0
        for i in range(test_len):
            result = torch.argmax(F.softmax(model(x_test[i])))
            answer = torch.argmax(y_test[i])

            if result.item() == answer.item():
                correct_count += 1

        print('accuracy : {0}'.format(correct_count / test_len))


tensorboard.close()