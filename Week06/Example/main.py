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
        self.layer1 = nn.Sequential(nn.Linear(27, 256),
                                    nn.ReLU(),
                                    nn.Dropout())
        self.layer2 = nn.Sequential(nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Dropout())
        self.layer3 = nn.Sequential(nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Dropout())
        self.layer4 = nn.Sequential(nn.Linear(64, 7))

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

data_count = len(xy)
x_data, y_data = SuffleData(x_data, y_data, data_count)

x_train = x_data[:training_len]
y_train = y_data[:training_len]

x_test = x_data[training_len:]
y_test = y_data[training_len:]
test_len = len(x_test)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

nb_epochs = 50


for epoch in range(nb_epochs + 1):
    x_train, y_train = SuffleData(x_train, y_train, data_count)

    cost = 0
    for i in range(1000):
        x_train, y_train = SuffleData(x_train, y_train, 20)
        hypothesis = model(x_train)
        output = torch.max(y_train, 1)[1]
        cost = F.cross_entropy(hypothesis, output)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    correct_count = 0
    for i in range(test_len):
        result = torch.argmax(F.softmax(model(x_test[i])))
        answer = torch.argmax(y_test[i])

        if result.item() == answer.item():
            correct_count += 1
    accuracy = (correct_count / test_len) * 100
    print('Epoch {:3d}/{} Cost: {:.3f} accuracy : {:.2f}%'.format( epoch + 1, nb_epochs, cost.item(), accuracy))

    tensorboard.WriteScalar("loss", epoch, cost)

tensorboard.close()