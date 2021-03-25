import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Utils import Tensorboard_Writer
from Utils import SuffleData

batch_size = 1024
training_len = 1800
use_normalize = True
drop_out = 0.1
exclude_cols = [11, 12, 19, 20, 26]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(27, 128),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out))
        self.layer2 = nn.Sequential(nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out))
        self.layer3 = nn.Sequential(nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out))
        self.layer4 = nn.Sequential(nn.Linear(32, 7))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


def normalize(data, exclude_cols_):
    for i in range(len(data[0])):
        if exclude_cols_.__contains__(i):
            continue

        cur_data = data[:, i]
        min, max = cur_data.min() , cur_data.max()
        normalize_data = (cur_data - min) / (max - min)
        data[:, i] = normalize_data
        data[:, i] = normalize_data


tensorboard = Tensorboard_Writer("Test3")

model = Classifier().to(device)

xy = np.loadtxt('faults.csv', delimiter=',') #1941
x_data = xy[:, :-7]
y_data = xy[:, -7:]

if use_normalize:
    normalize(x_data, exclude_cols)

x_data = torch.FloatTensor(x_data).to(device)
y_data = torch.FloatTensor(y_data).to(device)

x_data, y_data = SuffleData(x_data, y_data, len(xy))

x_train = x_data[:training_len]
y_train = y_data[:training_len]

x_test = x_data[training_len:]
y_test = y_data[training_len:]
test_len = len(x_test)

optimizer = optim.Adam(model.parameters(), lr=0.0005)

nb_epochs = 10000


for epoch in range(nb_epochs + 1):
    x_train, y_train = SuffleData(x_train, y_train, batch_size)

    hypothesis = model(x_train)
    output = torch.max(y_train, 1)[1]
    cost = F.cross_entropy(hypothesis, output)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
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