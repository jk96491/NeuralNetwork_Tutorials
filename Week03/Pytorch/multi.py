import torch
from Week03.Pytorch.Model import MyModel
import torch.optim as optim
import random
from Utils import SuffleData

use_cuda = True

input_size = 3
output_size = 1
hidden_size = 20
hidden_dim = 128

min_len = 1000
max_len = 100000

normalize = 0.01

subject_score_list = []
final_score_list = []

for i in range(max_len):
    kor = random.randrange(10, 100) * normalize
    math = random.randrange(10, 100) * normalize
    eng = random.randrange(10, 100) * normalize
    subject_score_list.append([kor, math, eng])

    final_score = (kor + math + eng) / 3.0

    final_score_list.append([final_score * normalize])

if use_cuda:
    x_train = torch.FloatTensor(subject_score_list).cuda()
    y_train = torch.FloatTensor(final_score_list).cuda()
    model = MyModel(hidden_size, input_size, output_size, hidden_dim).cuda()
else:
    x_train = torch.FloatTensor(subject_score_list)
    y_train = torch.FloatTensor(final_score_list)
    model = MyModel(hidden_size, input_size, output_size, hidden_dim)


optimizer = optim.Adam(model.parameters(), lr=0.001)

nb_epochs = 15000
for epoch in range(nb_epochs + 1):
    rand = random.randrange(min_len, max_len)
    x_train, y_train = SuffleData(x_train, y_train, rand)

    prediction = model(x_train)
    cost = torch.mean((prediction - y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
    #print(model.layers[0].weight.data[0])

    if epoch % 100 == 0:
        torch.save(model.state_dict(), 'Train_model/model.th')