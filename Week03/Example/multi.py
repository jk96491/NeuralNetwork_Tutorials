import torch
from Week03.Example.Model import MyModel
import torch.optim as optim
import random
import numpy as np

print(torch.cuda.is_available())

input_size = 3
output_size = 1
hidden_size = 10
hidden_dim = 128

subject_score_list = []
final_score_list = []

for i in range(100000):
    kor = random.randrange(60, 100)
    math = random.randrange(60, 100)
    eng = random.randrange(60, 100)
    subject_score_list.append([kor, math, eng])

    final_score = kor * 0.25 + math * 0.75 + eng * 0.5

    final_score_list.append([final_score])

x_train = torch.FloatTensor(subject_score_list)
y_train = torch.FloatTensor(final_score_list)

model = MyModel(hidden_size, input_size, output_size, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.000452)

nb_epochs = 5000
for epoch in range(nb_epochs + 1):

    s = np.arange(x_train.shape[0])
    np.random.shuffle(s)
    x_train = x_train[s]
    y_train = y_train[s]

    prediction = model(x_train)
    cost = torch.mean((prediction - y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
 #   print('Epoch {:4d}/{} Cost: {:.6f} weight : {}'.format(epoch, nb_epochs, cost.item(), model.layers[]))