import torch
from torch import optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[5], [8], [11]])

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.Adam([w, b], lr=0.001)

epoch = 10000

for i in range(epoch):
    predict = x_train * w + b
    loss = torch.mean((predict - y_train) ** 2)

    optimizer.zero_grad()       # gradient 초기화
    loss.backward()             # gradient 산출
    optimizer.step()            # 역전파 실행

    print('{0}번째 => loss : {1:5.5f},  w : {2},  b : {3}'.format(i + 1, loss.data, w.data, b.data))
