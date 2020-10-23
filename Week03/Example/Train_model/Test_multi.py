import torch
from Week03.Example.Model import MyModel

input_size = 3
output_size = 1
hidden_size = 0
hidden_dim = 1

model = MyModel(hidden_size, input_size, output_size, hidden_dim)

model.load_state_dict(torch.load('model.th'))

while True:
    kor = float(input('국어 점수를 입력 하시오=>'))
    math = float(input('수학 점수를 입력 하시오=>'))
    eng = float(input('영어 점수를 입력 하시오=>'))

    inputData = [kor, math, eng]

    predict = model(torch.FloatTensor(inputData)).item()
    answer = (kor + math + eng) / 3.0

    answer = round(answer, 2)
    predict = round(predict, 2)

    loss = abs(answer - predict)

    print('정답 : {0:0.5f}, 예측 : {1:0.5f}, 오차 : {2:0.5f}\n'.format(answer, predict, loss))