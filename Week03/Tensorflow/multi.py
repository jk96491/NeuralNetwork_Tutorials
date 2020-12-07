import random
import numpy as np
from Week03.Tensorflow.Model import MyModel

min_len = 50
max_len = 10000
normalize = 1

use_fit = False

train_data_len = 9000

subject_score_list = []
final_score_list = []

batch_size = 10
epochs = 5

for i in range(max_len):
    kor = random.randrange(10, 100) * normalize
    math = random.randrange(10, 100) * normalize
    eng = random.randrange(10, 100) * normalize
    subject_score_list.append([kor, math, eng])

    final_score = (kor + math + eng) / 3.0

    final_score_list.append([final_score * normalize])


X_data = np.asarray(subject_score_list)
Y_data = np.asarray(final_score_list)

x_train = X_data[:train_data_len]
y_train = Y_data[:train_data_len]

x_test = X_data[train_data_len:]
y_test = Y_data[train_data_len:]

model = MyModel()

model.train(x_train, y_train, batch_size, epochs, use_fit)
model.eval(x_test, y_test)
