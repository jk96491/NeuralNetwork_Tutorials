import tensorflow as tf
from tensorflow.keras import optimizers as optim
import random
import numpy as np
from Utils import SuffleData
from Week03.Tensorflow.Model import MyModel

input_size = 3
output_size = 1
hidden_size = 20
hidden_dim = 128

min_len = 50
max_len = 1000
normalize = 1


subject_score_list = []
final_score_list = []

for i in range(max_len):
    kor = random.randrange(10, 100) * normalize
    math = random.randrange(10, 100) * normalize
    eng = random.randrange(10, 100) * normalize
    subject_score_list.append([kor, math, eng])

    final_score = (kor + math + eng) / 3.0

    final_score_list.append([final_score * normalize])

x_train = np.asarray(subject_score_list)
y_train = np.asarray(final_score_list)

model = MyModel()

optimizers = optim.Adam(learning_rate=0.001)

nb_epochs = 15000
for epoch in range(nb_epochs + 1):
    rand = random.randrange(min_len, max_len)
    x_train, y_train = SuffleData(x_train, y_train, rand)

    with tf.GradientTape() as Tape:
        prediction = model(x_train)
        cost = tf.reduce_mean(tf.square(prediction - y_train))

    gradients = Tape.gradient(cost, model.trainable_variables)
    optimizers.apply_gradients(zip(gradients, model.trainable_variables))

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost))
