import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, optimizers
import random
from Utils import SuffleData

input_size = 3
output_size = 1
hidden_size = 20
hidden_dim = 128

min_len = 100
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

x_train = tf.convert_to_tensor(subject_score_list)
y_train = tf.convert_to_tensor(final_score_list)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        weight_init = tf.keras.initializers.RandomNormal()
        self.model = tf.keras.Sequential()
        self.model.add(Dense(1, use_bias=True, kernel_initializer=weight_init))

    def call(self, x):
        x = self.model(x)
        return x


model = MyModel()

print(model.summary())

opti = optimizers.Adam(learning_rate=0.001)

nb_epochs = 15000
for epoch in range(nb_epochs + 1):
    rand = random.randrange(min_len, max_len)
   # x_train, y_train = SuffleData(x_train, y_train, rand)

    with tf.GradientTape() as Tape:
        prediction = model(x_train)
        cost = tf.reduce_mean(tf.square(prediction - y_train))

    gradients = Tape.gradient(cost, model.trainable_variables)
    opti.apply_gradients(zip(gradients, model.trainable_variables))

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost))
    #print(model.layers[0].weight.data[0])
