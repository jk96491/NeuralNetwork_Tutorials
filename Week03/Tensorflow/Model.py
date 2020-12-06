import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        weight_init = tf.keras.initializers.RandomNormal()
        self.model = tf.keras.Sequential()
        self.model.add(Dense(3, use_bias=True, kernel_initializer=weight_init))

    def call(self, x):
        x = self.model(x)
        return x