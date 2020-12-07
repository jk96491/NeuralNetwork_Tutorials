import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from Utils import SuffleData


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        weight_init = tf.keras.initializers.RandomNormal()
        self.model = tf.keras.Sequential(Dense(1, use_bias=True, kernel_initializer=weight_init, input_shape=(3,)))

        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=self.optimizer, loss=self.mse_error)
        self.build(input_shape=(None, 3))

        print(self.model.summary())

    def call(self, inputs, training=None, mask=None):
        predict = self.model(inputs)
        return predict

    def train(self, x_train, y_train, batch_size, epoch, use_fit):
        if use_fit:
            self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch)
        else:
            for ep in range(epoch):
                for i in range(int(len(x_train) / batch_size)):
                    x, y = SuffleData(x_train, y_train, batch_size)
                    with tf.GradientTape() as Tape:
                        predict = self.call(x)
                        loss = self.mse_error(predict, y)

                    gradients = Tape.gradient(loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                    if (i + 1) % 100 == 0:
                        print('Epoch {:4d}/{} step{:4d} loss: {:.6f}'.format(ep + 1, epoch, i + 1, loss.numpy()))

    def eval(self, x_test, y_test):
        self.model.evaluate(x_test, y_test)

    def mse_error(self, predict, answer):
        loss = tf.reduce_mean(tf.square(predict - answer))
        return loss