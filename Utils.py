from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

class Tensorboard_Writer:
    def __init__(self, name):
        self.writer = SummaryWriter("./runs/" + name)

    def WriteScalar(self, key, iter, value):
        self.writer.add_scalar(key, value, iter)

    def close(self):
        self.writer.close()


def SuffleData(x_train, y_train, count):
    s = np.arange(x_train.shape[0])
    np.random.shuffle(s)

    x_train = x_train[s]
    y_train = y_train[s]

    x_train = x_train[:count]
    y_train = y_train[:count]

    return  x_train, y_train

