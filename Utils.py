from torch.utils.tensorboard import SummaryWriter


class Tensorboard_Writer:
    def __init__(self, name):
        self.writer = SummaryWriter("./runs/" + name)

    def WriteScalar(self, key, iter, value):
        self.writer.add_scalar(key, value, iter)

    def close(self):
        self.writer.close()

