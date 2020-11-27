import torch.nn as nn
from Advanced.Resnet.Utils import conv3x3
from Advanced.Resnet.Utils import conv1x1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # 1 X 1
        self.Layer1 = nn.Sequential(conv1x1(inplanes, planes),
                                    self.bn1,
                                    self.relu)
        # 3 X 3
        self.Layer2 = nn.Sequential(conv3x3(planes, planes, stride),
                                    self.bn2,
                                    self.relu)
        # 1 X 1
        self.Layer3 = nn.Sequential(conv1x1(planes, planes * self.expansion),
                                    self.bn3)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.Layer1(x)
        out = self.Layer2(out)
        out = self.Layer3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out