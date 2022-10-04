import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# 卷积层，卷积+批归一+激活函数
class Conv(nn.Module):
    def __init__(self, input, output, kernelSize, stride=1, padding='same'):
        super(Conv, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(input, output, kernelSize, stride, padding, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.m(x)


class Residual(nn.Module):
    def __init__(self, inputC):
        super(Residual, self).__init__()
        tempC = inputC // 2
        self.m = nn.Sequential(
            Conv(input, tempC, 1, 1, 0),
            Conv(tempC, inputC, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.m(x)


class convSet(nn.Module):
    def __init__(self, inputC, outputC, midC):
        super(convSet, self).__init__()
        self.m = nn.Sequential(
            Conv(inputC, outputC, 1),
            Conv(inputC, midC, 3),
            Conv(midC, outputC, 1),
            Conv(outputC, midC, 3),
            Conv(midC, outputC, 1)
        )

    def forward(self, x):
        return self.m(x)


class LastLayer(nn.Module):
    def __init__(self, inputC, outputC, anchor=None):
        super(LastLayer, self).__init__()
        self.grid = None
        self.anchor = np.array(anchor)
        self.anchorScaled = []
        self.stride = 1
        self.shape = None
        self.m = nn.Sequential(
            Conv(inputC, outputC * 2, 3),
            nn.Conv2d(inputC * 2, outputC, 1)
        )

    def forward(self, x):
        o = self.m(x)
        if self.grid is None:
            self._createGrid(o.shape)
        return o

    def _createGrid(self, shape):
        b, c, h, w = shape
        self.shape = (h, w)
        self.stride = CONST.inputShape[0] / h
        self.anchorScaled = torch.tensor(self.anchor / self.stride, device=CONST.device)
        grid = torch.ones((b, len(self.anchor), h, w, 4), device=CONST.device)
        gridY, gridX = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grid[..., 0] *= gridX.to(CONST.device).unsqueeze(0)
        grid[..., 1] *= gridY.to(CONST.device).unsqueeze(0)
        grid[..., 2] *= self.anchorScaled[:, 0].view(1, len(self.anchor), 1, 1)
        grid[..., 3] *= self.anchorScaled[:, 1].view(1, len(self.anchor), 1, 1)
        self.grid = grid


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        # 定义darknet53的层数
        self.layoutNumber = [1, 2, 8, 8, 4]
        self.layerA = nn.Sequential(
            Conv(3, 32, 3, 1, 1),
            self.MultiResidual(32, 64, 1),
            self.MultiResidual(64, 128, 2),
            self.MultiResidual(128, 256, 8)
        )
        self.layerB = self.MultiResidual(256, 512, 8)
        self.layerC = self.MultiResidual(512, 1024, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
            


    # 多层的残差网络
    def MultiResidual(self, inputC, outputC, count):
        t = [Conv(inputC, outputC, 3, 2, 1) if i == 0 else Residual(outputC) for i in range(count + 1)]
        return nn.Sequential(*t)
