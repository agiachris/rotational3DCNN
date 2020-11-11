import torch
import torch.nn as nn
import torch.nn.functional as F


def relu_max(x):
    return F.max_pool3d(F.relu(x), 2, 2)


class UNet3dBaseline(nn.Module):

    def __init__(self):
        """Toy model for basic testing
        """
        super(UNet3dBaseline, self).__init__()

        # three conv layers
        self.conv_1 = nn.Conv3d(1, 16, 3, 1, 1)
        self.conv_2 = nn.Conv3d(16, 32, 3, 1, 1)
        self.conv_3 = nn.Conv3d(32, 64, 3, 1, 1)

        # middle
        self.conv_4 = nn.Conv3d(64, 64, 3, 1, 1)

        # three deconv layers
        self.deconv_1 = nn.ConvTranspose3d(64, 64, 2, 2)
        self.deconv_2 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.deconv_3 = nn.ConvTranspose3d(32, 16, 2, 2)
        self.deconv_4 = nn.ConvTranspose3d(16, 1, 2, 2)

    def forward(self, x):
        # encoder
        x1 = relu_max(self.conv_1(x))           # 16
        x2 = relu_max(self.conv_2(x1))          # 8
        x3 = relu_max(self.conv_3(x2))          # 4
        x4 = relu_max(self.conv_4(x3))          # 2

        # decoder
        x5 = F.relu(self.deconv_1(x4)) + x3     # 4
        x6 = F.relu(self.deconv_2(x5)) + x2     # 8
        x7 = F.relu(self.deconv_3(x6)) + x1     # 16
        x8 = torch.sigmoid(self.deconv_4(x7))   # 32

        return x8
