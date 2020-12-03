import torch.nn as nn


class PreConv(nn.Module):

    def __init__(self, in_c, out_c, k, s, p):
        """Full Pre-Activation Convolutional Layer"""
        super(PreConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_c),
            nn.LeakyReLU(),
            nn.Conv3d(in_c, out_c, k, s, p, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_c, out_c):
        """Full Pre-activation Residual Block"""
        super(ResidualBlock, self).__init__()
        self.conv_1 = PreConv(in_c, out_c, 3, 1, 1)
        self.conv_2 = PreConv(out_c, out_c, 3, 1, 1)
        self.linear = nn.Conv3d(in_c, out_c, 1, 1)

    def forward(self, x):
        skip = self.linear(x)
        return skip + self.conv_2(self.conv_1(x))


class DoubleConv(nn.Module):

    def __init__(self, in_c, out_c):
        """(Conv ==> BatchNorm ==> ReLU)^2"""
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.Conv3d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm3d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class ResidualUNet(nn.Module):

    def __init__(self):
        """Residual U-Net with Full pre-activation residual encoder and standard double convolution
        decoder. Addition skip connections rather than concatenation to reduce memory requirements."""
        super(ResidualUNet, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)

        # residual encoder
        self.conv1 = nn.Conv3d(1, 16, 1)
        self.res_1 = ResidualBlock(16, 32)
        self.res_2 = ResidualBlock(32, 64)
        self.res_3 = ResidualBlock(64, 128)
        self.res_4 = ResidualBlock(128, 256)
        self.res_5 = ResidualBlock(256, 256)

        # standard double convolution decoder
        self.up_1 = nn.ConvTranspose3d(256, 256, 2, 2)
        self.dec_1 = DoubleConv(256, 128)
        self.up_2 = nn.ConvTranspose3d(128, 128, 2, 2)
        self.dec_2 = DoubleConv(128, 64)
        self.up_3 = nn.ConvTranspose3d(64, 64, 2, 2)
        self.dec_3 = DoubleConv(64, 32)
        self.up_4 = nn.ConvTranspose3d(32, 32, 2, 2)
        self.dec_4 = DoubleConv(32, 16)
        self.out_conv = nn.Conv3d(16, 1, 1)

    def forward(self, x):
        # encoder
        x1 = self.res_1(self.conv1(x))          # 32 x 32
        x2 = self.res_2(self.pool(x1))          # 16 x 64
        x3 = self.res_3(self.pool(x2))          # 8 x 128
        x4 = self.res_4(self.pool(x3))          # 4 x 256
        x5 = self.res_5(self.pool(x4))          # 2 x 256

        # decoder
        x6 = self.dec_1(self.up_1(x5) + x4)     # 4 x 128
        x7 = self.dec_2(self.up_2(x6) + x3)     # 8 x 64
        x8 = self.dec_3(self.up_3(x7) + x2)     # 16 x 32
        x9 = self.dec_4(self.up_4(x8) + x1)     # 32 x 16

        return self.out_conv(x9)
