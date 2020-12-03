import torch
import torch.nn as nn
import torch.nn.functional as F


#input: 1x3x128x128x128 tensor, declare UNet with input channel of 3
#convolution
def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


#transpose convolution
def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)

#pooling
def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


class UNet3dBaseline(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet3dBaseline, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU()
        
        # Down sampling
        self.pool = max_pooling_3d()
        self.down_1 = conv_block_3d(self.in_dim, self.num_filters, activation)
        self.down_2 = conv_block_3d(self.num_filters, self.num_filters * 2, activation)
        self.down_3 = conv_block_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.down_4 = conv_block_3d(self.num_filters * 4, self.num_filters * 8, activation)
        
        # Bridge
        self.bridge = conv_block_3d(self.num_filters * 8, self.num_filters * 16, activation)
        
        # Up sampling
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_3d(self.num_filters * 3, self.num_filters * 1, activation)
        
        # Output
        self.out = nn.Conv3d(self.num_filters, out_dim, kernel_size=1)

    def forward(self, x):
        # Down
        down_1 = self.down_1(x) # shape [1, 4, 128, 128, 128]
        pool_1 = self.pool(down_1) # shape [1, 4, 64, 64, 64]
        
        down_2 = self.down_2(pool_1) # shape [1, 8, 64, 64, 64]
        pool_2 = self.pool(down_2) # shape [1, 8, 32, 32, 32]
        
        down_3 = self.down_3(pool_2) # shape [1, 16, 32, 32, 32]
        pool_3 = self.pool(down_3) # shape [1, 16, 16, 16, 16]
        
        down_4 = self.down_4(pool_3) # shape [1, 32, 16, 16, 16]
        pool_4 = self.pool(down_4) # shape [1, 32, 8, 8, 8]
                
        # Bridge
        bridge = self.bridge(pool_4) # shape [1, 128, 4, 4, 4]
        
        # Up 
        trans_2 = self.trans_2(bridge) # shape [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4], dim=1) # shape [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2) # shape [1, 32, 16, 16, 16]
        
        trans_3 = self.trans_3(up_2) # shape [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1) # shape [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # shape [1, 16, 32, 32, 32]
        
        trans_4 = self.trans_4(up_3) # shape [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2], dim=1) # shape [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4) # shape [1, 8, 64, 64, 64]
        
        trans_5 = self.trans_5(up_4) # shape [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1], dim=1) # shape [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5) # shape [1, 4, 128, 128, 128]
        
        # Output
        out = self.out(up_5) # shape [1, 3, 128, 128, 128]
        return out
