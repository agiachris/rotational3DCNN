import torch
import torch.nn as nn


class UNet3dBaseline(nn.Module):

    def __init__(self, model_config):
        super(UNet3dBaseline, self).__init__()

        # create model with layers
        self.model_config = model_config

    def forward(self, x):
        # forward pass
        return x
