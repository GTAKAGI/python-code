import torch
import torch.nn as nn
##revised Gated-SNU##
class Unet():
    def _init_(self, in_channels, out_channels,kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv_1 = nn.Conv2d(self.in_channels,self.out_channels,self.kernel_size)
        self.pool_1 =