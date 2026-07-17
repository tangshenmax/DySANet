import torch
import torch.nn as nn
from _9CBAM import ChannelAttention,SpaceAttention
from torchvision.ops import DeformConv2d


class DeformableCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 18, 3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
   
    def forward(self, x_skip, x_up):
        offset = self.offset_conv(x_skip)
        return self.bn(self.deform_conv(x_up, offset))

    
class FRM(nn.Module):
    def __init__(self, in_c, compress_rate_c):
        super().__init__()
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1),
            nn.BatchNorm2d(in_c),
        )

        self.conv_up = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1),
            nn.BatchNorm2d(in_c),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(2*in_c, in_c, kernel_size=1),
            nn.BatchNorm2d(in_c),
        )

        self.space_gate = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c,1,kernel_size=1,bias=False),
            nn.Sigmoid()
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, in_c//compress_rate_c, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c//compress_rate_c, in_c, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.deformablecnn = DeformableCNN(in_c,in_c)

    def forward(self, x_skip, x_up):
        x_skip_conv = self.conv_skip(x_skip)
        x_up_conv = self.conv_up(x_up)
        x_add = x_skip_conv + x_up_conv
        space_weight = self.space_gate(x_add)
        channel_weight = self.channel_gate(x_add)
        x_skip_weighted = x_skip * space_weight * channel_weight
        x_up_refined = self.deformablecnn(x_skip_weighted, x_up)
        return torch.cat([x_skip_weighted, x_up_refined],dim=1)