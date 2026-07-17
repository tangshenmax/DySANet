import torch
import torch.nn as nn
from _9CBAM import SpaceAttention,ChannelAttention


class CPM(nn.Module):
    def __init__(self, in_channels, out_channels, rate_c):
        super().__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )


        self.cpm = channel(out_channels, rate_c)
        

        self.out_conv = nn.Conv2d(out_channels, out_channels, 1, bias = False)
        self.out_bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        feat1 = self.conv3(x)  
        feat2 = self.conv5(x)  
        feat3 = self.conv7(x)  
        feat4 = self.res(x)  

        """
        feat1 = self.space_attention(feat1)
        feat2 = self.space_attention(feat2)
        feat3 = self.space_attention(feat3)
        feat4 = self.space_attention(feat4)
        """
        add_sk = self.cpm(feat1,feat2,feat3)
        add = add_sk + feat4
        output = self.act(self.out_bn(self.out_conv(add)))
        return output


class channel(nn.Module):
    def __init__(self, in_channels, rate_c):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp_channel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // rate_c, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // rate_c, in_channels * 3, kernel_size=1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        x_add = x1 + x2 + x3
        pooled = self.mlp_channel(self.maxpool(x_add)) + self.mlp_channel(self.avgpool(x_add))
        # pooled: [B, 3*C, 1, 1]
        B, totalC, _, _ = pooled.shape

        
        base = totalC // 3
        rem = totalC - base * 3
        sizes = [base, base, base + rem]  

        vecs = torch.split(pooled, sizes, dim=1)  
        
        weights = torch.stack(vecs, dim=1)
        
        weights = self.softmax(weights)


        weight_a = weights[:, 0, :, :, :]  
        weight_b = weights[:, 1, :, :, :]  
        weight_c = weights[:, 2, :, :, :]  

        fused_a = x1 * weight_a
        fused_b = x2 * weight_b
        fused_c = x3 * weight_c

        fused = fused_a + fused_b + fused_c
        return fused


class inceptionnext(nn.Module):
    def __init__(self, in_channels, out_channels, band_kernel_size):
        super().__init__()
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.dwconv3x3 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=3, padding=1, groups=in_channels//4)
        self.dwconv1_k = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(1, band_kernel_size), padding=(0,band_kernel_size//2), groups=in_channels//4)
        self.dwconvk_1 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=in_channels//4)
        self.identity = nn.Identity()

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ReLU()

    def forward(self, x):

        x_res = self.res_conv(x)
        chunk1, chunk2, chunk3, chunk4 = torch.chunk(x, chunks=4, dim=1)
        x_3   = self.dwconv3x3(chunk1)  
        x_1_k = self.dwconv1_k(chunk2)  
        x_k_1 = self.dwconvk_1(chunk3)
        x_identity = self.identity(chunk4)     

        cat = torch.cat([x_3, x_1_k, x_k_1, x_identity], dim=1) 

        output = self.out_conv(cat)
        out = output + x_res         
        out = self.act(out)     
        return out


class UnetstyleEnDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.endecoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.sp = SpaceAttention()
        self.ch = ChannelAttention()
    def forward(self,x):
        return self.sp(self.ch(self.endecoder(x)))
