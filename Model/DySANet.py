import torch
import torch.nn as nn
import pandas as pd
from fvcore.nn import FlopCountAnalysis
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer, SwinTransformerStage, PatchMerging
from timm.layers.patch_embed import PatchEmbed 
from _25Fusion import Fusion3
from torchinfo import summary
from _1CPM import CPM
#from _10Down_Up import DownSample,UpSample
from _30JDSA import FRM


class BilinearUpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2,  align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=self.align_corners)
        return self.conv(x)


class Stem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        return self.stem(x)


class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )    
    def forward(self,x):
        return self.cnn(x)
 

class CU(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)



class DFS(nn.Module):
    def __init__(self, channel, channel_reduction, local_space=True):
        super().__init__()
        mid_channel = channel//4

        self.local_space = local_space
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        

        self.space_att_conv7 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.space_att_conv3 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
  
        self.channel_att_mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, channel, kernel_size=1, bias=False),
        )

        self.local_conv = nn.Sequential(
            nn.Conv2d(channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        self.global_conv = nn.Sequential(
            nn.Conv2d(channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )


        self.fuse = Fusion3(channel, channel_reduction)
    
    def forward(self, x_local, x_global):

        x_local_conv = self.local_conv(x_local)
        x_global_conv = self.global_conv(x_global)

        x_global_conv_maxpool = torch.max(x_global_conv, dim=1, keepdim=True)[0] 
        x_global_conv_avgpool = torch.mean(x_global_conv, dim=1, keepdim=True)
        x_global_conv_max_avg = torch.cat([x_global_conv_maxpool, x_global_conv_avgpool],dim=1) # (B,2,H,W)
        space_att7 = self.space_att_conv7(x_global_conv_max_avg)

        if self.local_space:
            x_local_conv_maxpool = torch.max(x_local_conv, dim=1, keepdim=True)[0] 
            x_local_conv_avgpool = torch.mean(x_local_conv, dim=1, keepdim=True)
            x_local_conv_max_avg = torch.cat([x_local_conv_maxpool, x_local_conv_avgpool],dim=1) # (B,2,H,W)
            space_att3 = self.space_att_conv3(x_local_conv_max_avg)         

            x_local_guided = x_local *  space_att7
            x_global_guided = x_global *  space_att3

            x_fuse = self.fuse(x_local_guided, x_global_guided)
            return x_fuse
        else:
            x_local_conv_maxpool = self.global_max_pool(x_local_conv)
            x_local_conv_avgpool = self.global_avg_pool(x_local_conv)
            channel_att = torch.sigmoid(self.channel_att_mlp(x_local_conv_maxpool) + self.channel_att_mlp(x_local_conv_avgpool))

            x_local_guided = x_local *  space_att7
            x_global_guided = x_global *  channel_att
            x_fuse = self.fuse(x_local_guided, x_global_guided)
            return x_fuse


class ChannelLast(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()


class ChannelFirst(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()            


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem(3,64)
        self.cnn1 = CU(64,64)      
        self.cnn2 = CU(64,128)
        self.cnn3 = CU(128,256)
        self.cnn4 = CU(256,512)
        


        
        self.swin1 = nn.Sequential(
            PatchEmbed(img_size=[224,224], patch_size=4, in_chans=3,embed_dim=64,flatten=False),
            ChannelLast(),
            SwinTransformerStage(dim=64, out_dim=64, input_resolution=[56, 56], depth=2, downsample=False, num_heads=2, window_size=7,
                                mlp_ratio=4, qkv_bias=True, proj_drop=0, attn_drop=0, drop_path=0.1, norm_layer=nn.LayerNorm),
            ChannelFirst(),
        )
        self.swin2 = nn.Sequential(
            ChannelLast(),
            PatchMerging(dim=64), # 64->128
            SwinTransformerStage(dim=128, out_dim=128, input_resolution=[28, 28], depth=2, downsample=False, num_heads=4, window_size=7,
                                mlp_ratio=4, qkv_bias=True, proj_drop=0, attn_drop=0, drop_path=0.1, norm_layer=nn.LayerNorm),
            ChannelFirst(),
        )
        self.swin3 = nn.Sequential(
            ChannelLast(), 
            PatchMerging(dim=128), # 128->256
            SwinTransformerStage(dim=256, out_dim=256, input_resolution=[14, 14], depth=2, downsample=False, num_heads=8, window_size=7,
                                mlp_ratio=4, qkv_bias=True, proj_drop=0, attn_drop=0, drop_path=0.1, norm_layer=nn.LayerNorm),
            ChannelFirst(),
        )
        self.swin4 = nn.Sequential(
            ChannelLast(),
            PatchMerging(dim=256), # 256->512
            SwinTransformerStage(dim=512, out_dim=512, input_resolution=[7, 7], depth=1, downsample=False, num_heads=16, window_size=7,
                                mlp_ratio=4, qkv_bias=True, proj_drop=0, attn_drop=0, drop_path=0.1, norm_layer=nn.LayerNorm),
            ChannelFirst(),
        )
        


        
        self.cgaf1 = DFS(64, 2, local_space = True)
        self.cgaf2 = DFS(128, 4, local_space = True)
        self.cgaf3 = DFS(256, 8, local_space = True)
        self.cgaf4 = DFS(512, 16, local_space = True)
        
        self.de4 = CPM(in_channels=512, out_channels=256, rate_c=8)
        self.de3 = CPM(in_channels=512, out_channels=256, rate_c=8)
        self.de2 = CPM(in_channels=256, out_channels=128, rate_c=4) 
        self.de1 = CPM(in_channels=128, out_channels=64, rate_c=2)
     
        

        self.sk3 = FRM(in_c=256, compress_rate_c=16)
        self.sk2 = FRM(in_c=128, compress_rate_c=8)
        self.sk1 = FRM(in_c=64, compress_rate_c=4)


        self.up3 = BilinearUpsampleConv(in_channels=256, out_channels=256)
        self.up2 = BilinearUpsampleConv(in_channels=256, out_channels=128)
        self.up1 = BilinearUpsampleConv(in_channels=128, out_channels=64)
        self.up0 = nn.Sequential(
            BilinearUpsampleConv(in_channels=64, out_channels=64),
            BilinearUpsampleConv(in_channels=64, out_channels=64)
        )

        self.dropout3 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.1)
        self.dropout1 = nn.Dropout2d(0.1)

        self.head4 = nn.Conv2d(256, 1, kernel_size=1, bias = False)
        self.head3 = nn.Conv2d(256, 1, kernel_size=1, bias = False)
        self.head2 = nn.Conv2d(128, 1, kernel_size=1, bias = False)
        self.head1 = nn.Conv2d(64, 1, kernel_size=1, bias = False)

    def forward(self, x): 
        xstem = self.stem(x) 
        xcnn1 = self.cnn1(xstem)
        xswin1 = self.swin1(x)
        xen1 = self.cgaf1(xcnn1, xswin1)


        xcnn2 = self.cnn2(xen1)
        xswin2 = self.swin2(xen1)
        xen2 = self.cgaf2(xcnn2, xswin2)


        xcnn3 = self.cnn3(xen2)
        xswin3 = self.swin3(xen2)
        xen3 = self.cgaf3(xcnn3, xswin3)


        xcnn4 = self.cnn4(xen3)
        xswin4 = self.swin4(xen3)
        xen4 = self.cgaf4(xcnn4, xswin4)

        xde4 = self.de4(xen4)


        x3_skip = self.sk3(xen3, self.up3(xde4))
        xde3 = self.de3(x3_skip)
        x3_de_out = self.dropout3(xde3)

        x2_skip = self.sk2(xen2, self.up2(x3_de_out))
        xde2 = self.de2(x2_skip)
        x2_de_out = self.dropout2(xde2)

        x1_skip = self.sk1(xen1, self.up1(x2_de_out))
        xde1 = self.de1(x1_skip)
        x1_de_out = self.dropout1(xde1)

        #out4 = self.head4(xde4)
        #out3 = self.head3(xde3)
        #out2 = self.head2(xde2)
        out1 = self.head1(self.up0(x1_de_out))
        return out1   


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def profile_by_module(model: nn.Module, x: torch.Tensor, targets):
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, x)
        by_module = flops.by_module()  

    named_modules = dict(model.named_modules())

    rows = []
    for name in targets:
        if name not in named_modules:
            print(f"Not Found: {name}")
            continue

        module = named_modules[name]
        params = count_params(module)
        module_flops = by_module.get(name, 0)

        rows.append({
            "module": name,
            "params(M)": params / 1e6,
            "flops(G)": module_flops / 1e9,
        })

    df = pd.DataFrame(rows)
    return df, flops

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet().to(device).eval()
    x = torch.randn(1, 3, 224, 224).to(device)

    targets = [
        "cnn1", "cnn2", "cnn3", "cnn4",
        "swin1", "swin2", "swin3", "swin4",
        "cgaf1", "cgaf2", "cgaf3", "cgaf4",
        "de4", "de3", "de2", "de1",
        "sk3", "sk2", "sk1",
        "up3", "up2", "up1", "up0",
        "final"
    ]

    df, flops = profile_by_module(model, x, targets)

    print(df.to_string(index=False))

    total_params = count_params(model) / 1e6
    total_flops = flops.total() / 1e9
    print(f"Total Params = {total_params:.2f} M")
    print(f"Total FLOPs  = {total_flops:.2f} G")

    df.to_csv("module_profile.csv", index=False)
    print("\nSaved: module_profile.csv")