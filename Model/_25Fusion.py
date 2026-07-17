import torch
import torch.nn as nn


class channel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp_channel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, in_channels*2, kernel_size=1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_local, x_global):
        x_add = x_local + x_global
        pooled = self.mlp_channel(self.maxpool(x_add)) + self.mlp_channel(self.avgpool(x_add))

        half_channels = pooled.size(1) // 2  
        vec_a = pooled[:, :half_channels, :, :]  
        vec_b = pooled[:, half_channels:, :, :]  

        weights = torch.stack([vec_a, vec_b], dim=1)  # [B, 2, C, 1, 1]
        weights = self.softmax(weights)
        weight_a = weights[:, 0, :, :, :]  # [B, C, 1, 1]
        weight_b = weights[:, 1, :, :, :]  # [B, C, 1, 1]

        fused_a = x_local * weight_a
        fused_b = x_global * weight_b

        fused = fused_a + fused_b
        return fused
    

class space(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp_space = nn.Sequential(
            nn.ReLU(inplace=True),                   
            nn.Conv2d(in_channels, 1, kernel_size=7,padding=3),
            nn.Sigmoid()
        )

    def forward(self, x_local, x_global):
        x_add = x_local + x_global
        pooled = self.mlp_space(x_add)
        fused = pooled*x_local + (1-pooled)*x_global
        return fused
    

class Fusion1(nn.Module):
    def __init__(self, channels):
        super(Fusion1, self).__init__()
        self.channels = channels
        self.gate_generator = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),  # 7x7 conv
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        feat_add = feat1 + feat2  
        max_map = torch.max(feat_add, dim=1, keepdim=True)[0]   # (B, 1, H, W)
        avg_map = torch.mean(feat_add, dim=1, keepdim=True)     # (B, 1, H, W)
        pooled = torch.cat([max_map, avg_map], dim=1)           # (B, 2, H, W)
        gate = self.gate_generator(pooled)                      # (B, 1, H, W)
        # ------------------------------------------------------------------------
        fused_feat = gate * feat1 + (1 - gate) * feat2  # (B, C, H, W)

        return fused_feat
    

class Fusion2(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, bias=False),
        )

    def forward(self,x_local,x_global):
        x_add = x_local + x_global
        max_pool = self.maxpool(x_add)
        avg_pool = self.avgpool(x_add)
        weight = torch.sigmoid(self.mlp(max_pool) + self.mlp(avg_pool))
        fuse = weight*x_local + (1-weight)*x_global
        return fuse
    

class Fusion3(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        mid = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_local, x_global):
        """
        x_local, x_global: [B, C, H, W]
        返回 fused: [B, C, H, W]
        """
        B, C, _, _ = x_local.shape
        v_local = self.avgpool(x_local)
        v_global = self.avgpool(x_global)

        e_local = self.mlp(v_local)   # [B, C, 1, 1]
        e_global = self.mlp(v_global) # [B, C, 1, 1]
        stacked = torch.stack([e_local, e_global], dim=1)
        weights = self.softmax(stacked)  # [B,2,C,1,1]

        w_local = weights[:, 0, :, :, :]  # [B, C, 1, 1]
        w_global = weights[:, 1, :, :, :] # [B, C, 1, 1]

        out = x_local * w_local + x_global * w_global
        return out
    

class Fusion4(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        mid = max(1, in_channels // reduction)
        self.mlp_local = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False)
        )
        self.mlp_global = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_local, x_global):
        v_local = self.avgpool(x_local)
        v_global = self.avgpool(x_global)
        
        e_local = self.mlp_local(v_local)
        e_global = self.mlp_global(v_global)
        
        stacked = torch.stack([e_local, e_global], dim=1)
        weights = self.softmax(stacked)
        
        w_local = weights[:, 0, :, :, :]
        w_global = weights[:, 1, :, :, :]
        
        return x_local * w_local + x_global * w_global