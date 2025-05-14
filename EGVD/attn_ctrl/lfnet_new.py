import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels, channels, 3, 1, 1)
        )
        
    def forward(self, x):
        return F.leaky_relu(x + self.conv(x), 0.1)

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels//8, 1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1)
        y = self.conv(y)
        y = y.unsqueeze(-1).unsqueeze(-1)
        return y

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels//8, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.conv(x)


class LatentFedNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=128):
        super().__init__()
        
        self.conv_cond = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(base_channels, base_channels, 3, 1, 1),
        )

        self.blocks = nn.ModuleList([
            ResBlock3D(base_channels) for _ in range(3)
        ])
        
        self.temporal_attention = nn.ModuleList([
            TemporalAttention(base_channels) for _ in range(1)
        ])
        
        self.spatial_attention = nn.ModuleList([
            SpatialAttention(base_channels) for _ in range(1)
        ])
        
        self.output = nn.Conv3d(base_channels, out_channels, 3, 1, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

                
    def forward(self, conditions_latent):
        conditions_latent = conditions_latent.permute(0, 2, 1, 3, 4) 
        x_cond = self.conv_cond(conditions_latent)      
        # x = x + x_evs
        x = x_cond
        for i, block in enumerate(self.blocks):
            x = block(x)
            if 1 <= i <= 1:
                att_idx = i - 1
                x = self.temporal_attention[att_idx](x) * x
                x = self.spatial_attention[att_idx](x) * x
        x_out = self.output(x) + conditions_latent
        print("use_lfnet!")

        return x_out.permute(0, 2, 1, 3, 4).contiguous()


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # 测试显存占用
# if __name__ == "__main__":
#     # 设置设备为 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 定义输入大小
#     batch_size = 1
#     time_steps = 9
#     channels = 4
#     height = 64
#     width = 64

#     conditions_latent = torch.randn(batch_size, time_steps, channels, height, width).to(device)

#     # 打印初始显存状态
#     print(f"Initial memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
#     print(f"Initial memory reserved: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")

#     # Instantiate the model
#     model = LatentFedNet(in_channels=channels, out_channels=channels).to(device)

#     # Forward pass
#     output = model(conditions_latent)
    
#     # 打印显存状态
#     print(f"Memory allocated after forward pass: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
#     print(f"Memory reserved after forward pass: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")

#     print("Output shape:", output.shape)
