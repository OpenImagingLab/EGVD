import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
        )
        self.downsample = nn.AvgPool3d((1,2,2))
        
    def forward(self, x):
        x = self.conv(x)
        return self.downsample(x)


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

class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_fusion = nn.Sequential(
            nn.Conv3d(channels*3, channels, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels, channels, 3, 1, 1),
        )
        
        self.weight_pred = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels*3, channels*3, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels*3, 3, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x_cond, x_evs, x_ref):
        cat_features = torch.cat([x_cond, x_evs, x_ref], dim=1)
        weights = self.weight_pred(cat_features)
        
        w1, w2, w3 = weights[:, [0], ...], weights[:, [1], ...], weights[:, [2], ...]
        weighted_sum = w1*x_cond + w2*x_evs + w3*x_ref
        
        fusion = self.conv_fusion(cat_features)
        return weighted_sum + fusion

class LatentEventFusion(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=128):
        super().__init__()
        
        self.conv_cond = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(base_channels, base_channels, 3, 1, 1),
        )
        
        # self.conv_evs = nn.Sequential(
        #     nn.Conv3d(in_channels * 2, base_channels, 3, (1,2,2), 1),
        #     nn.LeakyReLU(0.1),
        #     nn.Conv3d(base_channels, base_channels, 3, (1,2,2), 1),
        #     nn.LeakyReLU(0.1),
        #     nn.Conv3d(base_channels, base_channels, 3, (1,2,2), 1),
        #     nn.LeakyReLU(0.1)
        # )
        
        self.conv_evs = nn.Sequential(
            DownBlock(in_channels * 2, base_channels//2),
            DownBlock(base_channels//2, base_channels),
            DownBlock(base_channels, base_channels)
        )
        
        self.conv_ref = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(base_channels, base_channels, 3, 1, 1),
        )
        
        # self.feature_fusion = FeatureFusion(base_channels)
        
        self.blocks = nn.ModuleList([
            ResBlock3D(base_channels) for _ in range(5)
        ])
        
        self.temporal_attention = nn.ModuleList([
            TemporalAttention(base_channels) for _ in range(3)
        ])
        
        self.spatial_attention = nn.ModuleList([
            SpatialAttention(base_channels) for _ in range(3)
        ])
        
        self.output = nn.Conv3d(base_channels, out_channels, 3, 1, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

                
    def forward(self, conditions_latent, evs_latents, conditions_latent_ref):
        B, T, C, H, W = conditions_latent.shape
        device = conditions_latent.device
        
        if not (conditions_latent.shape == conditions_latent_ref.shape):
            raise ValueError("Input shapes must match")
            
        conditions_latent = conditions_latent.permute(0, 2, 1, 3, 4)
        evs_latents = evs_latents.permute(0, 2, 1, 3, 4)
        conditions_latent_ref = conditions_latent_ref.permute(0, 2, 1, 3, 4)
        
        x_cond = self.conv_cond(conditions_latent)
        x_evs = self.conv_evs(evs_latents)
        x_ref = self.conv_ref(conditions_latent_ref)
        
        # x = self.feature_fusion(x_cond, x_evs, x_ref)
        x = x_cond + x_ref + x_evs
        # x = x + x_evs
        
        # 残差学习模块如下
        x_res = x
        for i, res_3d_block in enumerate(self.blocks):
            x = res_3d_block(x)
            if 1 <= i <= 3:
                att_idx = i - 1
                x = self.temporal_attention[att_idx](x) * x
                x = self.spatial_attention[att_idx](x) * x
        x_out = x + x_res
        evs_out = self.output(x_out) 
        
        # 时序权重生成模块
        w_cond = torch.arange(T, device=device).float() / (T - 1)
        w_ref = 1 - torch.arange(T, device=device).float() / (T - 1)
        w_evs = torch.ones(T, device=device)        
        w_evs[0], w_evs[-1] = 0, 0 
        w_cond = w_cond.view(1, 1, T, 1, 1).expand(B, C, T, H, W)
        w_ref = w_ref.view(1, 1, T, 1, 1).expand(B, C, T, H, W)
        w_evs = w_evs.view(1, 1, T, 1, 1).expand(B, C, T, H, W)
        alpha = (1.0 / (w_cond  + w_ref)).to(device) #+ w_evs

        # 残差权重融合模块
        output = alpha * (w_evs * evs_out + w_cond * conditions_latent + w_ref * conditions_latent_ref)
        

        # if random.random() < 0.001:
        #     # 打印张量的最大值和最小值
        #     print("Max/Min values of tensors:")
        #     print(f"Output: max={torch.max(output)}, min={torch.min(output)}")
        #     print(f"EVS Output: max={torch.max(evs_out)}, min={torch.min(evs_out)}")
        #     print(f"Conditions Latent: max={torch.max(conditions_latent)}, min={torch.min(conditions_latent)}")
        #     print(f"Conditions Latent Ref: max={torch.max(conditions_latent_ref)}, min={torch.min(conditions_latent_ref)}")

        return output.permute(0, 2, 1, 3, 4).contiguous()


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
#     evs_latents = torch.randn(batch_size, time_steps, channels * 2, height*8, width*8).to(device)  # evs_latents 的通道数是 conditions_latent 的 4 倍
#     conditions_latent_ref = torch.randn(batch_size, time_steps, channels, height, width).to(device)

#     # 打印初始显存状态
#     print(f"Initial memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
#     print(f"Initial memory reserved: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")

#     # Instantiate the model
#     model = LatentEventFusion(in_channels=channels, out_channels=channels).to(device)

#     # Forward pass
#     output = model(conditions_latent, evs_latents, conditions_latent_ref)
    
#     # 打印显存状态
#     print(f"Memory allocated after forward pass: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
#     print(f"Memory reserved after forward pass: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")

#     print("Output shape:", output.shape)
