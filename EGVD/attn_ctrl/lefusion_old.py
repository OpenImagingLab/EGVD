import torch
import torch.nn as nn
import torch.nn.functional as F



class LatentEventFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LatentEventFusion, self).__init__()

        # 分别为 conditions_latent、evs_latents 和 conditions_latent_ref 设置卷积层
        self.conv_conditions_latent = nn.Conv3d(in_channels, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        # self.conv_evs_latents = nn.Conv3d(in_channels*4, 128, kernel_size=(3, 3, 3), stride=1, padding=1)  # 处理 evs_latents
        
        self.conv_evs_latents = nn.Sequential(
            nn.Conv3d(in_channels * 4, 128, kernel_size=(3, 3, 3), stride=(1,2,2), padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1,2,2), padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1,2,2), padding=1),
        )
        
        
        self.conv_conditions_latent_ref = nn.Conv3d(in_channels, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        
        self.conv3d_input = nn.Conv3d(128*3, 128, kernel_size=(3, 3, 3), stride=1, padding=1)

        self.residual_block1 = self._make_residual_block(128)
        self.channel_attention1 = self._make_channel_attention(128)
        self.residual_block2 = self._make_residual_block(128)
        self.channel_attention2 = self._make_channel_attention(128)
        self.residual_block3 = self._make_residual_block(128)
        self.channel_attention3 = self._make_channel_attention(128)
        self.residual_block4 = self._make_residual_block(128)
        self.channel_attention4 = self._make_channel_attention(128)
        self.residual_block5 = self._make_residual_block(128)
        self.channel_attention5 = self._make_channel_attention(128)
        self.residual_block6 = self._make_residual_block(128)
        self.channel_attention6 = self._make_channel_attention(128)
        self.residual_block7 = self._make_residual_block(128)
        

        self.output_conv = nn.Conv3d(128, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)

    def _make_residual_block(self, in_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
        )

    def _make_channel_attention(self, in_channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # 全局池化
            nn.Conv3d(in_channels, 32, kernel_size=1, stride=1, padding=0),  # 降维
            nn.LeakyReLU(0.1),
            nn.Conv3d(32, in_channels, kernel_size=1, stride=1, padding=0),  # 恢复维度
            nn.Sigmoid()
        )

    def forward(self, conditions_latent, evs_latents, conditions_latent_ref):

        B, T, C, H, W = conditions_latent.shape
        device = conditions_latent.device
        w_cond = torch.arange(T, device=device).float() / (T - 1)
        w_ref = 1 - torch.arange(T, device=device).float() / (T - 1)
        w_evs = torch.ones(T, device=device)
        w_evs[0] = 0
        # w_ref[0], w_cond[-1] = 2, 2
        w_cond = w_cond.view(1, T, 1, 1, 1).expand(B, T, C, H, W)
        w_ref = w_ref.view(1, T, 1, 1, 1).expand(B, T, C, H, W)
        w_evs = w_evs.view(1, T, 1, 1, 1).expand(evs_latents.shape)
        conditions_latent = w_cond * conditions_latent
        evs_latents = w_evs * evs_latents
        conditions_latent_ref = w_ref * conditions_latent_ref

        conditions_latent = conditions_latent.permute(0, 2, 1, 3, 4).contiguous()
        evs_latents = evs_latents.permute(0, 2, 1, 3, 4).contiguous()
        conditions_latent_ref = conditions_latent_ref.permute(0, 2, 1, 3, 4).contiguous()

        x_conditions_latent = self.conv_conditions_latent(conditions_latent)  # [b, 128, t, h, w]
        x_evs_latents = self.conv_evs_latents(evs_latents)  # [b, 128, t, h, w]
        x_conditions_latent_ref = self.conv_conditions_latent_ref(conditions_latent_ref)  # [b, 128, t, h, w]

        # average_feature = x_conditions_latent + x_evs_latents + x_conditions_latent_ref
        concat_feature = torch.cat((x_conditions_latent, x_evs_latents, x_conditions_latent_ref), dim=1)  # 拼接

        x = self.conv3d_input(concat_feature)  # [b, 128, t, h, w]

        x = self.residual_block1(x)+ x
        x = self.channel_attention1(x) * x 
        x = self.residual_block2(x)+ x
        x = self.channel_attention2(x) * x 
        x = self.residual_block3(x)+ x
        x = self.channel_attention3(x) * x 
        x = self.residual_block4(x)+ x
        x = self.channel_attention4(x) * x 
        x = self.residual_block5(x)+ x
        x = self.channel_attention5(x) * x 
        x = self.residual_block6(x)+ x
        x = self.channel_attention6(x) * x 
        x = self.residual_block7(x)+ x

        
        x = self.output_conv(x) + conditions_latent + conditions_latent_ref
        return x.permute(0, 2, 1, 3, 4).contiguous()  # 恢复到原始形状


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
#     channels = 3
#     height = 64
#     width = 64

#     conditions_latent = torch.randn(batch_size, time_steps, channels, height, width).to(device)
#     evs_latents = torch.randn(batch_size, time_steps, channels * 4, height*8, width*8).to(device)  # evs_latents 的通道数是 conditions_latent 的 4 倍
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



# # Test the TemporalAttentionBlock
# if __name__ == "__main__":
#     batch_size = 1
#     time_steps = 9
#     channels = 3
#     height = 64
#     width = 64

#     conditions_latent = torch.randn(batch_size, time_steps, channels, height, width)
#     evs_latents = torch.randn(batch_size, time_steps, channels*4, height, width)  # evs_latents 的通道数是 conditions_latent 的 4 倍
#     conditions_latent_ref = torch.randn(batch_size, time_steps, channels, height, width)

#     # Instantiate the model
#     model = TemporalAttentionBlock(in_channels=channels, out_channels=channels)

#     # Forward pass
#     output = model(conditions_latent, evs_latents, conditions_latent_ref)
#     print("output",output.shape)

#     print("Output shape:", output.shape)
