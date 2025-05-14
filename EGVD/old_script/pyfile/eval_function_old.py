import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import pyiqa
# LPIPS 计算类
class lpips:
    def __init__(self, loss_dict):
        self.func = pyiqa.create_metric('lpips', as_loss=loss_dict.get('as_loss', False), device=torch.device("cuda"))
        self.weight = loss_dict.get('weight', 1.0)

    def forward(self, x, y):
        return self.func(x, y) * self.weight


# SSIM 计算类
class ssim:
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.weight = loss_dict.get('weight', 1.0)
        self.loss_item = pyiqa.create_metric('ssim', as_loss=loss_dict.get('as_loss', False), device=torch.device("cuda"))

    def forward(self, x, y):
        return self.weight * self.loss_item(x.clamp(0., 1.), y.clamp(0., 1.))


# PSNR 计算类
class psnr:
    def __init__(self, loss_dict):
        self.loss_dict = loss_dict
        self.as_loss = loss_dict.get('as_loss', False)
        self.weight = loss_dict.get('weight', 1.0)
        self.loss_item = torch.nn.MSELoss()

    def forward(self, x, y, data_range=1.):
        if not self.as_loss:
            with torch.no_grad():
                mse = self.loss_item(x.clamp(0, 1.), y.clamp(0., 1.))
                return 10 * torch.log10(data_range**2 / (mse + 1e-14))
        else:
            mse = self.loss_item(x, y)
            return 10 * torch.log10(data_range**2 / mse)


# 计算所有评价指标的函数
def calculate_metrics(image1, image2, loss_dict_lpips=None, loss_dict_ssim=None, loss_dict_psnr=None):
    # Convert numpy arrays to PyTorch tensors
    if isinstance(image1, np.ndarray):
        image1 = torch.from_numpy(image1).float()
    if isinstance(image2, np.ndarray):
        image2 = torch.from_numpy(image2).float()

    # Ensure images are in the correct format for pyiqa (C, H, W)
    if image1.ndimension() == 3:
        image1 = image1.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    if image2.ndimension() == 3:
        image2 = image2.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    # Normalize the pixel values to the range [0, 1]
    image1 = image1 / 255.0 if image1.max() > 1.0 else image1
    image2 = image2 / 255.0 if image2.max() > 1.0 else image2

    # LPIPS计算
    lpips_metric = lpips(loss_dict_lpips) if loss_dict_lpips else None
    lpips_value = lpips_metric.forward(image1, image2) if lpips_metric else None

    # SSIM计算
    ssim_metric = ssim(loss_dict_ssim) if loss_dict_ssim else None
    ssim_value = ssim_metric.forward(image1, image2) if ssim_metric else None

    # PSNR计算
    psnr_metric = psnr(loss_dict_psnr) if loss_dict_psnr else None
    psnr_value = psnr_metric.forward(image1, image2) if psnr_metric else None

    return lpips_value, ssim_value, psnr_value
