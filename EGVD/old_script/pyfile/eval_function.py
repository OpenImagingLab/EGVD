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


# MANIQA 计算类
class maniqa:
    def __init__(self, loss_dict):
        self.func = pyiqa.create_metric('maniqa', as_loss=loss_dict.get('as_loss', False), device=torch.device("cuda"))
        self.weight = loss_dict.get('weight', 1.0)
    
    def forward(self, x, y=None):
        # MANIQA是无参考质量评估指标，只需要测试图像
        return self.func(x) * self.weight


# MUSIQ 计算类
class musiq:
    def __init__(self, loss_dict):
        self.func = pyiqa.create_metric('musiq', as_loss=loss_dict.get('as_loss', False), device=torch.device("cuda"))
        self.weight = loss_dict.get('weight', 1.0)
    
    def forward(self, x, y=None):
        # MUSIQ是无参考质量评估指标，只需要测试图像
        return self.func(x) * self.weight


# LIQE 计算类
class liqe:
    def __init__(self, loss_dict):
        self.func = pyiqa.create_metric('liqe', as_loss=loss_dict.get('as_loss', False), device=torch.device("cuda"))
        self.weight = loss_dict.get('weight', 1.0)
    
    def forward(self, x, y=None):
        # LIQE是无参考质量评估指标，只需要测试图像
        return self.func(x) * self.weight


# 计算所有评价指标的函数
def calculate_metrics(image1, image2=None, loss_dict_lpips=None, loss_dict_ssim=None, loss_dict_psnr=None, 
                     loss_dict_maniqa=None, loss_dict_musiq=None, loss_dict_liqe=None):
    # Convert numpy arrays to PyTorch tensors
    if isinstance(image1, np.ndarray):
        image1 = torch.from_numpy(image1).float()
    if image2 is not None and isinstance(image2, np.ndarray):
        image2 = torch.from_numpy(image2).float()

    # Ensure images are in the correct format for pyiqa (C, H, W)
    if image1.ndimension() == 3:
        image1 = image1.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    if image2 is not None and image2.ndimension() == 3:
        image2 = image2.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    # Normalize the pixel values to the range [0, 1]
    image1 = image1 / 255.0 if image1.max() > 1.0 else image1
    if image2 is not None:
        image2 = image2 / 255.0 if image2.max() > 1.0 else image2

    results = {}

    # 有参考指标计算 (需要参考图像)
    if image2 is not None:
        # LPIPS计算
        if loss_dict_lpips:
            lpips_metric = lpips(loss_dict_lpips)
            results['lpips'] = lpips_metric.forward(image1, image2)

        # SSIM计算
        if loss_dict_ssim:
            ssim_metric = ssim(loss_dict_ssim)
            results['ssim'] = ssim_metric.forward(image1, image2)

        # PSNR计算
        if loss_dict_psnr:
            psnr_metric = psnr(loss_dict_psnr)
            results['psnr'] = psnr_metric.forward(image1, image2)

    # 无参考指标计算 (只需要测试图像)
    # MANIQA计算
    if loss_dict_maniqa:
        maniqa_metric = maniqa(loss_dict_maniqa)
        results['maniqa'] = maniqa_metric.forward(image1)

    # MUSIQ计算
    if loss_dict_musiq:
        musiq_metric = musiq(loss_dict_musiq)
        results['musiq'] = musiq_metric.forward(image1)

    # LIQE计算
    if loss_dict_liqe:
        liqe_metric = liqe(loss_dict_liqe)
        results['liqe'] = liqe_metric.forward(image1)

    return results