import os
import torch
import argparse
import copy
from diffusers.utils import load_image, export_to_video
from diffusers import UNetSpatioTemporalConditionModel
from custom_diffusers.pipelines.pipeline_frame_interpolation_with_noise_injection import FrameInterpolationWithNoiseInjectionPipeline
# from custom_diffusers.pipelines.evs_pipeline_frame_interpolation_with_noise_injection_color_fuse2 import EVSFrameInterpolationWithNoiseInjectionPipeline
from custom_diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from attn_ctrl.attention_control import (AttentionStore, 
                                         register_temporal_self_attention_control, 
                                         register_temporal_self_attention_flip_control,
)
from dataset.stable_video_dataset_2svd_resize import StableVideoDataset,StableVideoTestDataset
from torch.utils.data import DataLoader
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image
import torch

from eval_function import calculate_metrics

def tensor_to_pillow(tensor, save_path):
    # Squeeze the batch dimension if exists and convert to numpy
    # print("val:",torch.max(tensor),torch.min(tensor))
    image_data = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()/2.0 + 0.5
    image_data = image_data /np.max(image_data) * 255
    image_data = image_data.astype("uint8")
    # Create a PIL image
    pil_image = Image.fromarray(image_data)
    # Save the image
    pil_image.save(save_path)
    return pil_image


def main(args):

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipe = FrameInterpolationWithNoiseInjectionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        scheduler=noise_scheduler,
        variant="fp16",
        torch_dtype=torch.float16, 
    )
    ref_unet = pipe.ori_unet
    state_dict = pipe.unet.state_dict()
    # computing delta w
    finetuned_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.checkpoint_dir,
        subfolder="unet",
        torch_dtype=torch.float16,
    ) 
    assert finetuned_unet.config.num_frames==9
    ori_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        "/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/stable-video-diffusion-img2vid",
        subfolder="unet",
        variant='fp16',
        torch_dtype=torch.float16,
    )


    finetuned_state_dict = finetuned_unet.state_dict()
    ori_state_dict = ori_unet.state_dict()
    for name, param in finetuned_state_dict.items():
        if 'temporal_transformer_blocks.0.attn1.to_v' in name or "temporal_transformer_blocks.0.attn1.to_out.0" in name:
            delta_w = param - ori_state_dict[name]
            state_dict[name] = state_dict[name] + delta_w
    pipe.unet.load_state_dict(state_dict)
    controller_ref= AttentionStore()
    register_temporal_self_attention_control(ref_unet, controller_ref)
    controller = AttentionStore()
    register_temporal_self_attention_flip_control(pipe.unet, controller, controller_ref)
    pipe = pipe.to(args.device)

    
    # run inference
    generator = torch.Generator(device=args.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
        
    dataset = StableVideoTestDataset(args,args.frames_dirs,num_frames=pipe.unet.config.num_frames,skip_sampling_rate=args.skip_sampling_rate)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, batch in enumerate(dataloader):
        # print(i)
        # evs = batch["event_voxel_bin"]
        frame2 = batch["conditions"]
        frame1 = batch["pixel_values"][:, 0]
        save_name = batch["save_name"]

        frames = pipe(image1=frame1, image2=frame2, height=frame1.shape[-2], width=frame1.shape[-1],
                    num_inference_steps=args.num_inference_steps, 
                    generator=generator,
                    weighted_average=args.weighted_average,
                    noise_injection_steps=args.noise_injection_steps,
                    noise_injection_ratio= args.noise_injection_ratio,
            ).frames[0]
        
        save_path = args.out_path
        save_path = save_path.replace("example",save_name[0])
        
        # 对每一帧进行 resize，确保尺寸为 512×512
        resized_frames = [frame.resize((512, 512), Image.BILINEAR) for frame in frames]

        if save_path.endswith('.gif'):
            resized_frames[0].save(save_path, save_all=True, append_images=resized_frames[1:], duration=142, loop=0)
        else:
            export_to_video(resized_frames, save_path, fps=7)

        
        from eval_function import calculate_metrics
        # 假设要保存txt结果
        txt_path = save_path.replace(".gif", "_metrics.txt")
        with open(txt_path, "w") as f:  # 使用'a'模式来附加内容
            f.write(f"Processing: {save_path}\n")
            
            lpips_values, ssim_values, psnr_values = [], [], []

            for i, frame in enumerate(frames):
                # 将 PIL 图像 resize 到 512x512（使用双线性插值）
                resized_frame = frame.resize((512, 512), Image.BILINEAR)
                # 保存 resize 后的帧
                resized_frame.save(save_path.replace(".gif", f"_{i+1}s.png"))
                
                frame_tensor = batch["pixel_values"][:, i]
                # 从 tensor 转为 PIL 图像后，也做 resize
                frame_tensor = F.interpolate(frame_tensor, size=(512, 512), mode='bilinear', align_corners=False)
                frame_pillow = tensor_to_pillow(frame_tensor, save_path.replace(".gif", f"_{i+1}.png"))
                # frame_pillow = frame_pillow.resize((512, 512), Image.BILINEAR)
                print(save_path.replace(".gif", f"_{i+1}.png"), len(frames))
                
                # 如果是计算评价指标的帧，则先转换为 numpy 数组，并用 cv2.resize 保证尺寸一致
                # 注意 cv2.resize 的尺寸参数顺序是 (width, height)
                image1 = np.array(resized_frame)
                image2 = np.array(frame_pillow)
                image1 = cv2.resize(image1, (512, 512), interpolation=cv2.INTER_LINEAR)
                image2 = cv2.resize(image2, (512, 512), interpolation=cv2.INTER_LINEAR)
                
                print(image1.shape, image2.shape, np.min(image1), np.max(image1), np.min(image2), np.max(image2))
                
                lpips_value, ssim_value, psnr_value = calculate_metrics(
                    image1, image2, 
                    loss_dict_lpips={'as_loss': False, 'weight': 1.0},
                    loss_dict_ssim={'weight': 1.0}, 
                    loss_dict_psnr={'as_loss': False, 'weight': 1.0}
                )
                
                # 将当前帧的结果追加到列表中
                lpips_values.append(lpips_value.cpu().numpy())
                ssim_values.append(ssim_value.cpu().numpy())
                psnr_values.append(psnr_value.cpu().numpy())

            # 计算平均指标
            avg_lpips = np.mean(lpips_values) if lpips_values else None
            avg_ssim = np.mean(ssim_values)if ssim_values else None
            avg_psnr = np.mean(psnr_values) if psnr_values else None

            # 写入平均指标到txt
            f.write(f"Average LPIPS: {avg_lpips}, Average SSIM: {avg_ssim}, Average PSNR: {avg_psnr}\n")
            f.write("\n")  # 在每个视频的结果后加一个空行


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    # parser.add_argument('--frame1_path', type=str, required=True)
    # parser.add_argument('--frame2_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--weighted_average', action='store_true')
    parser.add_argument('--noise_injection_steps', type=int, default=0)
    parser.add_argument('--noise_injection_ratio', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--frames_dirs', type=str, default='cuda:0')
    parser.add_argument('--event_filter', type=str, default=None)
    parser.add_argument('--skip_sampling_rate', type=int, default=1)
    args = parser.parse_args()
    out_dir = os.path.dirname(args.out_path)
    os.makedirs(out_dir, exist_ok=True)
    main(args)
