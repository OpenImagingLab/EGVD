import os
import torch
import argparse
import copy
from diffusers.utils import load_image, export_to_video
from diffusers import UNetSpatioTemporalConditionModel
from custom_diffusers.pipelines.pipeline_frame_interpolation_with_noise_injection import FrameInterpolationWithNoiseInjectionPipeline
from custom_diffusers.pipelines.evs_pipeline_frame_interpolation_with_noise_injection_color_fuse import EVSFrameInterpolationWithNoiseInjectionPipeline
from custom_diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from attn_ctrl.attention_control import (AttentionStore, 
                                         register_temporal_self_attention_control, 
                                         register_temporal_self_attention_flip_control,
)
from dataset.stable_video_dataset_fuse import StableVideoDataset,StableVideoTestDataset
from torch.utils.data import DataLoader
from einops import rearrange

import numpy as np
import cv2

from PIL import Image
import torch

def tensor_to_pillow(tensor, save_path):
    # Squeeze the batch dimension if exists and convert to numpy
    # print("val:",torch.max(tensor),torch.min(tensor))
    image_data = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()/2.0 + 0.5
    image_data = image_data/np.max(image_data) * 255
    image_data = image_data.astype("uint8")
    # Create a PIL image
    pil_image = Image.fromarray(image_data)
    # Save the image
    pil_image.save(save_path)


def main(args):

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # pipe = FrameInterpolationWithNoiseInjectionPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path, 
    #     scheduler=noise_scheduler,
    #     variant="fp16",
    #     torch_dtype=torch.float16, 
    # )
    pipe = EVSFrameInterpolationWithNoiseInjectionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        # args.checkpoint_dir,
        scheduler=noise_scheduler,
        variant="fp16",
        torch_dtype=torch.float16, 
    )
    
    lef_path = args.checkpoint_dir + "/lef_model_checkpoint.pt"
    print("load_lef_model:",lef_path)
    state_dict = torch.load(lef_path, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    pipe.lef_model.load_state_dict(new_state_dict)   
    # ref_unet = pipe.ori_unet
    
    # # print("-----"*10)

    
    # state_dict = pipe.unet.state_dict()
    # # computing delta w
    finetuned_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.checkpoint_dir,
        subfolder="unet",
        torch_dtype=torch.float16,
    ) 
    # # assert finetuned_unet.config.num_frames==14
    # ori_unet = UNetSpatioTemporalConditionModel.from_pretrained(
    #     "/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/stable-video-diffusion-img2vid",
    #     subfolder="unet",
    #     variant='fp16',
    #     torch_dtype=torch.float16,
    # )

    # # print("-----"*10)

    finetuned_state_dict = finetuned_unet.state_dict()
    # ori_state_dict = ori_unet.state_dict()
    # for name, param in finetuned_state_dict.items():
    #     if 'temporal_transformer_blocks.0.attn1.to_v' in name or "temporal_transformer_blocks.0.attn1.to_out.0" in name:
    #         delta_w = param - ori_state_dict[name]
    #         state_dict[name] = state_dict[name] + delta_w
    # pipe.unet.load_state_dict(state_dict)
    
    pipe.unet.load_state_dict(finetuned_state_dict)

    # # controller_ref= AttentionStore()
    # # register_temporal_self_attention_control(ref_unet, controller_ref)

    # # controller = AttentionStore()
    # # register_temporal_self_attention_flip_control(pipe.unet, controller, controller_ref)
    
    del finetuned_unet
    # del ori_unet

    pipe = pipe.to(args.device)


    print("-----"*10)
    
    # run inference
    generator = torch.Generator(device=args.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
        
    dataset = StableVideoTestDataset(args,args.frames_dirs,num_frames=pipe.unet.config.num_frames,skip_sampling_rate=args.skip_sampling_rate)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    print("-----"*10)
    for i, batch in enumerate(dataloader):
        print(i)
    
        evs = batch["event_voxel_bin"]
        frame2 = batch["conditions"]
        frame1 = batch["pixel_values"][:, 0]
        save_name = batch["save_name"]
        print(save_name)
        print(frame1.shape,frame2.shape,evs.shape)
        
        
        # frame11 = load_image(args.frame1_path)
        # # frame1 = frame1.resize((854, 640))
        # frame11 = frame11.resize((1024, 576))
        

        # frame22 = load_image(args.frame2_path)
        # frame22 = frame22.resize((1024, 576))
        # # # print("************"*20,frame11.shape,np.mean(frame11))
        # from torchvision import transforms
        # print("************" * 20, (tensor := transforms.ToTensor()(frame11)).shape, tensor.mean(),frame2.shape)
        
        # frame2 = frame2.resize((854, 640))
        # frame2 = frame2.resize((1024, 576))

        frames = pipe(image1=frame1, image2=frame2, evs=evs, height=frame1.shape[-2], width=frame1.shape[-1],
                    num_inference_steps=args.num_inference_steps, 
                    generator=generator,
                    weighted_average=args.weighted_average,
                    noise_injection_steps=args.noise_injection_steps,
                    noise_injection_ratio= args.noise_injection_ratio,
        ).frames[0]
        save_path = args.out_path
        save_path = save_path.replace("example",save_name[0])
        
        if save_path.endswith('.gif'):
            frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=142, loop=0)
        else:
            export_to_video(frames, save_path, fps=7)

        # frame1s = frames[0]  # 第一帧
        # frame2s = frames[-1]  # 最后一帧
        # frame1s.save(save_path.replace(".gif", "_1s.png"))
        # frame2s.save(save_path.replace(".gif", "_2s.png"))
        # # frame1.save(save_path.replace(".gif", "_1.png"))
        # # frame2.save(save_path.replace(".gif", "_2.png"))
        # frame1 = batch["pixel_values"][:, 0] 从batch["pixel_values"]取出所有framei，从frames里取出所有frameis保存下来
        # tensor_to_pillow(frame1,save_path.replace(".gif", "_1.png"))
        # tensor_to_pillow(frame2,save_path.replace(".gif", "_2.png"))

        # print(save_path.replace(".gif", "_1.png"), len(frames))
        for i, frame in enumerate(frames):
            frame.save(save_path.replace(".gif", f"_{i+1}s.png"))
            frame_tensor = batch["pixel_values"][:, i]
            tensor_to_pillow(frame_tensor, save_path.replace(".gif", f"_{i+1}.png"))
            print(save_path.replace(".gif", f"_{i+1}.png"), len(frames))


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
