import os
from glob import glob
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from natsort import natsorted
import cv2

def mask_function(event_image, kernel_size=31, kernel_size_erode=61,kernel_size_midele=31, iterations=1, sigma_log=10):
    event_image = np.abs(event_image) / np.max(np.abs(event_image))
    event_image_blurred = cv2.GaussianBlur(event_image, (kernel_size,kernel_size), sigma_log)
    _, binary_image = cv2.threshold(event_image_blurred, 0.01, 1, cv2.THRESH_BINARY)    
    # # # 对二值化图像进行腐蚀
    # kernel_erode = np.ones((kernel_size, kernel_size), np.uint8)  # 定义腐蚀的核
    # binary_image_eroded = cv2.erode(binary_image, kernel_erode, iterations=iterations)
    kernel_dilate = np.ones((kernel_size_erode, kernel_size_erode), np.uint8)  # 定义膨胀的核
    binary_image_dilated = cv2.dilate(binary_image, kernel_dilate, iterations=iterations )
    binary_median = cv2.medianBlur(binary_image_dilated.astype(np.uint8), kernel_size_midele)
    return binary_median

def save_debug_images_as_rgb(debug_dir, event_image, cumulative_image, B):
    os.makedirs(debug_dir, exist_ok=True)
    for b in range(B):
        event_channel = event_image[:, :, b]
        cumulative_channel = cumulative_image[:, :, b]
        event_channel_rgb = np.clip(event_channel * 255, 0, 255).astype(np.uint8)
        cumulative_channel_rgb = np.clip(cumulative_channel * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f"event_image_channel_{b}.png"), event_channel_rgb)
        cv2.imwrite(os.path.join(debug_dir, f"cumulative_image_channel_{b}.png"), cumulative_channel_rgb)
    
def create_event_image(args, x, y, p, t, shape, B=3, debug=False):
    height, width = shape[:2]
    event_image = np.zeros((height, width, B), dtype=np.float32)
    cumulative_image = np.zeros((height, width, B), dtype=np.float32) 
    start_time = t[0]
    end_time = t[-1]
    delta_T = end_time - start_time
    normalized_timestamps = (B - 1) * (t - start_time) / delta_T
    x = np.clip(x.astype(int), 0, width - 1)
    y = np.clip(y.astype(int), 0, height - 1)

    bin_idx = np.round(normalized_timestamps).astype(int)
    bin_idx = np.clip(bin_idx, 0, B - 1)  

    weights = np.maximum(0, 1 - np.abs(normalized_timestamps - bin_idx))
    np.add.at(event_image, (y, x, bin_idx), p * weights)

    # max_evs = np.max(event_image)
    # if max_evs > 0:
    #     event_image = np.clip(event_image.astype(np.float32) / max_evs, 0, 1)
    norm_value_evs = np.maximum(np.abs(np.min(event_image)), np.max(event_image))
    event_image = (event_image + norm_value_evs)/ (2 * norm_value_evs)

    if args.event_filter == "great_filter":
        for i in range(0, B, 3):  # 步长为3，处理每组三个通道
            mask_group = []
            for j in range(i, min(i + 3, B)):
                mask = (bin_idx <= j) & (bin_idx >= np.maximum(0, j - 2))
                cumulative_image_f = cumulative_image[:, :, j].copy()
                np.add.at(cumulative_image_f, (y[mask], x[mask]), np.abs(p[mask]))
                motion_mask = mask_function(cumulative_image_f)
                mask_group.append(motion_mask)
            if len(mask_group) == 3:  # 确保有3个mask
                combined_mask = np.logical_or(np.logical_or(mask_group[0], mask_group[1]), mask_group[2])
            else:
                combined_mask = mask_group[0]  # 如果只有1或2个mask，则直接使用它们
            for j in range(i, min(i + 3, B)):
                cumulative_image[:, :, j] = combined_mask.astype(np.uint8)
        if debug:
            save_debug_images_as_rgb("./debug_output", event_image, cumulative_image, B)
        return event_image * cumulative_image
    else:
        return event_image


# def create_event_image(args, x, y, p, t, shape, B=3, theta=1.0, debug=True):
#     height, width = shape[:2]
#     event_image = np.zeros((height, width, B), dtype=np.float32)
#     cumulative_image = np.zeros((height, width, B), dtype=np.float32)  # 用于累积从0到切分位置的事件
#     start_time = t[0]
#     end_time = t[-1]
#     delta_T = end_time - start_time
#     normalized_timestamps = (B - 1) * (t - start_time) / delta_T
#     x = np.clip(x.astype(int), 0, width - 1)
#     y = np.clip(y.astype(int), 0, height - 1)
    
#     # 计算时间片段对应的bin_idx
#     bin_idx = np.round(normalized_timestamps).astype(int)
#     bin_idx = np.clip(bin_idx, 0, B - 1)  # 限制bin_idx在[0, B-1]范围内
#     print("bin_idx", bin_idx)
    
#     # 计算每个时间戳在bin_idx上的权重
#     weights = np.maximum(0, 1 - np.abs(normalized_timestamps - bin_idx))
    
#     # 累积事件到对应的bin_idx位置
#     np.add.at(event_image, (y, x, bin_idx), p * weights)
    
#     # 对事件进行EDI模型的积分
#     for i in range(len(t) - 1):
#         # 获取事件的时间差，计算时间段内的事件累积
#         t_i = t[i]
#         t_j = t[i+1]
#         delta_t = t_j - t_i
        
#         # 计算事件的衰减
#         decay_factor = np.exp(theta * (np.sum(p[i:i+1])))  # 这里使用一个简单的积分示例，真实情况可能需要更复杂的积分计算
        
#         # 累积事件影响到cumulative_image
#         np.add.at(cumulative_image, (y[i], x[i], bin_idx[i]), p[i] * decay_factor)

#     # 归一化事件图像
#     max_evs = np.max(event_image)
#     if max_evs > 0:
#         event_image = np.clip(event_image.astype(np.float32) / max_evs, 0, 1)
    
#     # 如果启用调试模式，保存调试信息作为RGB图像
#     if debug:
#         save_debug_images_as_rgb("./debug_output", event_image, cumulative_image, B)
    
#     # 返回两个量：切分内累积的事件图像和从0到切分位置的累积
#     # return event_image, cumulative_image
#     return event_image

class StableVideoDataset(Dataset):
    def __init__(self, 
        args,
        video_data_dir, 
        max_num_videos=None,
        frame_hight=576, frame_width=1024, num_frames=14,
        is_reverse_video=True,
        random_seed=42,
        double_sampling_rate=False,
        
    ):  
        self.video_data_dir = video_data_dir
        video_names = sorted([video for video in os.listdir(video_data_dir) 
                    if os.path.isdir(os.path.join(video_data_dir, video))])
        
        self.length = min(len(video_names), max_num_videos) if max_num_videos is not None else len(video_names)
        
        self.video_names = video_names[:self.length]
        if double_sampling_rate:
            self.sample_frames = num_frames*2-1
            self.sample_stride = 2

        else:
            self.sample_frames = num_frames
            self.sample_stride = 1
        self.evnt_nums_between = num_frames 

        self.frame_width = frame_width
        self.frame_height = frame_hight
        self.pixel_transforms = transforms.Compose([
            # transforms.Resize((self.frame_height, self.frame_width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.is_reverse_video = is_reverse_video
        self.args = args
        np.random.seed(random_seed)

    def load_npz(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        sync_rgb = data['sync_rgb']
        return sync_rgb, data['x'], data['y'], data['p'], data['t'], data['sync_rgb_shape']

    def accumulate_events(self, video_frame_paths, insertion_rate_iter):
        x_list, y_list, p_list, t_list = [], [], [], []
        shape = None
        video_frame = []
        for i,pathi in enumerate(video_frame_paths):
            sync_rgb, x, y, p, t, sync_rgb_shape = self.load_npz(pathi)
            video_frame.append(sync_rgb[..., ::-1].astype(np.float32)/255.0)
            # if i < len(video_frame_paths):
            x_list.append(x)
            y_list.append(y)
            p_list.append(p)
            t_list.append(t)
            shape = sync_rgb_shape
        x_all = np.concatenate(x_list)
        y_all = np.concatenate(y_list)
        p_all = np.concatenate(p_list)
        t_all = np.concatenate(t_list)
        event_voxel_bin = create_event_image(self.args, x_all, y_all, p_all, t_all, shape, insertion_rate_iter * 3)
        video_frames = np.stack(video_frame, axis=0)
        pixel_values = torch.from_numpy(video_frames.transpose(0, 3, 1, 2))
        event_voxel_bin = torch.from_numpy(event_voxel_bin.transpose(2, 0, 1)).unsqueeze(1)
        event_voxel_bin = event_voxel_bin.view(insertion_rate_iter, 3, event_voxel_bin.shape[-2], event_voxel_bin.shape[-1])
        pixel_values = (pixel_values - torch.min(pixel_values))/( torch.max(pixel_values) - torch.min(pixel_values) )
        # print("value:",torch.max(pixel_values),torch.min(pixel_values))
        return pixel_values.contiguous(), event_voxel_bin.contiguous()

    def get_batch(self, idx):
        video_name = self.video_names[idx]
        # video_frame_paths = sorted(glob(os.path.join(self.video_data_dir, video_name, '*.png')))
        video_frame_paths = natsorted(glob(os.path.join(self.video_data_dir, video_name, 'RGB-EVS','*.npz')))
        start_idx = np.random.randint(len(video_frame_paths)-self.sample_frames+1)
        video_frame_paths = video_frame_paths[start_idx:start_idx+self.sample_frames:self.sample_stride]
        pixel_values, event_voxel_bin = self.accumulate_events( video_frame_paths,  self.evnt_nums_between )
        return pixel_values,event_voxel_bin

    def crop_center_patch(self, pixel_values, event_voxel_bin, crop_h=512, crop_w=512):
        height = pixel_values.shape[2]
        width = pixel_values.shape[3]
        center_h = height // 2
        center_w = width // 2
        start_h = center_h - crop_h // 2
        start_w = center_w - crop_w // 2
        cropped_pixel_values = pixel_values[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        cropped_event_voxel_bin = event_voxel_bin[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        return cropped_pixel_values, cropped_event_voxel_bin


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                
                pixel_values, event_voxel_bin = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        # import time
        # batch_start_time = time.time()
        # pixel_values, event_voxel_bin = self.get_batch(idx)
        # batch_end_time = time.time()
        # print(f"get_batch 耗时: {batch_end_time - batch_start_time:.4f} 秒")
        
        pixel_values = self.pixel_transforms(pixel_values)
        event_voxel_bin = self.pixel_transforms(event_voxel_bin)
        pixel_values,event_voxel_bin = self.crop_center_patch(pixel_values,event_voxel_bin)
        conditions = pixel_values[-1]
        
        if self.is_reverse_video:
            pixel_values = torch.flip(pixel_values, (0,))
            event_voxel_bin = torch.flip(event_voxel_bin, (0,))
            
        sample = dict(event_voxel_bin=event_voxel_bin, pixel_values=pixel_values, conditions=conditions )
        # sample = {"event_voxel_bin":event_voxel_bin, "pixel_values":pixel_values, "conditions":conditions}
        return sample



class StableVideoTestDataset(StableVideoDataset):
    def __init__(self, 
                 args,
                 video_data_dir, 
                 max_num_videos=None,
                 frame_hight=576, frame_width=1024, num_frames=14,
                 is_reverse_video=True,
                 random_seed=42,
                 double_sampling_rate=False,
                 
                 ):
        # 调用父类的构造函数，初始化继承的变量
        super().__init__(args,video_data_dir, max_num_videos, frame_hight, frame_width, num_frames, is_reverse_video, random_seed, double_sampling_rate)

    def get_batch(self, idx):
        video_name = self.video_names[idx]
        print(video_name)
        video_frame_paths = natsorted(glob(os.path.join(self.video_data_dir, video_name, 'RGB-EVS','*.npz')))
        # start_idx = np.random.randint(len(video_frame_paths)-self.sample_frames+1)
        start_idx = 1
        video_frame_paths = video_frame_paths[start_idx:start_idx+self.sample_frames:self.sample_stride]
        # print(video_frame_paths)
        pixel_values, event_voxel_bin = self.accumulate_events( video_frame_paths,  self.evnt_nums_between )
        save_name =  video_name + "_id" + str(start_idx)
        return pixel_values, event_voxel_bin, save_name


    def __getitem__(self, idx):
        pixel_values, event_voxel_bin, save_name = self.get_batch(idx)
        # while True:
        #     try:
        #         pixel_values, event_voxel_bin, save_name = self.get_batch(idx)
        #         break

        #     except Exception as e:
        #         idx = random.randint(0, self.length-1)

        # import time
        # batch_start_time = time.time()
        # pixel_values, event_voxel_bin = self.get_batch(idx)
        # batch_end_time = time.time()
        # print(f"get_batch 耗时: {batch_end_time - batch_start_time:.4f} 秒")
        # print("---"*20,pixel_values.shape, event_voxel_bin.shape)
        
        pixel_values = self.pixel_transforms(pixel_values)
        event_voxel_bin = self.pixel_transforms(event_voxel_bin)
        pixel_values,event_voxel_bin = self.crop_center_patch(pixel_values,event_voxel_bin)
        
        
        # n = event_voxel_bin.shape[0]
        # for i in range(n):
        #     event = event_voxel_bin[i, :, :, :].permute(1, 2, 0) 
        #     event = (event / torch.max(event))
        #     event = torch.clamp(event * 255, 0, 255).byte() 
        #     event = event.cpu().numpy()
            
        #     print(np.max(event),np.mean(event))
            
        #     name = "d2e2_a"+str(idx)+"_"+str(i)+"_.png"
        #     # name = "a"+str(idx)+"_"+str(i)+"_.png"
        #     cv2.imwrite("/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/"+name,event)
        # h = 512
        # w = 512
        # pixel_values = pixel_values[:, :, :h, :w]
        # event_voxel_bin = event_voxel_bin[:, :, :h, :w]        

        conditions = pixel_values[-1]
        
        if self.is_reverse_video:
            pixel_values = torch.flip(pixel_values, (0,))
            event_voxel_bin = torch.flip(event_voxel_bin, (0,))
            
        sample = dict(event_voxel_bin=event_voxel_bin, pixel_values=pixel_values, conditions=conditions,save_name=save_name )
        return sample

# # frame_hight=576, frame_width=1024, num_frames=14,
# #  frame_hight=512, frame_width=512, num_frames=9,
# class StableVideoTestDataset(Dataset):
#     def __init__(self, 
#         video_data_dir, 
#         max_num_videos=None,
#         frame_hight=576, frame_width=1024, num_frames=14,
#         is_reverse_video=True,
#         random_seed=42,
#         double_sampling_rate=False,
#     ):  
#         self.video_data_dir = video_data_dir
#         video_names = sorted([video for video in os.listdir(video_data_dir) 
#                     if os.path.isdir(os.path.join(video_data_dir, video))])
        
#         self.length = min(len(video_names), max_num_videos) if max_num_videos is not None else len(video_names)
#         # print("--"*100,self.length)
        
#         self.video_names = video_names[:self.length]
#         if double_sampling_rate:
#             self.sample_frames = num_frames*2-1
#             self.sample_stride = 2

#         else:
#             self.sample_frames = num_frames
#             self.sample_stride = 1
#         self.evnt_nums_between = num_frames 

#         self.frame_width = frame_width
#         self.frame_height = frame_hight
#         self.pixel_transforms = transforms.Compose([
#             transforms.Resize((self.frame_height, self.frame_width), interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
#         ])

#         self.is_reverse_video = is_reverse_video
#         np.random.seed(random_seed)

#     def load_npz(self, file_path):
#         data = np.load(file_path, allow_pickle=True)
#         sync_rgb = data['sync_rgb']
#         return sync_rgb, data['x'], data['y'], data['p'], data['t'], data['sync_rgb_shape']

#     def accumulate_events(self, video_frame_paths, insertion_rate_iter):
#         x_list, y_list, p_list, t_list = [], [], [], []
#         shape = None
#         video_frame = []
#         for i,pathi in enumerate(video_frame_paths):
#             sync_rgb, x, y, p, t, sync_rgb_shape = self.load_npz(pathi)
#             video_frame.append(sync_rgb[..., ::-1].astype(np.float32)/255.0)
#             # if i < len(video_frame_paths):
#             x_list.append(x)
#             y_list.append(y)
#             p_list.append(p)
#             t_list.append(t)
#             shape = sync_rgb_shape
#         x_all = np.concatenate(x_list)
#         y_all = np.concatenate(y_list)
#         p_all = np.concatenate(p_list)
#         t_all = np.concatenate(t_list)
#         event_voxel_bin = create_event_image(x_all, y_all, p_all, t_all, shape, insertion_rate_iter * 3)
#         video_frames = np.stack(video_frame, axis=0)
#         pixel_values = torch.from_numpy(video_frames.transpose(0, 3, 1, 2))
#         event_voxel_bin = torch.from_numpy(event_voxel_bin.transpose(2, 0, 1)).unsqueeze(1)
#         event_voxel_bin = event_voxel_bin.view(insertion_rate_iter, 3, event_voxel_bin.shape[-2], event_voxel_bin.shape[-1])
#         return pixel_values.contiguous(), event_voxel_bin.contiguous()

#     def get_batch(self, idx):
#         video_name = self.video_names[idx]
#         # print(video_name)
#         video_frame_paths = natsorted(glob(os.path.join(self.video_data_dir, video_name, 'RGB-EVS','*.npz')))
#         start_idx = np.random.randint(len(video_frame_paths)-self.sample_frames+1)
#         start_idx = 1
#         # video_frame_paths = video_frame_paths[start_idx:start_idx+self.sample_frames:self.sample_stride]
#         pixel_values, event_voxel_bin = self.accumulate_events( video_frame_paths,  self.evnt_nums_between )
#         save_name =  video_name + "_id" + str(start_idx)
#         return pixel_values, event_voxel_bin, save_name

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         while True:
#             try:
#                 pixel_values, event_voxel_bin, save_name = self.get_batch(idx)
#                 break

#             except Exception as e:
#                 idx = random.randint(0, self.length-1)

#         # import time
#         # batch_start_time = time.time()
#         # pixel_values, event_voxel_bin = self.get_batch(idx)
#         # batch_end_time = time.time()
#         # print(f"get_batch 耗时: {batch_end_time - batch_start_time:.4f} 秒")
#         # print("---"*20,pixel_values.shape, event_voxel_bin.shape)
        
#         pixel_values = self.pixel_transforms(pixel_values)
#         event_voxel_bin = self.pixel_transforms(event_voxel_bin)
        
#         # h = 512
#         # w = 512
#         # pixel_values = pixel_values[:, :, :h, :w]
#         # event_voxel_bin = event_voxel_bin[:, :, :h, :w]        

#         conditions = pixel_values[-1]
        
#         if self.is_reverse_video:
#             pixel_values = torch.flip(pixel_values, (0,))
#             event_voxel_bin = torch.flip(event_voxel_bin, (0,))
            
#         sample = dict(event_voxel_bin=event_voxel_bin, pixel_values=pixel_values, conditions=conditions,save_name=save_name )
#         return sample

