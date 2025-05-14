# from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
# import math
# import time
# import io
# import random
# import numpy as np
# import cv2
# from PIL import Image
# import torch.utils.data as data
# import os
# import random
# import numpy as np
# import glob
# import torch.utils.data as data
# from concurrent.futures import ThreadPoolExecutor
# import pickle
# import torch 
# from torchvision import transforms

# def load_file_list(file_list_path):
#     files = []
#     with open(file_list_path, "r") as fin:
#         for line in fin:
#             path = line.strip()
#             if path:
#                 files.append({"image_path": path, "prompt": ""})
#     return files


# def center_crop_arr(image, image_size):
#     h, w = image.shape[:2]
#     if min([h,w]) < image_size:  
#         scale = image_size / min(h, w)
#     else:
#         scale = 1
#     new_w = int(w * scale)
#     new_h = int(h * scale)
#     resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
#     start_x = (new_w - image_size) // 2
#     start_y = (new_h - image_size) // 2
#     cropped_image = resized_image[start_y:start_y + image_size, start_x:start_x + image_size]
#     return cropped_image


# def random_crop_arr(image, image_size, x_random, y_random): 
#     h, w = image.shape[:2]
#     if min([h,w]) < image_size:  
#         scale = image_size / min(h, w)
#     else:
#         scale = 1
#     new_w = int(w * scale)
#     new_h = int(h * scale)
#     resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
#     start_x = int((new_w - image_size) * x_random)
#     start_y = int((new_h - image_size) * y_random)
#     cropped_image = resized_image[start_y:start_y + image_size, start_x:start_x + image_size]
#     return cropped_image


# def create_event_image2(x, y, p, t, shape):
#     height, width = shape[:2]
#     event_image = np.zeros((height, width, 3), dtype=np.float32)
#     x = x.astype(int)
#     y = y.astype(int)
#     positive_timestamp_sum = np.zeros((height, width), dtype=np.float32)
#     negative_timestamp_sum = np.zeros((height, width), dtype=np.float32)
#     positive_event_count = np.zeros((height, width), dtype=np.int32)
#     negative_event_count = np.zeros((height, width), dtype=np.int32)

#     for i in range(len(x)):
#         if p[i] > 0:
#             positive_timestamp_sum[y[i], x[i]] += t[i] - t[0]
#             positive_event_count[y[i], x[i]] += 1
#         else:
#             negative_timestamp_sum[y[i], x[i]] += t[i] - t[0]
#             negative_event_count[y[i], x[i]] += 1
#         event_image[y[i], x[i], 0] += 1

#     positive_nonzero_indices = positive_event_count != 0
#     negative_nonzero_indices = negative_event_count != 0
#     positive_average_timestamp = np.zeros_like(positive_timestamp_sum)
#     negative_average_timestamp = np.zeros_like(negative_timestamp_sum)

#     positive_average_timestamp[positive_nonzero_indices] = (
#         positive_timestamp_sum[positive_nonzero_indices] / positive_event_count[positive_nonzero_indices]
#     )
#     negative_average_timestamp[negative_nonzero_indices] = (
#         negative_timestamp_sum[negative_nonzero_indices] / negative_event_count[negative_nonzero_indices]
#     )

#     event_image[:, :, 1] = positive_average_timestamp
#     event_image[:, :, 2] = negative_average_timestamp
#     if np.max(event_image[:, :, 0]) > 0:
#         event_image[:, :, 0] = np.clip(event_image[:, :, 0] / np.max(event_image[:, :, 0]) * 255, 0, 255)
#     if np.max(event_image[:, :, 1]) > 0:
#         event_image[:, :, 1] = np.clip(event_image[:, :, 1] / np.max(event_image[:, :, 1]) * 255, 0, 255)
#     if np.max(event_image[:, :, 2]) > 0:
#         event_image[:, :, 2] = np.clip(event_image[:, :, 2] / np.max(event_image[:, :, 2]) * 255, 0, 255)

#     return event_image.astype(np.uint8)

# def create_event_image(x, y, p, t, shape, B=3):
#     height, width = shape[:2]
#     event_image = np.zeros((height, width, B), dtype=np.float32)
#     start_time = t[0]
#     end_time = t[-1]
#     delta_T = end_time - start_time
#     normalized_timestamps = (B - 1) * (t - start_time) / delta_T
#     x = np.clip(x.astype(int), 0, width - 1)
#     y = np.clip(y.astype(int), 0, height - 1)
#     bin_idx = np.round(normalized_timestamps).astype(int)
#     bin_idx = np.clip(bin_idx, 0, B - 1)  # 限制bin_idx在[0, B-1]范围内
#     weights = np.maximum(0, 1 - np.abs(normalized_timestamps - bin_idx))
#     np.add.at(event_image, (y, x, bin_idx), p * weights)  # 使用 np.add.at 进行批量更新
#     for b in range(B):
#         if np.max(event_image[:, :, b]) > 0:
#             event_image[:, :, b] = np.clip(event_image[:, :, b] / np.max(event_image[:, :, b]) * 255, 0, 255)   
#     # max_vals = np.max(event_image, axis=(0, 1)) 
#     # nonzero_mask = max_vals > 0  
#     # event_image[:, :, nonzero_mask] = (event_image[:, :, nonzero_mask] / max_vals[nonzero_mask] * 255)
#     return event_image.astype(np.uint8)

# # 示例调用
# # event_image = create_event_image(x, y, p, t, shape=(height, width), B=3)

# class EventVideoDataset(data.Dataset):
#     # def __init__(self, folder_path: str, out_size: int, crop_type: str, insertion_rate: int):
#     def __init__(self, file_list: str, 
#                  out_size: int, 
#                  crop_type: str, 
#                  insertion_rate:int,
#                  frame_hight=576, 
#                  frame_width=1024, 
#                  num_frames=14,
#                  ):
#         super(EventVideoDataset, self).__init__()
#         self.folder_path = file_list
        
#         self.out_size = out_size
#         self.crop_type = crop_type
#         self.insertion_rate = insertion_rate 

#         assert self.crop_type in ["none", "center", "random"]
#         self.pixel_transforms = transforms.Compose([
#             transforms.Resize((self.frame_height, self.frame_width), interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
#         ])

#         # Cache the sorted files from each subfolder during initialization
#         self.subfolder_files = self._cache_files_from_all_folders()

#         self.file_list = self.subfolder_files
#         self.subfolder_lengths = [len(files) - self.insertion_rate for files in self.subfolder_files.values()]

#     def _cache_files_from_all_folders(self):
#         """Load and sort all npz files from each subfolder, caching the results."""
#         subfolder_files = {}
#         for folder in os.listdir(self.folder_path):
#             full_path = os.path.join(self.folder_path, folder)
#             if os.path.isdir(full_path):
#                 npz_files = glob.glob(os.path.join(full_path, 'RGB-EVS/*.npz'))
#                 npz_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))  # Sorting by filename pattern
#                 subfolder_files[full_path] = [{"image_path": file} for file in npz_files]
#         return subfolder_files

#     def load_npz(self, file_path):
#         data = np.load(file_path, allow_pickle=True)
#         sync_rgb = data['sync_rgb']
#         sync_rgb_enhanced = sync_rgb
#         return sync_rgb_enhanced, data['x'], data['y'], data['p'], data['t'], data['sync_rgb_shape']

#     def crop_image(self, image, crop_type, x_random=None, y_random=None):
#         if crop_type == "center":
#             return center_crop_arr(image, self.out_size)
#         elif crop_type == "random":
#             return random_crop_arr(image, self.out_size, x_random, y_random)
#         else:
#             return image

#     def accumulate_events(self, start_index, end_index, files, insertion_rate_iter):
#         x_list, y_list, p_list, t_list = [], [], [], []
#         shape = None
#         for i in range(start_index, end_index):
#             img_path = files[i]["image_path"]
#             sync_rgb, x, y, p, t, sync_rgb_shape = self.load_npz(img_path)
#             x_list.append(x)
#             y_list.append(y)
#             p_list.append(p)
#             t_list.append(t)
#             shape = sync_rgb_shape
#         x_all = np.concatenate(x_list)
#         y_all = np.concatenate(y_list)
#         p_all = np.concatenate(p_list)
#         t_all = np.concatenate(t_list)
#         return create_event_image(x_all, y_all, p_all, t_all, shape, insertion_rate_iter)

#     def _get_subfolder_and_local_index(self, index):
#         """Map the global index to a specific subfolder and local index."""
#         cumulative_length = 0
#         for subfolder, length in zip(self.subfolder_files.keys(), self.subfolder_lengths):
#             if index < cumulative_length + length:
#                 local_index = index - cumulative_length
#                 return subfolder, local_index
#             cumulative_length += length
#         raise IndexError(f"Index {index} out of range")

#     def __getitem__(self, index: int):
#         # Get the subfolder and local index from the global index
#         subfolder, local_index = self._get_subfolder_and_local_index(index)
#         image_files = self.subfolder_files[subfolder]
        
#         insertion_rate_iter = self.insertion_rate  # fixed
            
#         npz_files = image_files[local_index:local_index + insertion_rate_iter + 1]
#         first_frame_path = npz_files[0]["image_path"]
#         last_frame_path = npz_files[-1]["image_path"]

#         first_frame, _, _, _, _, _ = self.load_npz(first_frame_path)
#         last_frame, _, _, _, _, _ = self.load_npz(last_frame_path)

#         # insert_position = random.randint(1, insertion_rate_iter - 1)
#         x_random = random.random() 
#         y_random = random.random() 
#         first_frame = self.crop_image(first_frame, self.crop_type, x_random ,y_random)
#         last_frame = self.crop_image(last_frame, self.crop_type, x_random ,y_random)
        
#         gt_frames = []
#         for insert_position in range(1, insertion_rate_iter):
#             gt_frame_path = npz_files[insert_position]["image_path"]
#             gt_frame, _, _, _, _, _ = self.load_npz(gt_frame_path)
#             gt_frame = self.crop_image(gt_frame, self.crop_type, x_random ,y_random)
#             gt_frame = (gt_frame[..., ::-1].astype(np.float32) / 255.0) * 2 - 1 
#             gt_frames.append(gt_frame)
        
#         events = self.accumulate_events(local_index, local_index + insertion_rate_iter, image_files, insertion_rate_iter)
#         events = self.crop_image(events, self.crop_type, x_random ,y_random)

#         first_frame = (first_frame[..., ::-1].astype(np.float32) / 255.0)  # BGR to RGB and normalize
#         last_frame = (last_frame[..., ::-1].astype(np.float32) / 255.0)    # BGR to RGB and normalize
#         events = (events.astype(np.float32) / 255.0)  # BGR to RGB and normalize

#         video_frames = np.stack(gt_frames, axis=0)
#         pixel_values = torch.from_numpy(video_frames.transpose(0, 3, 1, 2))
#         pixel_values = self.pixel_transforms(pixel_values)
#         conditions = pixel_values[-1]
#         if self.is_reverse_video:
#             pixel_values = torch.flip(pixel_values, (0,))

#         prompt = ""
#         return {
#             "conditions": first_frame,
#             "conditions_rev": last_frame,
#             "pixel_values": pixel_values,
#             "evs": events,
#             "insertion_rate": insertion_rate_iter,
#             "prompt": prompt}

#     def __len__(self):
#         return sum(self.subfolder_lengths)






# class EventVideoTestDataset(EventVideoDataset):
#     def __init__(self, file_list: str, out_size: int, crop_type: str, insertion_rate: int):
#         # Initialize the parent class (EventRGBDataset) with common parameters
#         super(EventVideoTestDataset, self).__init__(file_list, out_size, crop_type, insertion_rate=insertion_rate)
#         # self.skip_rate = 8
#         # self.skip_rate = 4

#     def __getitem__(self, index: int):
#         """
#         Overrides the __getitem__ method to handle testing-specific behavior.
#         Reuses much of the logic from the parent class.
#         """
#         adjusted_index = index * self.insertion_rate  
#         # Get the subfolder and local index from the global index
#         subfolder, local_index = self._get_subfolder_and_local_index(adjusted_index)
#         image_files = self.subfolder_files[subfolder]

#         # Extract the sequence of frames based on the local index
#         npz_files = image_files[local_index:local_index + self.insertion_rate + 1]
#         first_frame_path = npz_files[0]["image_path"]
#         last_frame_path = npz_files[-1]["image_path"]

#         # Load first and last frames
#         first_frame, _, _, _, _, _ = self.load_npz(first_frame_path)
#         last_frame, _, _, _, _, _ = self.load_npz(last_frame_path)
#         first_frame = (first_frame[..., ::-1].astype(np.float32) / 255.0)
#         last_frame = (last_frame[..., ::-1].astype(np.float32) / 255.0)
        
#         events = self.accumulate_events(local_index, local_index + self.insertion_rate, image_files, self.insertion_rate)
#         events = (events.astype(np.float32) / 255.0)  # BGR to RGB and normalize
#         # first_frame = self.crop_image(first_frame, self.crop_type)
#         # last_frame = self.crop_image(last_frame, self.crop_type)

#         res =[{"f0": first_frame, 
#                "f1": last_frame,
#                "insertion_rate" : self.insertion_rate ,
#                "path0" : first_frame_path, 
#                "path1" : last_frame_path,
#                "evs": events
#                }]
#         # Set a fixed insert position for reproducibility in testing
#         for i in range(1,self.insertion_rate):
#             # if i % self.skip_rate !=0: continue 
            
#             insert_position = i  
#             gt_frame_path = npz_files[insert_position]["image_path"]
#             gt_frame, _, _, _, _, _ = self.load_npz(gt_frame_path)

#             # # Accumulate events for 0 to t and t to 1
#             # event_0_to_t = self.accumulate_events(local_index, local_index + insert_position, image_files)
#             # event_t_to_1 = self.accumulate_events(local_index + insert_position, local_index + self.insertion_rate, image_files)

#             # # Normalize frames (BGR to RGB and rescale)
#             # event_0_to_t = (event_0_to_t[..., ::-1].astype(np.float32) / 255.0)
#             # event_t_to_1 = (event_t_to_1[..., ::-1].astype(np.float32) / 255.0)
#             gt_frame = (gt_frame[..., ::-1].astype(np.float32) / 255.0) * 2 - 1  # Normalize to [-1, 1]
            
#             # event_0_to_t = self.crop_image(event_0_to_t, self.crop_type)
#             # event_t_to_1 = self.crop_image(event_t_to_1, self.crop_type)
#             # gt_frame = self.crop_image(gt_frame, self.crop_type)

#             # Return the data dictionary
#             prompt = ""  # No specific prompt required for testing
#             frame_i =  {
#                 # "f0": first_frame,
#                 # "f1": last_frame,
#                 "gt": gt_frame,
#                 # "evs0t": event_0_to_t,
#                 # "evst1": event_t_to_1,
#                 "ind": insert_position,
#                 "prompt": prompt,
#                 "gt_path":gt_frame_path 
#             }
#             res.append(frame_i)
#         return res
    
#     def __len__(self):
#         # Length of dataset considering insertion_rate as step size
#         total_frames = sum(self.subfolder_lengths)
#         return total_frames // self.insertion_rate










# # import torch
# # from torch.utils.data import DataLoader
# # import os 
# # from einops import rearrange
# # # file_list = "/ailab/user/zhangziran/Self_collected_DATASET/all_evs_rgb/" #"/ailab/user/zhangziran/diffuser/DiffBIR/file_list.txt"
# # # file_list = "/ailab/group/pjlab-sail/mayongrui/dataset/New_Captured_2"
# # file_list = "/mnt/workspace/zhangziran/Self_collected_DATASET/test_case_compare_diffusion"
# # out_size = 512
# # crop_type = "center"

# # # 实例化数据集
# # dataset =  EventRGBTestDataset( #EventRGBDataset(
# #     file_list=file_list,
# #     out_size=out_size,
# #     crop_type=crop_type,
# #     insertion_rate = 32
# # )


# # # # 实例化数据集
# # # dataset = EventVideoDataset( #EventRGBDataset(
# # #     file_list=file_list,
# # #     out_size=out_size,
# # #     crop_type=crop_type,
# # #     insertion_rate = 8
# # # )

# # # 创建 DataLoader
# # dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)

# # # 从 DataLoader 中读取一个批次的数据
# # output_dir = "/mnt/workspace/zhangziran/DiffEVS/results/test_case_dataloader"  # 输出路径，修改为你希望保存文件的路径
# # os.makedirs(output_dir, exist_ok=True)

# # device = "cuda"
# # for i, batch in enumerate(dataloader):
# #     print(i)
# #     # f0, f1, gts, evs, insertion_rate, prompt = batch["f0"], batch["f1"], batch["gt"], batch["evs"], batch["insertion_rate"], batch["prompt"]     
# #     # f0 = rearrange(f0, "b h w c -> b c h w").contiguous().float().to(device)
# #     # f1 = rearrange(f1, "b h w c -> b c h w").contiguous().float().to(device)
# #     # gt_merged = torch.stack(gts, dim=1)
# #     # gt = rearrange(gt_merged, "b t h w c -> (b t) c h w").contiguous().float().to(device)
# #     # evs = rearrange(evs, "b h w c -> b c h w").contiguous().float().to(device)
    
# #     # print("f0:",f0.shape,"f1:",f1.shape,"gt_merged:",gt_merged.shape ,"gt:",gt.shape,"evs:",evs.shape)
# #     # for datai in batch:
# #     #     i += 1 
# #     #     if i == 1:
# #     #         print(datai["path0"],datai["path1"])
# #     #     else:
# #     #         print(datai["gt_path"])

        
# #     # f0 = batch["f0"]
# #     # print(type(f0),batch["ind"])
# #     # break
    
# #     # 只读取一个批次
# #     # break