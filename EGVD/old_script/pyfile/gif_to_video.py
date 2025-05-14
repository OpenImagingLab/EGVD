import os
from moviepy.editor import VideoFileClip

# 定义GIF所在的文件夹路径
folder_path = '/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/results'

# 获取文件夹下的所有GIF文件
gif_files = [f for f in os.listdir(folder_path) if f.endswith('.gif')]

# 转换每个GIF文件
for gif_file in gif_files:
    input_gif_path = os.path.join(folder_path, gif_file)
    output_video_path = os.path.join(folder_path, f"{os.path.splitext(gif_file)[0]}_turbulence.mp4")
    
    # 加载GIF并设置fps（帧率）
    clip = VideoFileClip(input_gif_path)
    clip.write_videofile(output_video_path, fps=15)  # 设置合适的帧率

    print(f"Converted {gif_file} to {output_video_path}")
