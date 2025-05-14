

# CUDA_VISIBLE_DEVICES=1
#!bin/bash
noise_injection_steps=5
noise_injection_ratio=0.5
EVAL_DIR=example1


CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/svd_reverse_motion_with_attnflip
# CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/stable-video-diffusion-img2vid
MODEL_NAME=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/stable-video-diffusion-img2vid




# OUT_DIR=my_results/r20250222_2SVD_resize_skip3r
# out_fn=$OUT_DIR/'example.gif'
# python keyframe_interpolation_2svd_resize.py \
#     --frames_dirs="/mnt/workspace/zhangziran/Self_collected_DATASET/train_sd_vfi/test" \
#     --pretrained_model_name_or_path=$MODEL_NAME \
#     --checkpoint_dir=$CHECKPOINT_DIR \
#     --noise_injection_steps=$noise_injection_steps \
#     --noise_injection_ratio=$noise_injection_ratio \
#     --out_path=$out_fn   \
#     --skip_sampling_rate=3 \
#     --event_filter="None" 
#     # --event_filter="great_filter"
#         # --event_filter="None" 



# OUT_DIR=my_results/r20250222_2SVD_resize_skip2r
# out_fn=$OUT_DIR/'example.gif'
# python keyframe_interpolation_2svd_resize.py \
#     --frames_dirs="/mnt/workspace/zhangziran/Self_collected_DATASET/train_sd_vfi/test" \
#     --pretrained_model_name_or_path=$MODEL_NAME \
#     --checkpoint_dir=$CHECKPOINT_DIR \
#     --noise_injection_steps=$noise_injection_steps \
#     --noise_injection_ratio=$noise_injection_ratio \
#     --out_path=$out_fn   \
#     --skip_sampling_rate=2 \
#     --event_filter="None" 
#     # --event_filter="great_filter"
#         # --event_filter="None" 


OUT_DIR=my_results/r20250222_2SVD_resize_skip1r
out_fn=$OUT_DIR/'example.gif'
python keyframe_interpolation_2svd_resize.py \
    --frames_dirs="/mnt/workspace/zhangziran/Self_collected_DATASET/train_sd_vfi/test" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --noise_injection_steps=$noise_injection_steps \
    --noise_injection_ratio=$noise_injection_ratio \
    --out_path=$out_fn   \
    --skip_sampling_rate=1 \
    --event_filter="None" 
    # --event_filter="great_filter"
        # --event_filter="None" 
