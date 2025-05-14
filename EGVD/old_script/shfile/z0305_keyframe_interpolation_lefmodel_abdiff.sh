

# CUDA_VISIBLE_DEVICES=1
#!bin/bash
noise_injection_steps=5
noise_injection_ratio=0.5
EVAL_DIR=example1


CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/ckpt/checkpoints_z0220_newfusemodel_lownoise/svd_reverse_motion_with_attnflip/checkpoint-60000
MODEL_NAME=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/stable-video-diffusion-img2vid


OUT_DIR=my_results2/r20250305_debug5_lefmodel_60000_skip3
out_fn=$OUT_DIR/'example.gif'
python z0305_evs_keyframe_interpolation_color_fuse_lefmodel.py \
    --frames_dirs="/mnt/workspace/zhangziran/Self_collected_DATASET/train_sd_vfi/test" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --noise_injection_steps=$noise_injection_steps \
    --noise_injection_ratio=$noise_injection_ratio \
    --out_path=$out_fn   \
    --skip_sampling_rate=3 \
    --event_filter="great_filter"


OUT_DIR=my_results2/r20250305_debug5_lefmodel_60000_skip2
out_fn=$OUT_DIR/'example.gif'
python z0305_evs_keyframe_interpolation_color_fuse_lefmodel.py \
    --frames_dirs="/mnt/workspace/zhangziran/Self_collected_DATASET/train_sd_vfi/test" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --noise_injection_steps=$noise_injection_steps \
    --noise_injection_ratio=$noise_injection_ratio \
    --out_path=$out_fn   \
    --skip_sampling_rate=2 \
    --event_filter="great_filter"

OUT_DIR=my_results2/r20250305_debug5_lefmodel_60000_skip1
out_fn=$OUT_DIR/'example.gif'
python z0305_evs_keyframe_interpolation_color_fuse_lefmodel.py \
    --frames_dirs="/mnt/workspace/zhangziran/Self_collected_DATASET/train_sd_vfi/test" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --noise_injection_steps=$noise_injection_steps \
    --noise_injection_ratio=$noise_injection_ratio \
    --out_path=$out_fn   \
    --skip_sampling_rate=1 \
    --event_filter="great_filter"