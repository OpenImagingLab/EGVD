
# CUDA_VISIBLE_DEVICES=1
#!bin/bash
noise_injection_steps=5
noise_injection_ratio=0.5
EVAL_DIR=example1
# CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/svd_reverse_motion_with_attnflip
CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/ckpt/checkpoints_wnorm_wofilter_v5/svd_reverse_motion_with_attnflip/checkpoint-20000
# CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/ckpt/checkpoints_wnorm_v2/svd_reverse_motion_with_attnflip/checkpoint-10000
# MODEL_NAME=stabilityai/stable-video-diffusion-img2vid-xt
MODEL_NAME=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/stable-video-diffusion-img2vid
OUT_DIR=my_results/r20250129_large_resolution_evs_wnorm_wofilter_v5_20000

# mkdir -p $OUT_DIR
# for example_dir in $(ls -d $EVAL_DIR/*)
# do
#     example_name=$(basename $example_dir)
#     echo $example_name

out_fn=$OUT_DIR/'example.gif'
python evs_keyframe_interpolation.py \
    --frames_dirs="/mnt/workspace/zhangziran/Dataset/Gopro/gopro_v2e" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --noise_injection_steps=$noise_injection_steps \
    --noise_injection_ratio=$noise_injection_ratio \
    --out_path=$out_fn     \
    --event_filter="None"
