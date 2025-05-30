

# CUDA_VISIBLE_DEVICES=1
#!bin/bash
noise_injection_steps=5
noise_injection_ratio=0.5
EVAL_DIR=example1

# CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/ckpt/checkpoints_wnorm_wofilter_v7_1/svd_reverse_motion_with_attnflip/checkpoint-30000
CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/ckpt/checkpoints_wnorm_wofilter_v8_color3_mixskip2/svd_reverse_motion_with_attnflip/checkpoint-10000
MODEL_NAME=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/stable-video-diffusion-img2vid
OUT_DIR=my_results/r20250212_checkpoints_wnorm_wofilter_v8_color3_mixskip2_skip3

# mkdir -p $OUT_DIR
# for example_dir in $(ls -d $EVAL_DIR/*)
# do
#     example_name=$(basename $example_dir)
#     echo $example_name

out_fn=$OUT_DIR/'example.gif'
python evs_keyframe_interpolation_color.py \
    --frames_dirs="/mnt/workspace/zhangziran/Self_collected_DATASET/train_sd_vfi/test" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --noise_injection_steps=$noise_injection_steps \
    --noise_injection_ratio=$noise_injection_ratio \
    --out_path=$out_fn   \
    --skip_sampling_rate=3 \
    --event_filter="None"
        # --event_filter="great_filter" 

OUT_DIR=my_results/r20250212_checkpoints_wnorm_wofilter_v8_color3_mixskip2_skip2
out_fn=$OUT_DIR/'example.gif'
python evs_keyframe_interpolation_color.py \
    --frames_dirs="/mnt/workspace/zhangziran/Self_collected_DATASET/train_sd_vfi/test" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --noise_injection_steps=$noise_injection_steps \
    --noise_injection_ratio=$noise_injection_ratio \
    --out_path=$out_fn   \
    --skip_sampling_rate=2 \
    --event_filter="None"

OUT_DIR=my_results/r20250212_checkpoints_wnorm_wofilter_v8_color3_mixskip2_skip1
out_fn=$OUT_DIR/'example.gif'
python evs_keyframe_interpolation_color.py \
    --frames_dirs="/mnt/workspace/zhangziran/Self_collected_DATASET/train_sd_vfi/test" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --noise_injection_steps=$noise_injection_steps \
    --noise_injection_ratio=$noise_injection_ratio \
    --out_path=$out_fn   \
    --skip_sampling_rate=1 \
    --event_filter="None"



