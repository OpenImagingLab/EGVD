# CUDA_VISIBLE_DEVICES=1
#!bin/bash
noise_injection_steps=5
noise_injection_ratio=0.5
EVAL_DIR=example1
# CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/svd_reverse_motion_with_attnflip
CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/ckpt/checkpoints_single_svd_1embedding/svd_reverse_motion_with_attnflip/checkpoint-26000
# CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/ckpt/checkpoints_single_svd_1embedding/svd_reverse_motion_with_attnflip/checkpoint-11000
# CHECKPOINT_DIR=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/ckpt/checkpoints_new_evs2/svd_reverse_motion_with_attnflip/checkpoint-57500
# MODEL_NAME=stabilityai/stable-video-diffusion-img2vid-xt
MODEL_NAME=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/stable-video-diffusion-img2vid
OUT_DIR=my_results/r20250122_large_resolution_evs_1embedding_26000

# mkdir -p $OUT_DIR
# for example_dir in $(ls -d $EVAL_DIR/*)
# do
#     example_name=$(basename $example_dir)
#     echo $example_name

out_fn=$OUT_DIR/'example.gif'
python evs_1embedding_keyframe_interpolation.py \
    --frames_dirs="/mnt/workspace/zhangziran/Self_collected_DATASET/train_sd_vfi/test" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --noise_injection_steps=$noise_injection_steps \
    --noise_injection_ratio=$noise_injection_ratio \
    --out_path=$out_fn        
        
        # "/mnt/workspace/zhangziran/Dataset/bs_ergb/convert_test"
        # --frame1_path=$example_dir/frame1.png \
        # --frame2_path=$example_dir/frame2.png \
#     break 
# done


out_fn=$OUT_DIR/'example.gif'
python evs_1embedding_keyframe_interpolation.py \
    --frames_dirs="/mnt/workspace/zhangziran/Dataset/bs_ergb/convert_test" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --noise_injection_steps=$noise_injection_steps \
    --noise_injection_ratio=$noise_injection_ratio \
    --out_path=$out_fn        
        
        # "/mnt/workspace/zhangziran/Dataset/bs_ergb/convert_test"
        # --frame1_path=$example_dir/frame1.png \
        # --frame2_path=$example_dir/frame2.png \
#     break 
# done