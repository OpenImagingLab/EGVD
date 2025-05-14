cd /mnt/workspace/zhangziran/
source .bashrc
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/mnt/workspace/zhangziran/Anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/workspace/zhangziran/Anaconda/etc/profile.d/conda.sh" ]; then
        . "/mnt/workspace/zhangziran/Anaconda/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/workspace/zhangziran/Anaconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


conda activate SVD
mkdir /root/.cache/huggingface/
cd /mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main

cp -r /mnt/workspace/zhangziran/.cache/huggingface/accelerate /root/.cache/huggingface/.
# MODEL_NAME=stabilityai/stable-video-diffusion-img2vid
MODEL_NAME=/mnt/workspace/zhangziran/DiffEVS/svd_keyframe_interpolation-main/checkpoints/stable-video-diffusion-img2vid
TRAIN_DIR=/mnt/workspace/zhangziran/Self_collected_DATASET/train_sd_vfi/train/ #../keyframe_interpolation_data/synthetic_videos_frames
VALIDATION_DIR=None
# python train_reverse_motion_with_attnflip_1gpu.py \
accelerate launch --mixed_precision="fp16" evs_train_single_svd3_color_dlc_wf_lownoise_ab_mmf.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --variant "fp16" \
  --num_frames 9 \
  --train_data_dir=$TRAIN_DIR \
  --validation_data_dir=$VALIDATION_DIR \
  --max_train_samples=400 \
  --train_batch_size=2 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs=1200 --checkpointing_steps=30000 \
  --validation_epochs=50 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --skip_sampling_rate=-1 \
  --output_dir="ckpt/checkpoints_z0226_abmmf/svd_reverse_motion_with_attnflip" \
  --cache_dir="ckpt/checkpoints_z0226_abmmf/svd_reverse_motion_with_attnflip_cache" \
  --report_to="tensorboard" \
  --event_filter="great_filter"
    # --event_filter="None"
    # --event_filter="None" 
# --num_frames 14 \