#!/bin/bash

# ===============================================================
# Usage:
#   bash sft_7b.sh
#
# Note:
#   You can set the model path via environment variable:
#   MODEL_PATH=/path/to/your/model bash sft_7b.sh
# ===============================================================

# --- Distributed Training Configuration ---
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
# NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l) # Auto-detect
NPROC_PER_NODE=1  # Hardcoded as per original config

# --- DeepSpeed Configuration ---
DEEPSPEED_CONFIG=./scripts/zero3.json

# --- Model Configuration ---
# Uses HuggingFace ID by default, or override with environment variable
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-7B-Instruct"}

# --- Training Hyperparameters (Do Not Change) ---
lr=2e-7
batch_size=2
grad_accum_steps=1

# --- Entry Point ---
ENTRY_FILE=qwenvl/train/train_qwen.py

# --- Dataset Configuration ---
# Ensure these datasets are registered in your dataloader
DATASETS="youcook2,epickitchen,egoprocel,ego4d,visor，egoit"

# --- Output Configuration ---
RUN_NAME="qwen2vl-baseline"
OUTPUT_DIR=${OUTPUT_DIR:-"./output/${RUN_NAME}"}
mkdir -p ${OUTPUT_DIR}

# --- Arguments Construction ---
# Pixel configurations: 1080*28*28; 4*28*28; 384*28*28; 64*28*28

ARGS="
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_use ${DATASETS} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 846720 \
    --min_pixels 3136 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 131072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${RUN_NAME} \
    --report_to wandb \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --video_max_frames 768 \
    --video_min_frames 4"

# Note: The following args are commented out in original config
# --video_max_frame_pixels 301056
# --video_min_frame_pixels 50176

# --- Launch Training ---
echo "Starting training with Model: ${MODEL_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${ENTRY_FILE} ${ARGS}