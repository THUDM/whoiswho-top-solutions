#! /usr/bin/env bash

set -ex

LR=3e-5
NUM_GPUS=8
LORA_RANK=128
LORA_ALPHA=256
LORA_DROUPOUT=0.1

MAX_SOURCE_LEN=16000
MAX_TARGET_LEN=128
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=16
EPOCH=2
SAVE_INTERVAL=250
WARMUP_RATIO=0.1
SCHEDULAR=cosine

RUN_NAME=text
BASE_MODEL_PATH=your_path_to_chatglm/chatglm3-6b-32k
PUB_PATH=path_to_pub
TRAIN_PATH=path_to_train_author
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)


mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS  finetune.py \
    --train_format input-output \
    --pub_data $PUB_PATH \
    --train_data $TRAIN_PATH \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROUPOUT \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size  $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --num_train_epochs $EPOCH \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --bf16 \
    --deepspeed configs/deepspeed.json  2>&1 | tee ${OUTPUT_DIR}/train.log

