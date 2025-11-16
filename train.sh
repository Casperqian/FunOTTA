#!/bin/bash

train_model() {
    local TASK=$1
    local CONFIG=$2
    local CUDA_DEVICE=$3
    local SEED=$4
    local ALGORITHM="ERM"
    local DATASET=$TASK
    local HPARAMS='{"backbone": "'$CONFIG'", "lr": 1e-4}'
    local OUTPUT_DIR="./checkpoint/${TASK}/${CONFIG}"

    echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

    echo "Training task: $TASK | model: $CONFIG | seed: $SEED"

    local LOG_FILE="./logs/${TASK}_${CONFIG}_seed${SEED}.log"
    mkdir -p "$(dirname "$LOG_FILE")"

    nohup python -m domainbed.scripts.train \
        --data_dir=/change/to/your/path/ \
        --output_dir=$OUTPUT_DIR \
        --algorithm=$ALGORITHM \
        --dataset=$DATASET \
        --hparams="$HPARAMS" \
        --trial_seed=$SEED \
        --train_envs=0 \
        > "$LOG_FILE" 2>&1 &

    echo "Started background job for: $TASK | $CONFIG | seed: $SEED → log: $LOG_FILE"
}


TASK_LIST=(
    # "MultipleEnvironmentDR"
    # "Glaucoma"
)

CONFIG_LIST=(
    # "resnet50-BN"
    "resnet18-BN"
)

SEEDS=(0 1 2 3 4)   
CUDA_DEVICE=0

for TASK in "${TASK_LIST[@]}"; do
    for CONFIG in "${CONFIG_LIST[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            train_model "$TASK" "$CONFIG" "$CUDA_DEVICE" "$SEED"
            sleep 20   
        done
    done
done

echo "All jobs have been started in background."
