#!/bin/bash

declare -A INPUT_DIRS

CONFIG_GLAUCOMA_R18="glaucoma_resnet18"
INPUT_DIRS[$CONFIG_GLAUCOMA_R18]="./checkpoint/Glaucoma/resnet18-BN/trial_seed-0/"

CONFIG_GLAUCOMA_R50="glaucoma_resnet50"
INPUT_DIRS[$CONFIG_GLAUCOMA_R50]="./checkpoint/Glaucoma/resnet50-BN/trial_seed-0/"

CONFIG_DR_R18="diabetic_retinopathy_resnet18"
INPUT_DIRS[$CONFIG_DR_R18]="./checkpoint/MultipleEnvironmentDR/resnet18-BN/trial_seed-0/"

CONFIG_DR_R50="diabetic_retinopathy_resnet50"
INPUT_DIRS[$CONFIG_DR_R50]="./checkpoint/MultipleEnvironmentDR/resnet50-BN/trial_seed-0/"

CONFIG_FETAL8_R50="featal-8_resnet50"
INPUT_DIRS[$CONFIG_FETAL8_R50]="./checkpoint/Fetal8/resnet50-BN/trial_seed-0/"

DATA_DIR="/data2/zengqian/GTTA/datasets/"

run_model() {
    local CONFIG_NAME=$1
    local GPU_ID=$2
    local INPUT_DIR="${INPUT_DIRS[$CONFIG_NAME]}"
    
    echo "Running Finetuning for config: $CONFIG_NAME on GPU $GPU_ID"
    
    for epoch in 5 10
    do
        for freeze in "" "--freeze_extractor"
        do
            echo ">>> Running $CONFIG_NAME"
            
            CUDA_VISIBLE_DEVICES=$GPU_ID python -m domainbed.scripts.finetune \
                --input_dir=$INPUT_DIR \
                --data_dir=$DATA_DIR \
                --ft_batch_size 32 \
                --epoch $epoch \
                $freeze
        done
    done
}

# run_model "$CONFIG_GLAUCOMA_R18" "0"
# run_model "$CONFIG_GLAUCOMA_R50" "0"
# run_model "$CONFIG_DR_R18" "3"
# run_model "$CONFIG_DR_R50" "3"
run_model "$CONFIG_FETAL8_R50" "3"

echo "All runs finished."
