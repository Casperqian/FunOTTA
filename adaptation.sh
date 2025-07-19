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

CONFIG_FETAL8R_R50="featal-8r_resnet50"
INPUT_DIRS[$CONFIG_FETAL8R_R50]="./checkpoint/Fetal8-R/resnet50-BN/trial_seed-0/"

DATA_DIR="/data2/zengqian/GTTA/datasets/"

ADAPT_ALGORITHMS=("TentClf" "PLClf" "SHOT" "T3A" "TAST" "UniDG" "DeYOClf" "SAR" "SARClf" "EATA" "EATAClf" "Ours")
# ADAPT_ALGORITHMS=("DeYOClf")

run_adaptation() {
    local CONFIG_NAME=$1
    local GPU_ID=$2
    local INPUT_DIR="${INPUT_DIRS[$CONFIG_NAME]}"

    echo "Running unsupervised adaptation for config: $CONFIG_NAME on GPU $GPU_ID"

    for algorithm in "${ADAPT_ALGORITHMS[@]}"; do
        echo ">>> Running algorithm: $algorithm"

        CUDA_VISIBLE_DEVICES=$GPU_ID python -m domainbed.scripts.unsupervised_adaptation \
            --input_dir="$INPUT_DIR" \
            --data_dir="$DATA_DIR" \
            --adapt_algorithm="$algorithm" \
            --epoch 1 \
            --evaluate

        echo ">>> Finished: $algorithm"
    done
}

# run_adaptation "$CONFIG_GLAUCOMA_R18" "3"
# run_adaptation "$CONFIG_GLAUCOMA_R50" "3"
# run_adaptation "$CONFIG_DR_R18" "1"
# run_adaptation "$CONFIG_DR_R50" "1"
# run_adaptation "$CONFIG_FETAL8_R50" "1"
run_adaptation "$CONFIG_FETAL8R_R50" "1"

echo "All adaptation runs finished."
