
train_model() {
    local TASK=$1
    local CONFIG=$2
    local CUDA_DEVICE=$3
    local ALGORITHM="ERM"
    local DATASET=$TASK
    local HPARAMS='{"backbone": "'$CONFIG'", "lr": 1e-4}'
    local OUTPUT_DIR="./checkpoint/${TASK}/${CONFIG}/"

    echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

    echo "Training task: $TASK with model: $CONFIG"
    python -m domainbed.scripts.train \
        --data_dir=./datasets/ \
        --output_dir=$OUTPUT_DIR \
        --algorithm=$ALGORITHM \
        --dataset=$DATASET \
        --hparams="$HPARAMS" \
        --trial_seed=0 \
        --train_envs=0

    echo "Finished training task: $TASK with model: $CONFIG"
}

train_model "MultipleEnvironmentDR" "resnet50-BN" 0 

echo "All tasks and configurations have been trained."
