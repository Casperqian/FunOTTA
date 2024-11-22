# Function to adapt the source model with different algorithms

adaptation() {
    local TASK=$1
    local CONFIG=$2
    local CUDA_DEVICE=$3
    local INPUT_DIR="./checkpoint/${TASK}/${CONFIG}/trial_seed-0/"
    local ADAPT_ALGORITHMS=("TentClf" "PLClf" "SHOT" "T3A" "TAST" "UniDG" "DeYOClf" "Ours")

    echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

    echo "Running task: $TASK with configuration: $CONFIG"
    for algorithm in "${ADAPT_ALGORITHMS[@]}"; do
        echo "Running $algorithm..."
        python -m domainbed.scripts.unsupervised_adaptation \
            --input_dir=$INPUT_DIR \
            --adapt_algorithm=$algorithm
    done
    echo "Finished running task: $TASK with configuration: $CONFIG"
}

adaptation "MultipleEnvironmentDR" "resnet50-BN" 0  

echo "All tasks, configurations, and algorithms have been run."
