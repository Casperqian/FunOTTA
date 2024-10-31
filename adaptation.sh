export CUDA_VISIBLE_DEVICES=0

INPUT_DIR="./checkpoint/MultipleEnvironmentDR/resnet18-BN/trial_seed-0/"
ADAPT_ALGORITHMS=("TentClf" "PLClf" "SHOT" "T3A" "TAST" "UniDG" "DeYOClf" "Ours")

for algorithm in "${ADAPT_ALGORITHMS[@]}"; do
    echo "Running $algorithm..."
    python -m domainbed.scripts.unsupervised_adaptation \
        --input_dir=$INPUT_DIR \
        --adapt_algorithm=$algorithm
done

echo "All algorithms have been run."
