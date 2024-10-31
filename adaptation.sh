export CUDA_VISIBLE_DEVICES=2

INPUT_DIR="./checkpoint/MultipleEnvironmentDR/resnet18-BN/trial_seed-0/"
ADAPT_ALGORITHMS=("Ours")

for algorithm in "${ADAPT_ALGORITHMS[@]}"; do
    echo "Running $algorithm..."
    python -m domainbed.scripts.unsupervised_adaptation \
        --input_dir=$INPUT_DIR \
        --adapt_algorithm=$algorithm
done

echo "All algorithms have been run."
