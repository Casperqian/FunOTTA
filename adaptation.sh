#!/bin/bash
set -u

declare -A INPUT_BASE_DIRS

CONFIG_GLAUCOMA_R18="glaucoma_resnet18"
INPUT_BASE_DIRS[$CONFIG_GLAUCOMA_R18]="./checkpoint/Glaucoma/resnet18-BN/"

CONFIG_GLAUCOMA_R50="glaucoma_resnet50"
INPUT_BASE_DIRS[$CONFIG_GLAUCOMA_R50]="./checkpoint/Glaucoma/resnet50-BN/"

CONFIG_DR_R18="diabetic_retinopathy_resnet18"
INPUT_BASE_DIRS[$CONFIG_DR_R18]="./checkpoint/MultipleEnvironmentDR/resnet18-BN/"

CONFIG_LSDR_R18="label_shift_diabetic_retinopathy_resnet18"
INPUT_BASE_DIRS[$CONFIG_LSDR_R18]="./checkpoint/LabelShiftDR/resnet18-BN/"

CONFIG_DR_R50="diabetic_retinopathy_resnet50"
INPUT_BASE_DIRS[$CONFIG_DR_R50]="./checkpoint/MultipleEnvironmentDR/resnet50-BN/"

CONFIG_LSDR_R50="label_shift_diabetic_retinopathy_resnet50"
INPUT_BASE_DIRS[$CONFIG_LSDR_R50]="./checkpoint/LabelShiftDR/resnet50-BN/"

CONFIG_LSDR_R18="label_shift_diabetic_retinopathy_resnet18"
INPUT_BASE_DIRS[$CONFIG_LSDR_R18]="./checkpoint/LabelShiftDR/resnet18-BN/"

CONFIG_FETAL8_R50="fetal-8_resnet50"
INPUT_BASE_DIRS[$CONFIG_FETAL8_R50]="./checkpoint/Fetal8/resnet50-BN/"

CONFIG_FETAL8R_R50="fetal-8r_resnet50"
INPUT_BASE_DIRS[$CONFIG_FETAL8R_R50]="./checkpoint/Fetal8-R/resnet50-BN/"

DATA_DIR="/to/your/path/"
LOG_ROOT="./logs"

ADAPT_ALGORITHMS=("Ours")

SEEDS=(0)


detect_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
  else
    echo 1
  fi
}
GPU_COUNT=$(detect_gpu_count)
MAX_PARALLEL=2
MAX_PARALLEL="${MAX_PARALLEL:-$GPU_COUNT}"

timestamp() { date +"%Y%m%d-%H%M%S"; }

# 动态找内存占用低于阈值的GPU，返回当前占用最少的GPU
get_free_gpu() {
  local threshold_mb=${1:-1000}
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  local candidates=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
                      awk -v th="$threshold_mb" '{if($2<=th) print $1}'))
  if [ ${#candidates[@]} -gt 0 ]; then
    echo "${candidates[0]}"
  else
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -n1 | awk '{print $1}'
  fi
}

running_jobs() { jobs -pr | wc -l; }

wait_for_slot() {
  local LIMIT="$1"
  while [ "$(running_jobs)" -ge "$LIMIT" ]; do
    sleep 5
  done
}

trap 'echo -e "\n[Trap] Caught SIGINT, killing children..."; jobs -pr | xargs -r kill; exit 130' INT

launch_job() {
  local GPU_ID="$1"
  local INPUT_DIR="$2"
  local ALGO="$3"
  local CONFIG_NAME="$4"
  local SEED="$5"

  local TSTMP
  TSTMP="$(timestamp)"
  local LOG_DIR="${LOG_ROOT}/${CONFIG_NAME}/seed-${SEED}"
  mkdir -p "$LOG_DIR"
  local LOG_FILE="${LOG_DIR}/${ALGO}_${TSTMP}.log"

  echo "[Submit] ${CONFIG_NAME} | seed=${SEED} | ${ALGO} -> GPU ${GPU_ID}"
  echo "         input_dir=${INPUT_DIR}"
  echo "         log=${LOG_FILE}"

  (
    echo "==== $(date) | START ${CONFIG_NAME} | seed=${SEED} | ${ALGO} | GPU=${GPU_ID} ===="
    echo "INPUT_DIR=${INPUT_DIR}"
    echo "DATA_DIR=${DATA_DIR}"
    echo

    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    python -m domainbed.scripts.unsupervised_adaptation \
      --input_dir="${INPUT_DIR}" \
      --data_dir="${DATA_DIR}" \
      --adapt_algorithm="${ALGO}" \
      --epoch 1 \
      --evaluate

    EXIT_CODE=$?
    echo
    echo "==== $(date) | END ${CONFIG_NAME} | seed=${SEED} | ${ALGO} | GPU=${GPU_ID} | exit=${EXIT_CODE} ===="
    exit ${EXIT_CODE}
  ) > "${LOG_FILE}" 2>&1 &
}

run_adaptation() {
  local CONFIG_NAME="$1"
  local BASE_DIR="${INPUT_BASE_DIRS[$CONFIG_NAME]}"

  if [ -z "${BASE_DIR:-}" ]; then
    echo "[Error] Unknown CONFIG_NAME: $CONFIG_NAME"
    return 1
  fi

  for SEED in "${SEEDS[@]}"; do
    local INPUT_DIR="${BASE_DIR}trial_seed-${SEED}/"
    if [ ! -d "$INPUT_DIR" ]; then
      echo "[Warn] INPUT_DIR not found: $INPUT_DIR (skip)"
      continue
    fi

    for ALGO in "${ADAPT_ALGORITHMS[@]}"; do
      wait_for_slot "$MAX_PARALLEL"
      GPU_ID=$(get_free_gpu 1000)
      launch_job "$GPU_ID" "$INPUT_DIR" "$ALGO" "$CONFIG_NAME" "$SEED"
      sleep 10
    done
  done
}

# run_adaptation "$CONFIG_GLAUCOMA_R18"
# run_adaptation "$CONFIG_GLAUCOMA_R50"
# run_adaptation "$CONFIG_DR_R18"
# run_adaptation "$CONFIG_DR_R50"
# run_adaptation "$CONFIG_FETAL8_R50"
# run_adaptation "$CONFIG_FETAL8R_R50"
# run_adaptation "$CONFIG_LSDR_R50"
# run_adaptation "$CONFIG_LSDR_R18"

echo "[Main] All jobs submitted. Waiting for completion..."
wait 
echo "[Main] All adaptation runs finished."
