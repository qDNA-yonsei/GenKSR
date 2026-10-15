#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PY:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/kqd2d_1000shot_${STAMP}}"
mkdir -p "${LOG_DIR}"

TF_RUN="results/conditional_heisenberg_KQD_2d_16qubit_pennylane/gcn_encoder_3_256-transformer_l8_d256_h8_featone_hot_D_conditional/ns1000/iter400_lr0.0005_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine"
MAMBA_RUN="results/conditional_heisenberg_KQD_2d_16qubit_pennylane/gcn_encoder_3_256-mamba_l8_d256_featone_hot_D_conditional/ns1000/iter400_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_dstate16_dconv4_expand2"
OUT_DIR="data_comparison_16qubit_1000samples_2d_full"
FIG_DIR="figures/kqd_2d_1000shot_full"
STATUS="${LOG_DIR}/status.log"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${STATUS}"
}

archive_if_exists() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    local archive="${path}.interrupted_${STAMP}"
    log_status "Archiving existing run: ${path} -> ${archive}"
    mv "${path}" "${archive}"
  fi
}

wait_pair() {
  local left_pid="$1"
  local right_pid="$2"
  local left_name="$3"
  local right_name="$4"
  local left_status=0
  local right_status=0

  wait "${left_pid}" || left_status=$?
  wait "${right_pid}" || right_status=$?

  if [[ "${left_status}" -ne 0 || "${right_status}" -ne 0 ]]; then
    log_status "FAILED: ${left_name} status=${left_status}, ${right_name} status=${right_status}"
    exit 1
  fi
}

log_status "Starting 2D KQD 1000-shot pipeline"
log_status "Logs: ${LOG_DIR}"

archive_if_exists "${TF_RUN}"
archive_if_exists "${MAMBA_RUN}"

log_status "Training transformer on GPU 1"
CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/train.py \
  --config configs/experiment/kqd_2d_transformer.yaml \
  --data.train_samples 1000 \
  --training.iterations 400 \
  > "${LOG_DIR}/train_transformer.log" 2>&1 &
TF_TRAIN_PID=$!

log_status "Training mamba on GPU 2"
CUDA_VISIBLE_DEVICES=2 "${PY}" scripts/train.py \
  --config configs/experiment/kqd_2d_mamba.yaml \
  --data.train_samples 1000 \
  --training.iterations 400 \
  > "${LOG_DIR}/train_mamba.log" 2>&1 &
MAMBA_TRAIN_PID=$!

log_status "Waiting for training jobs: transformer pid=${TF_TRAIN_PID}, mamba pid=${MAMBA_TRAIN_PID}"
wait_pair "${TF_TRAIN_PID}" "${MAMBA_TRAIN_PID}" "transformer_train" "mamba_train"
log_status "Training complete"

log_status "Sampling transformer on GPU 1"
CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/sample.py \
  --results-dir "${TF_RUN}" \
  --snapshots 1000 \
  --krylov-dim-range 14 \
  --splits test \
  > "${LOG_DIR}/sample_transformer.log" 2>&1 &
TF_SAMPLE_PID=$!

log_status "Sampling mamba on GPU 2"
CUDA_VISIBLE_DEVICES=2 "${PY}" scripts/sample.py \
  --results-dir "${MAMBA_RUN}" \
  --snapshots 1000 \
  --krylov-dim-range 14 \
  --splits test \
  > "${LOG_DIR}/sample_mamba.log" 2>&1 &
MAMBA_SAMPLE_PID=$!

log_status "Waiting for sampling jobs: transformer pid=${TF_SAMPLE_PID}, mamba pid=${MAMBA_SAMPLE_PID}"
wait_pair "${TF_SAMPLE_PID}" "${MAMBA_SAMPLE_PID}" "transformer_sample" "mamba_sample"
log_status "Sampling complete"

log_status "Evaluating 1000-shot KQD"
"${PY}" scripts/evaluate.py \
  --task kqd \
  --lattice 2d \
  --n-qubit 16 \
  --max-krylov-dim 15 \
  --snapshots 1000 \
  --num-samples 1000 \
  --data-root ./data/heisenberg_KQD_2d/ \
  --transformer-results "${TF_RUN}" \
  --mamba-results "${MAMBA_RUN}" \
  --models ideal shadow transformer mamba \
  --splits test \
  --output-dir "${OUT_DIR}" \
  > "${LOG_DIR}/evaluate.log" 2>&1

log_status "Creating figure"
"${PY}" scripts/figures/fig2_table1_kqd.py \
  --mode figure \
  --results "${OUT_DIR}/all_results.pkl" \
  --output-dir "${FIG_DIR}" \
  > "${LOG_DIR}/figure.log" 2>&1

log_status "Creating table"
"${PY}" scripts/figures/fig2_table1_kqd.py \
  --mode table \
  --data-comparison-dirs "${OUT_DIR}" \
  --output-dir "${FIG_DIR}" \
  > "${LOG_DIR}/table.log" 2>&1

log_status "DONE"
