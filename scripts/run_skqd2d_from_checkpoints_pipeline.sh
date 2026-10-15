#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PY:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/skqd2d_from_checkpoints_${STAMP}}"
mkdir -p "${LOG_DIR}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/genksr_mpl_${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

MAIN_DATA_ROOT="./data/heisenberg_SQKD_2d/"
J2_DATA_ROOT="./data/heisenberg_SQKD_2d_j2_0.5/"

TF_RUN="results/conditional_heisenberg_SQKD_2d_16qubit/gcn_encoder_3_256-transformer_l8_d256_h8_featone_hot_D_conditional/ns1000/iter100_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine"
MAMBA_RUN="results/conditional_heisenberg_SQKD_2d_16qubit/gcn_encoder_3_256-mamba_l8_d256_featone_hot_D_conditional/ns1000/iter100_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_dstate16_dconv4_expand2"

J2_TF_SAMPLES="results/skqd_j2_0.5_samples/transformer"
J2_MAMBA_SAMPLES="results/skqd_j2_0.5_samples/mamba"

FIG8_EVAL_1000="evaluation_sqkd_16qubit_kd30_1000samples"
FIG8_EVAL_5000="evaluation_sqkd_16qubit_kd30_5000samples"
FIG8_EVAL_10000="evaluation_sqkd_16qubit_kd30_10000samples"
J2_EVAL="evaluation_sqkd_j2_0.5_1000samples"
FIG_DIR="figures/skqd_2d"

STATUS="${LOG_DIR}/status.log"

log_status() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${STATUS}"
}

archive_if_exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    local archive="${path}.previous_${STAMP}"
    log_status "Archiving existing path: ${path} -> ${archive}"
    mv "${path}" "${archive}"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log_status "Missing required file: ${path}"
    exit 1
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

log_status "Starting SKQD continuation from existing checkpoints"
log_status "Logs: ${LOG_DIR}"
log_status "Using transformer checkpoint at epoch logged ~= 71/100"
log_status "Using mamba checkpoint at epoch logged ~= 88/100"

require_file "${MAIN_DATA_ROOT}/16qubits/train/data_D29_id29.npy"
require_file "${MAIN_DATA_ROOT}/16qubits/test/data_D29_id19.npy"
require_file "${J2_DATA_ROOT}/16qubits/test/data_D29_id0.npy"
require_file "${TF_RUN}/checkpoints/checkpoint_final.pth.tar"
require_file "${MAMBA_RUN}/checkpoints/checkpoint_final.pth.tar"

archive_if_exists "${TF_RUN}/samples"
archive_if_exists "${MAMBA_RUN}/samples"
archive_if_exists "${J2_TF_SAMPLES}"
archive_if_exists "${J2_MAMBA_SAMPLES}"
archive_if_exists "${FIG8_EVAL_1000}"
archive_if_exists "${FIG8_EVAL_5000}"
archive_if_exists "${FIG8_EVAL_10000}"
archive_if_exists "${J2_EVAL}"

log_status "Sampling random-J2 test set, 10000 generated samples, transformer on GPU 1"
CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/sample.py \
  --results-dir "${TF_RUN}" \
  --snapshots 10000 \
  --krylov-dim-range 30 \
  --splits test \
  > "${LOG_DIR}/sample_main_transformer_10000.log" 2>&1 &
TF_SAMPLE_PID=$!

log_status "Sampling random-J2 test set, 10000 generated samples, mamba on GPU 2"
CUDA_VISIBLE_DEVICES=2 "${PY}" scripts/sample.py \
  --results-dir "${MAMBA_RUN}" \
  --snapshots 10000 \
  --krylov-dim-range 30 \
  --splits test \
  > "${LOG_DIR}/sample_main_mamba_10000.log" 2>&1 &
MAMBA_SAMPLE_PID=$!

log_status "Waiting for random-J2 sampling jobs: transformer pid=${TF_SAMPLE_PID}, mamba pid=${MAMBA_SAMPLE_PID}"
wait_pair "${TF_SAMPLE_PID}" "${MAMBA_SAMPLE_PID}" "transformer_sample_main" "mamba_sample_main"
log_status "Random-J2 sampling complete"

for samples in 1000 5000 10000; do
  out_dir="evaluation_sqkd_16qubit_kd30_${samples}samples"
  log_status "Evaluating random-J2 Fig8 budget num_samples=${samples}"
  "${PY}" scripts/evaluate.py \
    --task sqkd \
    --n-qubit 16 \
    --max-krylov-dim 30 \
    --snapshots 10000 \
    --num-samples "${samples}" \
    --data-root "${MAIN_DATA_ROOT}" \
    --transformer-results "${TF_RUN}" \
    --mamba-results "${MAMBA_RUN}" \
    --models device transformer mamba \
    --splits test \
    --output-dir "${out_dir}" \
    > "${LOG_DIR}/evaluate_main_${samples}.log" 2>&1
done

log_status "Sampling fixed J2=0.5 test set, transformer on GPU 1"
CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/sample.py \
  --results-dir "${TF_RUN}" \
  --output-results-dir "${J2_TF_SAMPLES}" \
  --data-root "${J2_DATA_ROOT}" \
  --snapshots 10000 \
  --krylov-dim-range 30 \
  --splits test \
  > "${LOG_DIR}/sample_j2_transformer_10000.log" 2>&1 &
TF_J2_SAMPLE_PID=$!

log_status "Sampling fixed J2=0.5 test set, mamba on GPU 2"
CUDA_VISIBLE_DEVICES=2 "${PY}" scripts/sample.py \
  --results-dir "${MAMBA_RUN}" \
  --output-results-dir "${J2_MAMBA_SAMPLES}" \
  --data-root "${J2_DATA_ROOT}" \
  --snapshots 10000 \
  --krylov-dim-range 30 \
  --splits test \
  > "${LOG_DIR}/sample_j2_mamba_10000.log" 2>&1 &
MAMBA_J2_SAMPLE_PID=$!

log_status "Waiting for fixed-J2 sampling jobs: transformer pid=${TF_J2_SAMPLE_PID}, mamba pid=${MAMBA_J2_SAMPLE_PID}"
wait_pair "${TF_J2_SAMPLE_PID}" "${MAMBA_J2_SAMPLE_PID}" "transformer_sample_j2" "mamba_sample_j2"
log_status "Fixed-J2 sampling complete"

log_status "Evaluating fixed J2=0.5 Fig3/Table2 run"
"${PY}" scripts/evaluate.py \
  --task sqkd \
  --n-qubit 16 \
  --max-krylov-dim 30 \
  --snapshots 10000 \
  --num-samples 1000 \
  --data-root "${J2_DATA_ROOT}" \
  --transformer-results "${J2_TF_SAMPLES}" \
  --mamba-results "${J2_MAMBA_SAMPLES}" \
  --models device transformer mamba \
  --splits test \
  --output-dir "${J2_EVAL}" \
  > "${LOG_DIR}/evaluate_j2_0.5.log" 2>&1

mkdir -p "${FIG_DIR}"

log_status "Creating Fig3"
"${PY}" scripts/figures/fig3_table2_fig8_skqd.py \
  --mode fig3 \
  --results "${J2_EVAL}/all_results.pkl" \
  --output-dir "${FIG_DIR}" \
  > "${LOG_DIR}/figure3.log" 2>&1

log_status "Creating Table2"
"${PY}" scripts/figures/fig3_table2_fig8_skqd.py \
  --mode table2 \
  --results "${J2_EVAL}/all_results.pkl" \
  --output-dir "${FIG_DIR}" \
  > "${LOG_DIR}/table2.log" 2>&1

log_status "Creating Fig8"
"${PY}" scripts/figures/fig3_table2_fig8_skqd.py \
  --mode fig8 \
  --results-by-samples \
    "1000:${FIG8_EVAL_1000}/all_results.pkl" \
    "5000:${FIG8_EVAL_5000}/all_results.pkl" \
    "10000:${FIG8_EVAL_10000}/all_results.pkl" \
  --d-train 15 \
  --d-extrap 30 \
  --output-dir "${FIG_DIR}" \
  > "${LOG_DIR}/figure8.log" 2>&1

log_status "DONE"
