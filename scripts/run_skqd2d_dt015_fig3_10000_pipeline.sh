#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PY:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ITERATIONS="${ITERATIONS:-100}"
RUN_LABEL="${RUN_LABEL:-retrained}"
if [[ "${ITERATIONS}" != "100" && "${RUN_LABEL}" == "retrained" ]]; then
  RUN_LABEL="retrained_iter${ITERATIONS}"
fi
LOG_DIR="${LOG_DIR:-logs/skqd2d_dt015_fig3_10000_iter${ITERATIONS}_${STAMP}}"
mkdir -p "${LOG_DIR}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/genksr_mpl_${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

MAIN_DATA_ROOT="./data/heisenberg_SQKD_2d_dt0.15/"
MAIN_DATA_DIR="${MAIN_DATA_ROOT}/16qubits"
J2_DATA_ROOT="./data/heisenberg_SQKD_2d_j2_0.5_dt0.15/"
J2_DATA_DIR="${J2_DATA_ROOT}/16qubits"

RESULTS_ROOT="./results_dt0.15"
TF_RUN="${RESULTS_ROOT}/conditional_heisenberg_SQKD_2d_16qubit/gcn_encoder_3_256-transformer_l8_d256_h8_featone_hot_D_conditional/ns1000/iter${ITERATIONS}_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine"
MAMBA_RUN="${RESULTS_ROOT}/conditional_heisenberg_SQKD_2d_16qubit/gcn_encoder_3_256-mamba_l8_d256_featone_hot_D_conditional/ns1000/iter${ITERATIONS}_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_dstate16_dconv4_expand2"

J2_TF_SAMPLES="./results/skqd_j2_0.5_dt0.15_${RUN_LABEL}_10000/transformer"
J2_MAMBA_SAMPLES="./results/skqd_j2_0.5_dt0.15_${RUN_LABEL}_10000/mamba"
J2_EVAL="./evaluation_sqkd_j2_0.5_dt0.15_${RUN_LABEL}_10000samples_fast"
FIG_DIR="./figures/skqd_2d_dt0.15_${RUN_LABEL}_10000shots_d2"

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

main_data_complete() {
  [[ -f "${MAIN_DATA_DIR}/train/data_D29_id29.npy" && -f "${MAIN_DATA_DIR}/test/data_D29_id19.npy" ]]
}

j2_data_complete() {
  [[ -f "${J2_DATA_DIR}/test/data_D29_id0.npy" ]]
}

log_status "Starting corrected SKQD Fig3 pipeline: dt=0.15, 10000 shots, iterations=${ITERATIONS}, run_label=${RUN_LABEL}"
log_status "Logs: ${LOG_DIR}"

if main_data_complete; then
  log_status "Corrected random-J2 data already exists; skipping generation"
else
  log_status "Generating corrected random-J2 SQKD data with dt=0.15"
  "${PY}" scripts/generate_data.py \
    --task sqkd \
    --n-qubit 16 \
    --L 4 \
    --krylov-dim 30 \
    --num-hamiltonians-train 30 \
    --num-hamiltonians-test 20 \
    --num-trotter-steps 6 \
    --shots 10000 \
    --coupling-model j1j2 \
    --J1 1.0 \
    --J2-range 0.0 1.0 \
    --sqkd-dt 0.15 \
    --sqkd-simulator statevector \
    --output-dir "${MAIN_DATA_DIR}/" \
    --seed 42 \
    > "${LOG_DIR}/generate_main_dt0.15.log" 2>&1
fi

if j2_data_complete; then
  log_status "Corrected fixed J2=0.5 data already exists; skipping generation"
else
  log_status "Generating corrected fixed J2=0.5 SQKD data with dt=0.15"
  "${PY}" scripts/generate_data.py \
    --task sqkd \
    --n-qubit 16 \
    --L 4 \
    --krylov-dim 30 \
    --num-hamiltonians-train 0 \
    --num-hamiltonians-test 1 \
    --num-trotter-steps 6 \
    --shots 10000 \
    --coupling-model j1j2 \
    --J1 1.0 \
    --J2-range 0.5 0.5 \
    --sqkd-dt 0.15 \
    --sqkd-simulator statevector \
    --output-dir "${J2_DATA_DIR}/" \
    --seed 42 \
    > "${LOG_DIR}/generate_j2_0.5_dt0.15.log" 2>&1
fi

archive_if_exists "${TF_RUN}"
archive_if_exists "${MAMBA_RUN}"
archive_if_exists "${J2_TF_SAMPLES}"
archive_if_exists "${J2_MAMBA_SAMPLES}"
archive_if_exists "${J2_EVAL}"
archive_if_exists "${FIG_DIR}"

log_status "Training transformer on GPU 1"
CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/train.py \
  --config configs/experiment/skqd_2d_transformer.yaml \
  --results_dir "${RESULTS_ROOT}" \
  --data.data_root "${MAIN_DATA_ROOT}" \
  --data.sqkd_dt 0.15 \
  --decoder.d_model 256 \
  --decoder.n_layers 8 \
  --decoder.n_heads 8 \
  --encoder.arch gcn_encoder_3_256 \
  --encoder.gcn_dim 256 \
  --training.iterations "${ITERATIONS}" \
  > "${LOG_DIR}/train_transformer.log" 2>&1 &
TF_TRAIN_PID=$!

log_status "Training mamba on GPU 2"
CUDA_VISIBLE_DEVICES=2 "${PY}" scripts/train.py \
  --config configs/experiment/skqd_2d_mamba.yaml \
  --results_dir "${RESULTS_ROOT}" \
  --data.data_root "${MAIN_DATA_ROOT}" \
  --data.sqkd_dt 0.15 \
  --decoder.d_model 256 \
  --decoder.n_layers 8 \
  --encoder.arch gcn_encoder_3_256 \
  --encoder.gcn_dim 256 \
  --training.iterations "${ITERATIONS}" \
  > "${LOG_DIR}/train_mamba.log" 2>&1 &
MAMBA_TRAIN_PID=$!

log_status "Waiting for training jobs: transformer pid=${TF_TRAIN_PID}, mamba pid=${MAMBA_TRAIN_PID}"
wait_pair "${TF_TRAIN_PID}" "${MAMBA_TRAIN_PID}" "transformer_train" "mamba_train"
log_status "Training complete"

log_status "Sampling fixed J2=0.5, 10000 samples, transformer on GPU 1"
CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/sample.py \
  --results-dir "${TF_RUN}" \
  --output-results-dir "${J2_TF_SAMPLES}" \
  --data-root "${J2_DATA_ROOT}" \
  --snapshots 10000 \
  --krylov-dim-range 30 \
  --splits test \
  --sqkd-dt 0.15 \
  --seed 42 \
  > "${LOG_DIR}/sample_j2_transformer_10000.log" 2>&1 &
TF_SAMPLE_PID=$!

log_status "Sampling fixed J2=0.5, 10000 samples, mamba on GPU 2"
CUDA_VISIBLE_DEVICES=2 "${PY}" scripts/sample.py \
  --results-dir "${MAMBA_RUN}" \
  --output-results-dir "${J2_MAMBA_SAMPLES}" \
  --data-root "${J2_DATA_ROOT}" \
  --snapshots 10000 \
  --krylov-dim-range 30 \
  --splits test \
  --sqkd-dt 0.15 \
  --seed 42 \
  > "${LOG_DIR}/sample_j2_mamba_10000.log" 2>&1 &
MAMBA_SAMPLE_PID=$!

log_status "Waiting for sampling jobs: transformer pid=${TF_SAMPLE_PID}, mamba pid=${MAMBA_SAMPLE_PID}"
wait_pair "${TF_SAMPLE_PID}" "${MAMBA_SAMPLE_PID}" "transformer_sample_j2" "mamba_sample_j2"
log_status "Sampling complete"

log_status "Evaluating corrected Fig3 run with fast SKQD evaluator"
"${PY}" scripts/evaluate_sqkd_fast2d.py \
  --n-qubit 16 \
  --max-krylov-dim 30 \
  --num-samples 10000 \
  --snapshots 10000 \
  --data-root "${J2_DATA_ROOT}" \
  --transformer-results "${J2_TF_SAMPLES}" \
  --mamba-results "${J2_MAMBA_SAMPLES}" \
  --models device transformer mamba \
  --splits test \
  --output-dir "${J2_EVAL}" \
  > "${LOG_DIR}/evaluate_j2_0.5_10000_fast.log" 2>&1

log_status "Creating D>=2 Fig3"
"${PY}" scripts/figures/fig3_table2_fig8_skqd.py \
  --mode fig3 \
  --results "${J2_EVAL}/all_results.pkl" \
  --d-min 2 \
  --output-dir "${FIG_DIR}" \
  > "${LOG_DIR}/figure3.log" 2>&1

log_status "DONE"
