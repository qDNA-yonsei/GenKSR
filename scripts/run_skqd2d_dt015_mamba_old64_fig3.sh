#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PY:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ITERATIONS="${ITERATIONS:-100}"
LOG_DIR="${LOG_DIR:-logs/skqd2d_dt015_mamba_old64_iter${ITERATIONS}_${STAMP}}"
mkdir -p "${LOG_DIR}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/genksr_mpl_${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

MAIN_DATA_ROOT="./data/heisenberg_SQKD_2d_dt0.15/"
J2_DATA_ROOT="./data/heisenberg_SQKD_2d_j2_0.5_dt0.15/"
RESULTS_ROOT="./results_dt0.15"

MAMBA_RUN="${RESULTS_ROOT}/conditional_heisenberg_SQKD_2d_16qubit/gcn_proj_3_16-mamba_l4_d128_featone_hot_D_conditional/ns1000/iter${ITERATIONS}_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_dstate64_dconv8_expand4"
TF_SAMPLES="${TF_SAMPLES:-./results/skqd_j2_0.5_dt0.15_retrained_10000/transformer}"
MAMBA_SAMPLES="./results/skqd_j2_0.5_dt0.15_mamba_old64_iter${ITERATIONS}_10000/mamba"
J2_EVAL="./evaluation_sqkd_j2_0.5_dt0.15_mamba_old64_iter${ITERATIONS}_10000samples_fast"
FIG_DIR="./figures/skqd_2d_dt0.15_mamba_old64_iter${ITERATIONS}_10000shots_d2"

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

if [[ ! -d "${MAIN_DATA_ROOT}/16qubits/train" ]]; then
  echo "Missing training data: ${MAIN_DATA_ROOT}/16qubits/train" >&2
  exit 1
fi
if [[ ! -d "${J2_DATA_ROOT}/16qubits/test" ]]; then
  echo "Missing fixed-J2 data: ${J2_DATA_ROOT}/16qubits/test" >&2
  exit 1
fi
if [[ ! -d "${TF_SAMPLES}/samples/test/model" ]]; then
  echo "Missing transformer samples: ${TF_SAMPLES}/samples/test/model" >&2
  exit 1
fi

log_status "Starting old-setting Mamba SKQD run: dt=0.15, iterations=${ITERATIONS}"
log_status "Logs: ${LOG_DIR}"
log_status "Mamba setting: gcn_proj_3_16, l4 d128, d_state=64, d_conv=8, expand=4"

archive_if_exists "${MAMBA_RUN}"
archive_if_exists "${MAMBA_SAMPLES}"
archive_if_exists "${J2_EVAL}"
archive_if_exists "${FIG_DIR}"

log_status "Training old-setting mamba on GPU 2"
CUDA_VISIBLE_DEVICES=2 "${PY}" scripts/train.py \
  --config configs/experiment/skqd_2d_mamba.yaml \
  --results_dir "${RESULTS_ROOT}" \
  --data.data_root "${MAIN_DATA_ROOT}" \
  --data.sqkd_dt 0.15 \
  --decoder.d_state 64 \
  --decoder.d_conv 8 \
  --decoder.expand 4 \
  --training.iterations "${ITERATIONS}" \
  > "${LOG_DIR}/train_mamba_old64.log" 2>&1
log_status "Training complete"

log_status "Sampling fixed J2=0.5, 10000 samples, old-setting mamba on GPU 2"
CUDA_VISIBLE_DEVICES=2 "${PY}" scripts/sample.py \
  --results-dir "${MAMBA_RUN}" \
  --output-results-dir "${MAMBA_SAMPLES}" \
  --data-root "${J2_DATA_ROOT}" \
  --snapshots 10000 \
  --krylov-dim-range 30 \
  --splits test \
  --sqkd-dt 0.15 \
  --seed 42 \
  > "${LOG_DIR}/sample_j2_mamba_old64_10000.log" 2>&1
log_status "Sampling complete"

log_status "Evaluating fixed J2=0.5 with existing transformer and old-setting mamba"
"${PY}" scripts/evaluate_sqkd_fast2d.py \
  --n-qubit 16 \
  --max-krylov-dim 30 \
  --num-samples 10000 \
  --snapshots 10000 \
  --data-root "${J2_DATA_ROOT}" \
  --transformer-results "${TF_SAMPLES}" \
  --mamba-results "${MAMBA_SAMPLES}" \
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
