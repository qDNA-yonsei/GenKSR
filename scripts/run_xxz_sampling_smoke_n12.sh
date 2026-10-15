#!/usr/bin/env bash
# Minimal best-checkpoint sampling validation for both N=12 decoders.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TRAIN_ROOT="${ROOT}/results/v3_xxz_sparsity_scaling_formal_seed0_20260715/conditional_xxz_sparsity_SQKD_1d_12qubit"
OUTPUT_ROOT="${ROOT}/artifacts/v3_xxz_sparsity_scaling/formal_sampling_smoke_best_seed20260715"

TF_RUN="${TRAIN_ROOT}/gcn_proj_3_16-transformer_l4_d128_h4_featone_hot_D_conditional/ns1000/iter100_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_cnormnone_seed0"
MAMBA_RUN="${TRAIN_ROOT}/gcn_proj_3_16-mamba_l4_d128_featone_hot_D_conditional/ns1000/iter100_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_dstate16_dconv4_expand2_cnormnone_seed0"

names=("n12_transformer" "n12_mamba")
runs=("${TF_RUN}" "${MAMBA_RUN}")

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite existing smoke output: ${OUTPUT_ROOT}" >&2
  exit 2
fi

for run in "${runs[@]}"; do
  if [[ ! -f "${run}/checkpoints/model_best.pth.tar" ]]; then
    echo "Best checkpoint not found: ${run}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${ROOT}" || exit 2

printf 'name\tgpu\tpid\tsource_run\toutput_dir\n' > "${OUTPUT_ROOT}/pids.tsv"
: > "${OUTPUT_ROOT}/exit_status.tsv"

pids=()
for i in "${!names[@]}"; do
  name="${names[$i]}"
  run="${runs[$i]}"
  output_dir="${OUTPUT_ROOT}/${name}"
  log="${OUTPUT_ROOT}/logs/${name}.log"

  (
    exec env \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES=1 \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=12 \
      MKL_NUM_THREADS=12 \
      OPENBLAS_NUM_THREADS=12 \
      "${PYTHON_BIN}" scripts/sample.py \
      --results-dir "${run}" \
      --output-results-dir "${output_dir}" \
      --checkpoint model_best.pth.tar \
      --snapshots 16 \
      --seed 20260715 \
      --krylov-dims 0 14 \
      --splits test
  ) > "${log}" 2>&1 &

  pid=$!
  pids+=("${pid}")
  printf '%s\t1\t%s\t%s\t%s\n' \
    "${name}" "${pid}" "${run}" "${output_dir}" >> "${OUTPUT_ROOT}/pids.tsv"
done

overall_status=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  if wait "${pid}"; then
    exit_code=0
  else
    exit_code=$?
    overall_status=1
  fi
  printf '%s\t%s\t%s\t%s\n' \
    "${name}" "${pid}" "${exit_code}" "$(date --iso-8601=seconds)" >> "${OUTPUT_ROOT}/exit_status.tsv"
done

{
  echo "finish_time=$(date --iso-8601=seconds)"
  echo "overall_status=${overall_status}"
} > "${OUTPUT_ROOT}/launcher_status.txt"

exit "${overall_status}"
