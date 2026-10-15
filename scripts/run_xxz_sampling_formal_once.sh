#!/usr/bin/env bash
# Formal single-pass test sampling from all six best XXZ sparsity checkpoints.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
TRAIN_ROOT="${ROOT}/results/v3_xxz_sparsity_scaling_formal_seed0_20260715"
OUTPUT_ROOT="${ROOT}/artifacts/v3_xxz_sparsity_scaling/formal_sampling_best_trainseed0_sampleseed20260716"
MANIFEST="${ROOT}/data/xxz_sparsity_1d/ratio_manifest.csv"
EXPECTED_MANIFEST_SHA256="e2feddb97957e2654c738232ea327afe0b3d279090517da509c1b362098ff351"
SAMPLING_SEED=20260716

names=(
  "n12_transformer"
  "n12_mamba"
  "n16_transformer"
  "n16_mamba"
  "n20_transformer"
  "n20_mamba"
)

# Avoid GPU 0 because it hosts unrelated processes.  The assignment balances
# the slower Transformer workers between the two free GPUs.
gpus=(1 2 1 2 2 1)

runs=(
  "${TRAIN_ROOT}/conditional_xxz_sparsity_SQKD_1d_12qubit/gcn_proj_3_16-transformer_l4_d128_h4_featone_hot_D_conditional/ns1000/iter100_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_cnormnone_seed0"
  "${TRAIN_ROOT}/conditional_xxz_sparsity_SQKD_1d_12qubit/gcn_proj_3_16-mamba_l4_d128_featone_hot_D_conditional/ns1000/iter100_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_dstate16_dconv4_expand2_cnormnone_seed0"
  "${TRAIN_ROOT}/conditional_xxz_sparsity_SQKD_1d_16qubit/gcn_proj_3_16-transformer_l4_d128_h4_featone_hot_D_conditional/ns1000/iter100_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_cnormnone_seed0"
  "${TRAIN_ROOT}/conditional_xxz_sparsity_SQKD_1d_16qubit/gcn_proj_3_16-mamba_l4_d128_featone_hot_D_conditional/ns1000/iter100_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_dstate16_dconv4_expand2_cnormnone_seed0"
  "${TRAIN_ROOT}/conditional_xxz_sparsity_SQKD_1d_20qubit/gcn_proj_3_16-transformer_l4_d128_h4_featone_hot_D_conditional/ns1000/iter100_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_cnormnone_seed0"
  "${TRAIN_ROOT}/conditional_xxz_sparsity_SQKD_1d_20qubit/gcn_proj_3_16-mamba_l4_d128_featone_hot_D_conditional/ns1000/iter100_lr0.001_wd0.0_bs256_dropout0.1_samplestruct2_lrschedulewarmup_cosine_dstate16_dconv4_expand2_cnormnone_seed0"
)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Frozen manifest not found: ${MANIFEST}" >&2
  exit 2
fi

actual_manifest_sha256="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
if [[ "${actual_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
  echo "Manifest hash mismatch: ${actual_manifest_sha256}" >&2
  exit 2
fi

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite existing formal sampling output: ${OUTPUT_ROOT}" >&2
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

{
  echo "launch_time=$(date --iso-8601=seconds)"
  echo "root=${ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "training_seed=0"
  echo "sampling_seed=${SAMPLING_SEED}"
  echo "sampling_repeats=1"
  echo "snapshots_per_hamiltonian_dim=1000"
  echo "krylov_dims=0..14"
  echo "split=test"
  echo "checkpoint=model_best.pth.tar"
  echo "manifest_sha256=${actual_manifest_sha256}"
  "${PYTHON_BIN}" --version
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  sha256sum scripts/sample.py
  for run in "${runs[@]}"; do
    sha256sum "${run}/checkpoints/model_best.pth.tar"
  done
} > "${OUTPUT_ROOT}/provenance.txt" 2>&1

printf 'name\tgpu\tpid\tsource_run\toutput_dir\n' > "${OUTPUT_ROOT}/pids.tsv"
: > "${OUTPUT_ROOT}/exit_status.tsv"

pids=()
for i in "${!names[@]}"; do
  name="${names[$i]}"
  gpu="${gpus[$i]}"
  run="${runs[$i]}"
  output_dir="${OUTPUT_ROOT}/${name}"
  log="${OUTPUT_ROOT}/logs/${name}.log"

  (
    exec env \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=8 \
      MKL_NUM_THREADS=8 \
      OPENBLAS_NUM_THREADS=8 \
      "${PYTHON_BIN}" scripts/sample.py \
      --results-dir "${run}" \
      --output-results-dir "${output_dir}" \
      --checkpoint model_best.pth.tar \
      --snapshots 1000 \
      --seed "${SAMPLING_SEED}" \
      --krylov-dim-range 15 \
      --splits test
  ) > "${log}" 2>&1 &

  pid=$!
  pids+=("${pid}")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${gpu}" "${pid}" "${run}" "${output_dir}" >> "${OUTPUT_ROOT}/pids.tsv"
done

echo "all_workers_started_time=$(date --iso-8601=seconds)" > "${OUTPUT_ROOT}/launcher_status.txt"

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
} >> "${OUTPUT_ROOT}/launcher_status.txt"

exit "${overall_status}"
