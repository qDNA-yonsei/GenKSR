#!/usr/bin/env bash
# Launch the six seed-0 formal XXZ sparsity training runs.
#
# GPU layout:
#   GPU 0: N=12 Transformer + Mamba
#   GPU 1: N=16 Transformer + Mamba
#   GPU 2: N=20 Transformer + Mamba

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RESULTS_ROOT="${ROOT}/results/v3_xxz_sparsity_scaling_formal_seed0_20260715"
RUN_DIR="${ROOT}/artifacts/v3_xxz_sparsity_scaling/formal_training_seed0_20260715"
LOG_DIR="${RUN_DIR}/logs"
MANIFEST="${ROOT}/data/xxz_sparsity_1d/ratio_manifest.csv"
EXPECTED_MANIFEST_SHA256="e2feddb97957e2654c738232ea327afe0b3d279090517da509c1b362098ff351"

names=(
  "n12_transformer_seed0"
  "n12_mamba_seed0"
  "n16_transformer_seed0"
  "n16_mamba_seed0"
  "n20_transformer_seed0"
  "n20_mamba_seed0"
)
gpus=(0 0 1 1 2 2)
configs=(
  "configs/experiment/skqd_1d_xxz_sparsity_n12_transformer.yaml"
  "configs/experiment/skqd_1d_xxz_sparsity_n12_mamba.yaml"
  "configs/experiment/skqd_1d_xxz_sparsity_n16_transformer.yaml"
  "configs/experiment/skqd_1d_xxz_sparsity_n16_mamba.yaml"
  "configs/experiment/skqd_1d_xxz_sparsity_n20_transformer.yaml"
  "configs/experiment/skqd_1d_xxz_sparsity_n20_mamba.yaml"
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

if [[ -e "${RESULTS_ROOT}" ]]; then
  echo "Refusing to overwrite existing results root: ${RESULTS_ROOT}" >&2
  exit 2
fi

if [[ -e "${RUN_DIR}/pids.tsv" ]]; then
  echo "This formal launch already has a PID record: ${RUN_DIR}/pids.tsv" >&2
  exit 2
fi

for config in "${configs[@]}"; do
  if [[ ! -f "${ROOT}/${config}" ]]; then
    echo "Config not found: ${ROOT}/${config}" >&2
    exit 2
  fi
done

mkdir -p "${LOG_DIR}"
cd "${ROOT}" || exit 2

{
  echo "launch_time=$(date --iso-8601=seconds)"
  echo "root=${ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "results_root=${RESULTS_ROOT}"
  echo "seed=0"
  echo "omp_num_threads=12"
  echo "manifest=${MANIFEST}"
  echo "manifest_sha256=${actual_manifest_sha256}"
  "${PYTHON_BIN}" --version
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  sha256sum "${configs[@]}"
} > "${RUN_DIR}/provenance.txt" 2>&1

printf 'name\tgpu\tpid\tconfig\tlog\n' > "${RUN_DIR}/pids.tsv"
: > "${RUN_DIR}/exit_status.tsv"

pids=()
for i in "${!names[@]}"; do
  name="${names[$i]}"
  gpu="${gpus[$i]}"
  config="${configs[$i]}"
  log="${LOG_DIR}/${name}.log"

  (
    exec env \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=12 \
      MKL_NUM_THREADS=12 \
      OPENBLAS_NUM_THREADS=12 \
      "${PYTHON_BIN}" scripts/train.py \
      --config "${config}" \
      --results_dir "${RESULTS_ROOT}"
  ) > "${log}" 2>&1 &

  pid=$!
  pids+=("${pid}")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${gpu}" "${pid}" "${config}" "${log}" >> "${RUN_DIR}/pids.tsv"
done

echo "all_started_time=$(date --iso-8601=seconds)" > "${RUN_DIR}/launcher_status.txt"

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
    "${name}" "${pid}" "${exit_code}" "$(date --iso-8601=seconds)" >> "${RUN_DIR}/exit_status.tsv"
done

{
  echo "finish_time=$(date --iso-8601=seconds)"
  echo "overall_status=${overall_status}"
} >> "${RUN_DIR}/launcher_status.txt"

exit "${overall_status}"
