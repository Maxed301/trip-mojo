#!/usr/bin/env bash
#SBATCH -J trip-mojo-h200
#SBATCH -o /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/mojo_h200_%j.out
#SBATCH -e /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/mojo_h200_%j.err
#SBATCH -D /lustre/bio/mdick/CUDA/trip-mojo
#SBATCH -p nvidia_gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT="${ROOT:-/lustre/bio/mdick/CUDA}"
REPO="${REPO:-${ROOT}/trip-mojo}"
PROFILES="${PROFILES:-${ROOT}/PROFILES/trip-mojo}"
CONTAINER="${CONTAINER:-${ROOT}/trip-dev.sif}"
UV="${UV:-${ROOT}/bin/uv}"
THREADS="${THREADS:-12}"
GPU_TARGET="${GPU_TARGET:-sm_90a}"
GPU_MEMORY_CHUNK_PERCENT="${GPU_MEMORY_CHUNK_PERCENT:-10}"
GPU_MEMORY_SIZE_PERCENT="${GPU_MEMORY_SIZE_PERCENT:-10}"
JOB_ID="${SLURM_JOB_ID:-manual}"
BUILD_DIR="${BUILD_DIR:-${REPO}/build/h200-${JOB_ID}}"
RUN_LOG="${PROFILES}/mojo_h200_${JOB_ID}.log"
POWER_CSV="${PROFILES}/mojo_h200_${JOB_ID}_power.csv"
UV_PYTHON_DIR="${UV_PYTHON_DIR:-${ROOT}/.uv-python}"
TMPDIR="${TMPDIR:-${ROOT}/.tmp}"

mkdir -p "${PROFILES}" "${BUILD_DIR}" "${UV_PYTHON_DIR}" "${TMPDIR}"
test -x "${UV}"
cd "${REPO}"

echo "job=${JOB_ID}"
echo "host=$(hostname)"
echo "repo=${REPO}"
echo "uv=${UV}"
echo "commit=$(git rev-parse HEAD)"
echo "dirty_files=$(git status --porcelain | wc -l)"
echo "gpu_target=${GPU_TARGET}"
echo "gpu_memory_chunk_percent=${GPU_MEMORY_CHUNK_PERCENT}"
echo "gpu_memory_size_percent=${GPU_MEMORY_SIZE_PERCENT}"
echo "threads=${THREADS}"
echo "build_dir=${BUILD_DIR}"
echo "run_log=${RUN_LOG}"
echo "power_log=${POWER_CSV}"
nvidia-smi

nvidia-smi \
  --query-gpu=timestamp,name,power.draw,utilization.gpu,utilization.memory,memory.used,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv -l 1 > "${POWER_CSV}" &
POWER_PID=$!
cleanup() {
  if kill -0 "${POWER_PID}" 2>/dev/null; then
    kill "${POWER_PID}" 2>/dev/null || true
    wait "${POWER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

time -p apptainer exec --nv -B /lustre:/lustre \
  "${CONTAINER}" env \
    REPO="${REPO}" BUILD_DIR="${BUILD_DIR}" UV="${UV}" \
    THREADS="${THREADS}" GPU_TARGET="${GPU_TARGET}" \
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT="${GPU_MEMORY_CHUNK_PERCENT}" \
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT="${GPU_MEMORY_SIZE_PERCENT}" \
    UV_CACHE_DIR="${ROOT}/.uv-cache" \
    UV_PYTHON_INSTALL_DIR="${UV_PYTHON_DIR}" TMPDIR="${TMPDIR}" \
    bash --noprofile --norc -lc '
      set -euo pipefail
      cd "${REPO}"
      "${UV}" sync --frozen
      MOJO="${REPO}/.venv/bin/mojo"
      "${MOJO}" --version
      "${MOJO}" build -I . -O3 -g1 \
        -D FDCB_CPU_THREADS="${THREADS}" \
        -D FDCB_ABI_ACCELERATOR=true \
        --target-accelerator "${GPU_TARGET}" \
        --emit shared-lib fdcb_abi.mojo -Xlinker -lm \
        -o "${BUILD_DIR}/libtrip_fdcb_mojo.so"

      "${MOJO}" run -I . -O3 \
        -D FDCB_CPU_THREADS="${THREADS}" \
        --target-accelerator "${GPU_TARGET}" \
        tests/test_fdcb_accelerator.mojo
      "${MOJO}" run -I . -O3 \
        -D FDCB_CPU_THREADS="${THREADS}" \
        -D FDCB_MIXED32=true \
        --target-accelerator "${GPU_TARGET}" \
        tests/test_fdcb_accelerator_mixed32.mojo

      cc -O2 -DFDCB_TEST_REQUIRE_ACCELERATOR -Iinclude \
        tests/ffi/test_fdcb_abi.c \
        -L"${BUILD_DIR}" -ltrip_fdcb_mojo \
        -Wl,-rpath,"${BUILD_DIR}" -o "${BUILD_DIR}/test_fdcb_abi"
      cc -O2 -Iinclude tests/ffi/test_clinical_dose_abi.c \
        -L"${BUILD_DIR}" -ltrip_fdcb_mojo \
        -Wl,-rpath,"${BUILD_DIR}" -o "${BUILD_DIR}/test_clinical_dose_abi"
      "${BUILD_DIR}/test_fdcb_abi"
      "${BUILD_DIR}/test_clinical_dose_abi"
      stat -c "artifact %y %s %n" "${BUILD_DIR}/libtrip_fdcb_mojo.so"
    ' 2>&1 | tee "${RUN_LOG}"

cleanup
trap - EXIT

awk -F, '
  NR > 1 {
    p=$3; gsub(/^[[:space:]]+|[[:space:]]+$/, "", p); gsub(/ W/, "", p); p += 0;
    t=$7; gsub(/^[[:space:]]+|[[:space:]]+$/, "", t); gsub(/ C/, "", t); t += 0;
    m=$6; gsub(/^[[:space:]]+|[[:space:]]+$/, "", m); gsub(/ MiB/, "", m); m += 0;
    ps += p; ts += t; ms += m;
    if (p > pmax) pmax = p;
    if (t > tmax) tmax = t;
    if (m > mmax) mmax = m;
    n++;
  }
  END {
    if (n > 0) {
      printf("power mean=%.2fW max=%.2fW samples=%d\n", ps/n, pmax, n);
      printf("temp mean=%.2fC max=%.2fC\n", ts/n, tmax);
      printf("vram mean=%.2fMiB max=%.2fMiB\n", ms/n, mmax);
    }
  }
' "${POWER_CSV}"
