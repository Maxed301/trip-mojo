#!/usr/bin/env bash
#SBATCH -J trip-mojo-mi100
#SBATCH -o /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/mojo_mi100_%j.out
#SBATCH -e /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/mojo_mi100_%j.err
#SBATCH -D /lustre/bio/mdick/CUDA/trip-mojo
#SBATCH -p amd_gpu
#SBATCH --constraint=mi100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

ROOT="${ROOT:-/lustre/bio/mdick/CUDA}"
REPO="${REPO:-${ROOT}/trip-mojo}"
PROFILES="${PROFILES:-${ROOT}/PROFILES/trip-mojo}"
CONTAINER="${CONTAINER:-${ROOT}/trip-dev.sif}"
UV="${UV:-${ROOT}/bin/uv}"
TOOLCHAIN="${TOOLCHAIN:-${ROOT}/mi100-toolchain}"
THREADS="${THREADS:-12}"
JOB_ID="${SLURM_JOB_ID:-manual}"
BUILD_DIR="${BUILD_DIR:-${REPO}/build/mi100-${JOB_ID}}"
IMPORT_DIR="${BUILD_DIR}/mojo-import"
MOJO_LIB="${REPO}/.venv/lib/python3.12/site-packages/modular/lib"

mkdir -p "${PROFILES}" "${BUILD_DIR}" "${IMPORT_DIR}" \
  "${ROOT}/.tmp" "${ROOT}/.uv-python"
test -f "${TOOLCHAIN}/std.mojopkg"
echo 'da280f4d1d5023c0d21f35cf543d81a7a890b8a6fd6fec54e8c91f77808401de' \
  "${TOOLCHAIN}/std.mojopkg" | sha256sum -c

echo "job=${JOB_ID} host=$(hostname) commit=$(git -C "${REPO}" rev-parse HEAD)"
echo "build=${BUILD_DIR} target=gfx908 threads=${THREADS}"

apptainer exec --rocm -B /lustre:/lustre "${CONTAINER}" env \
  ROOT="${ROOT}" REPO="${REPO}" BUILD_DIR="${BUILD_DIR}" \
  IMPORT_DIR="${IMPORT_DIR}" TOOLCHAIN="${TOOLCHAIN}" UV="${UV}" \
  THREADS="${THREADS}" \
  UV_CACHE_DIR="${ROOT}/.uv-cache" \
  UV_PYTHON_INSTALL_DIR="${ROOT}/.uv-python" TMPDIR="${ROOT}/.tmp" \
  MODULAR_CRASH_REPORTING_ENABLED=false \
  MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
  MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=20 \
  LD_LIBRARY_PATH="${MOJO_LIB}:/.singularity.d/libs:${LD_LIBRARY_PATH:-}" \
  bash --noprofile --norc -lc '
    set -euo pipefail
    cd "${REPO}"
    "${UV}" sync --frozen
    ln -sf "${TOOLCHAIN}/std.mojopkg" "${IMPORT_DIR}/std.mojopkg"
    ln -sf "${REPO}/.venv/lib/python3.12/site-packages/modular/lib/mojo/layout.mojopkg" \
      "${IMPORT_DIR}/layout.mojopkg"
    export MODULAR_MOJO_MAX_IMPORT_PATH="${IMPORT_DIR}"
    MOJO="${REPO}/.venv/bin/mojo"
    "${MOJO}" build -I . -O3 -g1 -D FDCB_CPU_THREADS="${THREADS}" \
      -D FDCB_ABI_ACCELERATOR=true --target-accelerator gfx908 \
      --emit shared-lib fdcb_abi.mojo -Xlinker -lm \
      -o "${BUILD_DIR}/libtrip_fdcb_mojo.so"
    "${MOJO}" run -I . -O3 -D FDCB_CPU_THREADS="${THREADS}" \
      --target-accelerator gfx908 tests/test_fdcb_accelerator.mojo
    "${MOJO}" run -I . -O3 -D FDCB_CPU_THREADS="${THREADS}" \
      -D FDCB_MIXED32=true --target-accelerator gfx908 \
      tests/test_fdcb_accelerator_mixed32.mojo
    cc -O2 -DFDCB_TEST_REQUIRE_ACCELERATOR -Iinclude tests/ffi/test_fdcb_abi.c \
      -L"${BUILD_DIR}" -ltrip_fdcb_mojo -Wl,-rpath,"${BUILD_DIR}" \
      -o "${BUILD_DIR}/test_fdcb_abi"
    cc -O2 -Iinclude tests/ffi/test_clinical_dose_abi.c \
      -L"${BUILD_DIR}" -ltrip_fdcb_mojo -Wl,-rpath,"${BUILD_DIR}" \
      -o "${BUILD_DIR}/test_clinical_dose_abi"
    "${BUILD_DIR}/test_fdcb_abi"
    "${BUILD_DIR}/test_clinical_dose_abi"
  '
