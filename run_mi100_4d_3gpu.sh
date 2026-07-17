#!/usr/bin/env bash
#SBATCH -J mojo-mi100-4d-3gpu
#SBATCH -o /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/mi100_4d_3gpu_%j.out
#SBATCH -e /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/mi100_4d_3gpu_%j.err
#SBATCH -D /lustre/bio/mdick/CUDA
#SBATCH -p amd_gpu
#SBATCH --constraint=mi100
#SBATCH --gres=gpu:3
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=160G
#SBATCH --time=00:10:00

set -euo pipefail

ROOT="${ROOT:-/lustre/bio/mdick/CUDA}"
REPO="${REPO:-${ROOT}/trip-mojo}"
TRIP="${TRIP:-${ROOT}/trip_temp}"
PROFILES="${PROFILES:-${ROOT}/PROFILES/trip-mojo}"
CONTAINER="${CONTAINER:-${ROOT}/trip-dev.sif}"
UV="${UV:-${ROOT}/bin/uv}"
TOOLCHAIN="${TOOLCHAIN:-${ROOT}/mi100-toolchain}"
CONTROL_THREADS="${CONTROL_THREADS:-32}"
PACK_THREADS="${PACK_THREADS:-16}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTDIR="${PROFILES}/mi100_4d_3gpu_${JOB_ID}"
BUILD_DIR="${REPO}/build/mi100-3gpu-${JOB_ID}"
IMPORT_DIR="${BUILD_DIR}/mojo-import"
WORK="${SLURM_TMPDIR:-/tmp}/trip-mojo-mi100-4d-3gpu-${JOB_ID}"
EXEC_SRC="${ROOT}/TRIP_DATA/P101_4Dopt/exec/P101_ITV_full4DITVplan.lustre.exec"
REFERENCE="${REFERENCE_DIR:-${PROFILES}/cpu_4d_ref_21714}"
TRIP_BINARY="${TRIP}/build-mojo-h200/TRiP98"
MOJO_RUNTIME="${REPO}/.venv/lib/python3.12/site-packages/modular/lib"

mkdir -p "${OUTDIR}/mojo" "${BUILD_DIR}" "${IMPORT_DIR}" "${WORK}" \
  "${ROOT}/.tmp" "${ROOT}/.uv-python"
test -x "${TRIP_BINARY}"
test -f "${TOOLCHAIN}/std.mojopkg"
test -f "${EXEC_SRC}"
for field in 1 2; do
  test -f "${REFERENCE}/4DP101_field${field}.rst"
done
echo 'da280f4d1d5023c0d21f35cf543d81a7a890b8a6fd6fec54e8c91f77808401de' \
  "${TOOLCHAIN}/std.mojopkg" | sha256sum -c

sed \
  -e "s/maxthreads(50)/maxthreads(${CONTROL_THREADS})/g" \
  -e "s/maxthreads(40)/maxthreads(${CONTROL_THREADS})/g" \
  -e "s|field 1 / write file(.*)|field 1 / write file(${OUTDIR}/mojo/4DP101_field1.rst)|" \
  -e "s|field 2 / write file(.*)|field 2 / write file(${OUTDIR}/mojo/4DP101_field2.rst)|" \
  "${EXEC_SRC}" > "${WORK}/mojo.exec"

echo "job=${JOB_ID} host=$(hostname) devices=3 control_threads=${CONTROL_THREADS} pack_threads=${PACK_THREADS} output=${OUTDIR}"
echo "trip_temp_commit=$(git -C "${TRIP}" rev-parse HEAD)"
echo "trip_mojo_commit=$(git -C "${REPO}" rev-parse HEAD) target=gfx908"

apptainer exec --rocm -B /lustre:/lustre -B "${WORK}:${WORK}" \
  "${CONTAINER}" env \
    ROOT="${ROOT}" REPO="${REPO}" BUILD_DIR="${BUILD_DIR}" \
    IMPORT_DIR="${IMPORT_DIR}" TOOLCHAIN="${TOOLCHAIN}" UV="${UV}" \
    CONTROL_THREADS="${CONTROL_THREADS}" PACK_THREADS="${PACK_THREADS}" \
    WORK="${WORK}" OUTDIR="${OUTDIR}" TRIP_BINARY="${TRIP_BINARY}" \
    UV_CACHE_DIR="${ROOT}/.uv-cache" \
    UV_PYTHON_INSTALL_DIR="${ROOT}/.uv-python" TMPDIR="${ROOT}/.tmp" \
    MODULAR_CRASH_REPORTING_ENABLED=false \
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=90 \
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=90 \
    LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/lib64:${BUILD_DIR}:${MOJO_RUNTIME}:/.singularity.d/libs:${LD_LIBRARY_PATH:-}" \
    bash --noprofile --norc -lc '
      set -euo pipefail
      rocm-smi --showproductname --showmeminfo vram
      cd "${REPO}"
      "${UV}" sync --frozen
      ln -sf "${TOOLCHAIN}/std.mojopkg" "${IMPORT_DIR}/std.mojopkg"
      ln -sf "${REPO}/.venv/lib/python3.12/site-packages/modular/lib/mojo/layout.mojopkg" \
        "${IMPORT_DIR}/layout.mojopkg"
      export MODULAR_MOJO_MAX_IMPORT_PATH="${IMPORT_DIR}"
      MOJO="${REPO}/.venv/bin/mojo"
      "${MOJO}" build -I . -O3 -g1 \
        -D FDCB_CPU_THREADS=12 -D FDCB_PACK_THREADS="${PACK_THREADS}" \
        -D FDCB_ABI_ACCELERATOR=true --target-accelerator gfx908 \
        --emit shared-lib fdcb_abi.mojo -Xlinker -lm \
        -o "${BUILD_DIR}/libtrip_fdcb_mojo.so"
      "${MOJO}" run -I . -O3 --target-accelerator gfx908 \
        tests/test_fdcb_three_accelerators.mojo
      cc -O2 tools/compare_rst_particles.c -lm \
        -o "${BUILD_DIR}/compare_rst_particles"

      while true; do
        date --iso-8601=seconds
        rocm-smi --showuse --showmemuse --showpower
        sleep 1
      done > "${OUTDIR}/power.log" 2>&1 &
      POWER_PID=$!
      cleanup() {
        kill "${POWER_PID}" 2>/dev/null || true
        wait "${POWER_PID}" 2>/dev/null || true
      }
      trap cleanup EXIT

      START=$(date +%s.%N)
      env OMP_NUM_THREADS="${CONTROL_THREADS}" \
        TRIP_FDCB_MOJO=1 TRIP_FDCB_MOJO_DEVICES=3 \
        TRIP_FDCB_MOJO_PACK_THREADS="${PACK_THREADS}" \
        TRIP_FDCB_MOJO_DEVICE_BOOTSTRAP=1 \
        "${TRIP_BINARY}" < "${WORK}/mojo.exec" \
        > "${OUTDIR}/mojo/run.log" 2>&1
      END=$(date +%s.%N)
      awk -v start="${START}" -v end="${END}" \
        '"'"'BEGIN { printf("mojo_wall=%.3f\n", end-start) }'"'"' \
        > "${OUTDIR}/mojo.time"
      cleanup
      trap - EXIT
    '

cat "${OUTDIR}/mojo.time"
if grep -Eq '^<(E|SYS)>|FDCB .* ABI error' "${OUTDIR}/mojo/run.log"; then
  grep -E '^<(E|SYS)>|FDCB .* ABI error' "${OUTDIR}/mojo/run.log" >&2
  exit 1
fi
grep -E 'accelerator devices=|FDCB Mojo pack|FDCB Mojo optimize|FDCB Mojo:|OptVoxelSetup:|OptCmd:' \
  "${OUTDIR}/mojo/run.log" | tail -40
for field in 1 2; do
  native="${REFERENCE}/4DP101_field${field}.rst"
  candidate="${OUTDIR}/mojo/4DP101_field${field}.rst"
  if cmp -s "${native}" "${candidate}"; then
    echo "field${field}_byte_identical=yes"
  else
    echo "field${field}_byte_identical=no"
  fi
  "${BUILD_DIR}/compare_rst_particles" "${native}" "${candidate}"
done
