#!/usr/bin/env bash
#SBATCH -J mojo-mi100-3d
#SBATCH -o /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/mi100_3d_%j.out
#SBATCH -e /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/mi100_3d_%j.err
#SBATCH -D /lustre/bio/mdick/CUDA
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
TRIP="${TRIP:-${ROOT}/trip_temp}"
PROFILES="${PROFILES:-${ROOT}/PROFILES/trip-mojo}"
CONTAINER="${CONTAINER:-${ROOT}/trip-dev.sif}"
UV="${UV:-${ROOT}/bin/uv}"
TOOLCHAIN="${TOOLCHAIN:-${ROOT}/mi100-toolchain}"
THREADS="${THREADS:-12}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTDIR="${PROFILES}/mi100_3d_${JOB_ID}"
BUILD_DIR="${REPO}/build/mi100-${JOB_ID}"
IMPORT_DIR="${BUILD_DIR}/mojo-import"
WORK="${SLURM_TMPDIR:-/tmp}/trip-mojo-mi100-3d-${JOB_ID}"
TRIP_BINARY="${TRIP_BINARY:-${TRIP}/build-mojo-h200/TRiP98}"
EXEC_SRC="${REPO}/P101_iGTV_3Dplan.exec"
REFERENCE="${ROOT}/TRIP_DATA/P101"
MOJO_RUNTIME="${REPO}/.venv/lib/python3.12/site-packages/modular/lib"

mkdir -p "${OUTDIR}" "${BUILD_DIR}" "${IMPORT_DIR}" "${WORK}" \
  "${ROOT}/.tmp" "${ROOT}/.uv-python"
test -x "${TRIP_BINARY}"
test -f "${EXEC_SRC}"
test -f "${TOOLCHAIN}/std.mojopkg"
echo 'da280f4d1d5023c0d21f35cf543d81a7a890b8a6fd6fec54e8c91f77808401de' \
  "${TOOLCHAIN}/std.mojopkg" | sha256sum -c

awk '/^dose / { exit } { print }' "${EXEC_SRC}" | sed \
  -e "s|/home/max/Projects/TRIP_DATA|${ROOT}/TRIP_DATA|g" \
  -e "s/maxthreads(40)/maxthreads(${THREADS})/g" \
  -e "s|field 1 / write file(.*)|field 1 / write file(${OUTDIR}/StaticP101_2110_field1_iGTV_R.rst)|" \
  -e "s|field 2 / write file(.*)|field 2 / write file(${OUTDIR}/StaticP101_2110_field2_iGTV_R.rst)|" \
  > "${WORK}/mojo.exec"
echo quit >> "${WORK}/mojo.exec"

echo "job=${JOB_ID} host=$(hostname) threads=${THREADS} output=${OUTDIR}"
echo "trip_temp_commit=$(git -C "${TRIP}" rev-parse HEAD)"
echo "trip_mojo_commit=$(git -C "${REPO}" rev-parse HEAD) target=gfx908"
rocm-smi --showproductname --showmeminfo vram || true

apptainer exec --rocm -B /lustre:/lustre -B "${WORK}:${WORK}" \
  "${CONTAINER}" env \
    ROOT="${ROOT}" REPO="${REPO}" BUILD_DIR="${BUILD_DIR}" \
    IMPORT_DIR="${IMPORT_DIR}" TOOLCHAIN="${TOOLCHAIN}" UV="${UV}" \
    THREADS="${THREADS}" WORK="${WORK}" OUTDIR="${OUTDIR}" \
    TRIP_BINARY="${TRIP_BINARY}" \
    UV_CACHE_DIR="${ROOT}/.uv-cache" \
    UV_PYTHON_INSTALL_DIR="${ROOT}/.uv-python" TMPDIR="${ROOT}/.tmp" \
    MODULAR_CRASH_REPORTING_ENABLED=false \
    MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=80 \
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=80 \
    LD_LIBRARY_PATH="${BUILD_DIR}:${MOJO_RUNTIME}:/.singularity.d/libs:${LD_LIBRARY_PATH:-}" \
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
      START=$(date +%s.%N)
      env OMP_NUM_THREADS="${THREADS}" \
        TRIP_FDCB_MOJO=1 TRIP_FDCB_MOJO_MATRIX=1 TRIP_FDCB_MOJO_DIRECT=1 \
        TRIP_FDCB_MOJO_DEVICE_BOOTSTRAP=1 \
        "${TRIP_BINARY}" < "${WORK}/mojo.exec" > "${OUTDIR}/run.log" 2>&1
      END=$(date +%s.%N)
      awk -v start="${START}" -v end="${END}" \
        '"'"'BEGIN { printf("mojo_wall=%.3f\n", end-start) }'"'"' \
        > "${OUTDIR}/mojo.time"
    '

cat "${OUTDIR}/mojo.time"
grep -E 'FDCB Mojo physical matrix|FDCB Mojo pack|FDCB Mojo optimize|FDCB Mojo:|OptVoxelSetup:|OptCmd:' \
  "${OUTDIR}/run.log" | tail -30
for field in 1 2; do
  name="StaticP101_2110_field${field}_iGTV_R.rst"
  if cmp -s "${REFERENCE}/${name}" "${OUTDIR}/${name}"; then
    echo "field${field}_byte_identical=yes"
  else
    echo "field${field}_byte_identical=no"
  fi
  "${REPO}/build/h200/compare_rst_particles" \
    "${REFERENCE}/${name}" "${OUTDIR}/${name}"
done
