#!/usr/bin/env bash
#SBATCH -J mojo-h200-4d
#SBATCH -o /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/h200_4d_%j.out
#SBATCH -e /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/h200_4d_%j.err
#SBATCH -D /lustre/bio/mdick/CUDA
#SBATCH -p nvidia_gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=140G
#SBATCH --time=00:10:00

set -euo pipefail

ROOT="${ROOT:-/lustre/bio/mdick/CUDA}"
REPO="${REPO:-${ROOT}/trip-mojo}"
TRIP="${TRIP:-${ROOT}/trip_temp}"
PROFILES="${PROFILES:-${ROOT}/PROFILES/trip-mojo}"
CONTAINER="${CONTAINER:-${ROOT}/trip-dev.sif}"
THREADS="${THREADS:-12}"
CONTROL_THREADS="${CONTROL_THREADS:-32}"
PACK_THREADS="${PACK_THREADS:-16}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTDIR="${PROFILES}/h200_4d_${JOB_ID}"
WORK="${SLURM_TMPDIR:-/tmp}/trip-mojo-h200-4d-${JOB_ID}"
CASE="${ROOT}/TRIP_DATA/P101_4Dopt"
EXEC_SRC="${CASE}/exec/P101_ITV_full4DITVplan.lustre.exec"
REFERENCE="${REFERENCE_DIR:-${PROFILES}/cpu_4d_ref_21714}"
MOJO_BUILD="${REPO}/build/h200"
TRIP_BINARY="${TRIP}/build-mojo-h200/TRiP98"
MOJO_RUNTIME="${REPO}/.venv/lib/python3.12/site-packages/modular/lib"

mkdir -p "${OUTDIR}/mojo" "${WORK}"
test -x "${TRIP_BINARY}"
test -f "${MOJO_BUILD}/libtrip_fdcb_mojo.so"
test -f "${EXEC_SRC}"
for field in 1 2; do
  test -f "${REFERENCE}/4DP101_field${field}.rst"
done

sed \
  -e "s/maxthreads(50)/maxthreads(${CONTROL_THREADS})/g" \
  -e "s/maxthreads(40)/maxthreads(${CONTROL_THREADS})/g" \
  -e "s|field 1 / write file(.*)|field 1 / write file(${OUTDIR}/mojo/4DP101_field1.rst)|" \
  -e "s|field 2 / write file(.*)|field 2 / write file(${OUTDIR}/mojo/4DP101_field2.rst)|" \
  "${EXEC_SRC}" > "${WORK}/mojo.exec"

echo "job=${JOB_ID} host=$(hostname) optimizer_threads=${THREADS} control_threads=${CONTROL_THREADS} pack_threads=${PACK_THREADS} output=${OUTDIR}"
echo "trip_temp_commit=$(git -C "${TRIP}" rev-parse HEAD)"
echo "trip_mojo_commit=$(git -C "${REPO}" rev-parse HEAD)"
nvidia-smi

nvidia-smi \
  --query-gpu=timestamp,name,power.draw,utilization.gpu,utilization.memory,memory.used,temperature.gpu \
  --format=csv -l 1 > "${OUTDIR}/power.csv" &
POWER_PID=$!
cleanup() {
  kill "${POWER_PID}" 2>/dev/null || true
  wait "${POWER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

RUNTIME_PATH="${MOJO_BUILD}:${MOJO_RUNTIME}:/.singularity.d/libs:${LD_LIBRARY_PATH:-}"
START=$(date +%s.%N)
apptainer exec --nv -B /lustre:/lustre -B "${WORK}:${WORK}" \
  "${CONTAINER}" env \
    OMP_NUM_THREADS="${CONTROL_THREADS}" LD_LIBRARY_PATH="${RUNTIME_PATH}" \
    TRIP_FDCB_MOJO_PACK_THREADS="${PACK_THREADS}" \
    MODULAR_CRASH_REPORTING_ENABLED=false \
    MODULAR_DEVICE_CONTEXT_SYNC_MODE=true \
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=80 \
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=80 \
    TRIP_FDCB_MOJO=1 TRIP_FDCB_MOJO_MATRIX=1 TRIP_FDCB_MOJO_DIRECT=1 \
    TRIP_FDCB_MOJO_DEVICE_BOOTSTRAP=1 \
    "${TRIP_BINARY}" < "${WORK}/mojo.exec" \
    > "${OUTDIR}/mojo/run.log" 2>&1
END=$(date +%s.%N)
awk -v start="${START}" -v end="${END}" \
  'BEGIN { printf("mojo_wall=%.3f\n", end-start) }' | tee "${OUTDIR}/mojo.time"

cleanup
trap - EXIT

grep -E 'FDCB Mojo physical matrix|FDCB Mojo pack|FDCB Mojo optimize|FDCB Mojo:|OptVoxelSetup:|OptCmd:' \
  "${OUTDIR}/mojo/run.log" | tail -40
for field in 1 2; do
  native="${REFERENCE}/4DP101_field${field}.rst"
  candidate="${OUTDIR}/mojo/4DP101_field${field}.rst"
  if cmp -s "${native}" "${candidate}"; then
    echo "field${field}_byte_identical=yes"
  else
    echo "field${field}_byte_identical=no"
  fi
  "${MOJO_BUILD}/compare_rst_particles" "${native}" "${candidate}"
done
