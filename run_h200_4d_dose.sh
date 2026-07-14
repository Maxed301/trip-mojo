#!/usr/bin/env bash
#SBATCH -J mojo-h200-4d-dose
#SBATCH -o /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/h200_4d_dose_%j.out
#SBATCH -e /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/h200_4d_dose_%j.err
#SBATCH -D /lustre/bio/mdick/CUDA
#SBATCH -p nvidia_gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=140G
#SBATCH --time=01:00:00

set -euo pipefail

ROOT="${ROOT:-/lustre/bio/mdick/CUDA}"
REPO="${REPO:-${ROOT}/trip-mojo}"
TRIP="${TRIP:-${ROOT}/trip_temp}"
PROFILES="${PROFILES:-${ROOT}/PROFILES/trip-mojo}"
CONTAINER="${CONTAINER:-${ROOT}/trip-dev.sif}"
THREADS="${THREADS:-32}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTDIR="${PROFILES}/h200_4d_dose_${JOB_ID}"
WORK="${SLURM_TMPDIR:-/tmp}/trip-mojo-h200-4d-dose-${JOB_ID}"
CASE="${ROOT}/TRIP_DATA/P101_4Dopt"
EXEC_SRC="${CASE}/exec/P101_ITV_full4DITVplan.lustre.exec"
REFERENCE="${REFERENCE_DIR:-${PROFILES}/cpu_4d_ref_21714}"
MOJO_BUILD="${REPO}/build/h200"
TRIP_BINARY="${TRIP}/build-mojo-h200/TRiP98"
MOJO_RUNTIME="${REPO}/.venv/lib/python3.12/site-packages/modular/lib"

mkdir -p "${OUTDIR}/reference" "${OUTDIR}/mojo" "${WORK}"
for path in "${TRIP_BINARY}" "${MOJO_BUILD}/libtrip_fdcb_mojo.so" \
  "${MOJO_BUILD}/compare_float_cubes" "${EXEC_SRC}" \
  "${REFERENCE}/4DP101_field1.rst" "${REFERENCE}/4DP101_field2.rst"; do
  test -e "${path}"
done

make_exec() {
  local mode="$1" stem="$2"
  local file="${WORK}/${mode}.exec"
  sed '/^field 1 \/ new/,$d' "${EXEC_SRC}" > "${file}"
  {
    printf 'field 1 / read rst(%s/4DP101_field1.rst)\n' "${REFERENCE}"
    printf 'field 2 / read rst(%s/4DP101_field2.rst)\n' "${REFERENCE}"
    printf 'field 1 / perfectrescan statelimits(*)\n'
    printf 'field 2 / perfectrescan statelimits(*)\n'
    printf 'voi "Tumor" / writeComTrafo prefix(%s/%s_)\n' "${WORK}" "${mode}"
    printf 'trafo "%s/%s_ComTranslation" / read\n' "${WORK}" "${mode}"
    printf 'dose "%s" / calc bio bioalg(ld) alg(msdb) direct nosvv norbe datatype(float) voi(Tumor) write maxthreads(%s)\n' "${stem}" "${THREADS}"
    printf 'quit\n'
  } >> "${file}"
}

make_exec reference "${OUTDIR}/reference/P101_4D"
make_exec mojo "${OUTDIR}/mojo/P101_4D"

RUNTIME_PATH="${MOJO_BUILD}:${MOJO_RUNTIME}:/.singularity.d/libs:${LD_LIBRARY_PATH:-}"
run_case() {
  local mode="$1" start end
  start=$(date +%s.%N)
  if [[ "${mode}" == mojo ]]; then
    apptainer exec --nv -B /lustre:/lustre -B "${WORK}:${WORK}" \
      "${CONTAINER}" env OMP_NUM_THREADS="${THREADS}" \
      LD_LIBRARY_PATH="${RUNTIME_PATH}" MODULAR_CRASH_REPORTING_ENABLED=false \
      MODULAR_DEVICE_CONTEXT_SYNC_MODE=true TRIP_CLINICAL_DOSE_MOJO=1 \
      TRIP_CLINICAL_DOSE_MOJO_ACCELERATOR=1 \
      "${TRIP_BINARY}" < "${WORK}/${mode}.exec" > "${OUTDIR}/${mode}/run.log" 2>&1
  else
    apptainer exec --nv -B /lustre:/lustre -B "${WORK}:${WORK}" \
      "${CONTAINER}" env OMP_NUM_THREADS="${THREADS}" \
      LD_LIBRARY_PATH="${RUNTIME_PATH}" MODULAR_CRASH_REPORTING_ENABLED=false \
      "${TRIP_BINARY}" < "${WORK}/${mode}.exec" > "${OUTDIR}/${mode}/run.log" 2>&1
  fi
  end=$(date +%s.%N)
  awk -v mode="${mode}" -v start="${start}" -v end="${end}" \
    'BEGIN { printf("%s_wall=%.3f\n", mode, end-start) }' | tee "${OUTDIR}/${mode}.time"
}

echo "job=${JOB_ID} host=$(hostname) threads=${THREADS} output=${OUTDIR}"
echo "trip_temp_commit=$(git -C "${TRIP}" rev-parse HEAD)"
echo "trip_mojo_commit=$(git -C "${REPO}" rev-parse HEAD)"
nvidia-smi
run_case reference
run_case mojo

for type in phys bio; do
  "${MOJO_BUILD}/compare_float_cubes" \
    "${OUTDIR}/reference/P101_4D.${type}.dos" \
    "${OUTDIR}/mojo/P101_4D.${type}.dos"
done
grep -E 'DoseCmd:|Mojo clinical dose|error|Error|<E>' \
  "${OUTDIR}/reference/run.log" "${OUTDIR}/mojo/run.log" | tail -80 || true
