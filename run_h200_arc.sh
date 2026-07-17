#!/usr/bin/env bash
#SBATCH -J mojo-h200-arc
#SBATCH -o /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/h200_arc_%j.out
#SBATCH -e /lustre/bio/mdick/CUDA/PROFILES/trip-mojo/h200_arc_%j.err
#SBATCH -D /lustre/bio/mdick/CUDA
#SBATCH -p nvidia_gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH -N 1
#SBATCH -c 72
#SBATCH --mem=240G
#SBATCH --time=01:00:00

set -euo pipefail

ROOT="${ROOT:-/lustre/bio/mdick/CUDA}"
REPO="${REPO:-${ROOT}/trip-mojo}"
TRIP="${TRIP:-${ROOT}/trip_arc}"
PROFILES="${PROFILES:-${ROOT}/PROFILES/trip-mojo}"
CONTAINER="${CONTAINER:-${ROOT}/trip-dev.sif}"
EXEC_SRC="${EXEC_SRC:-/lustre/bio/lvolz/CAK257/arc.exec}"
PATIENT_ROOT="${PATIENT_ROOT:-/lustre/biopatient/DATA/PatientData/GSI_HN/CAK257}"
BASEDATA="${BASEDATA:-${ROOT}/TRIP_DATA/Basedata}"
CONTROL_THREADS="${CONTROL_THREADS:-70}"
PACK_THREADS="${PACK_THREADS:-32}"
DOSE_FIELD="${DOSE_FIELD:-}"
DOSE_BACKEND="${DOSE_BACKEND:-accelerator}"
DIAG_NATIVE_DOSE="${DIAG_NATIVE_DOSE:-0}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTDIR="${PROFILES}/h200_arc_${JOB_ID}"
MOJO_BUILD="${REPO}/build/h200-arc"
TRIP_BINARY="${TRIP}/build-mojo-h200/TRiP98"
MOJO_RUNTIME="${REPO}/.venv/lib/python3.12/site-packages/modular/lib"
WORK_ROOT="${SLURM_TMPDIR:-/tmp}/trip-mojo-arc-${JOB_ID}"

mkdir -p "${OUTDIR}/reference" "${OUTDIR}/mojo"
mkdir -p "${WORK_ROOT}"
[[ -z "${DOSE_FIELD}" || "${DOSE_FIELD}" =~ ^[1-9][0-9]*$ ]]
[[ "${DOSE_BACKEND}" == accelerator || "${DOSE_BACKEND}" == cpu ]]
[[ "${DIAG_NATIVE_DOSE}" == 0 || "${DIAG_NATIVE_DOSE}" == 1 ]]
test -f "${EXEC_SRC}"
test -f "${PATIENT_ROOT}/CAK257000.hed"
test -f "${PATIENT_ROOT}/CAK257000.ctx"
test -f "${PATIENT_ROOT}/CAK257000.vdx"
test -f "${BASEDATA}/initGSI_12C_bio.exec"
test -f "${BASEDATA}/GSI/19990211.hlut"
test -f "${BASEDATA}/RBE/chordom02.rbe"
test -f "${BASEDATA}/RBE/hirn02.rbe"
TRIP_COMPILE_DIR="${TRIP_COMPILE_DIR:-${WORK_ROOT}/trip-build}" THREADS=12 \
  PACK_THREADS="${PACK_THREADS}" "${REPO}/build_h200_arc.sh"
RUNTIME_PATH="${MOJO_BUILD}:${MOJO_RUNTIME}:/.singularity.d/libs:${LD_LIBRARY_PATH:-}"

echo "job=${JOB_ID} host=$(hostname) control_threads=${CONTROL_THREADS} pack_threads=${PACK_THREADS} dose_field=${DOSE_FIELD:-all} dose_backend=${DOSE_BACKEND} output=${OUTDIR}"
echo "arc_commit=$(git -C "${TRIP}" rev-parse HEAD)"
echo "trip_mojo_commit=$(git -C "${REPO}" rev-parse HEAD)"
nvidia-smi

nvidia-smi --query-gpu=timestamp,name,power.draw,utilization.gpu,utilization.memory,memory.used,temperature.gpu \
  --format=csv -l 1 > "${OUTDIR}/power.csv" &
POWER_PID=$!
cleanup() {
  kill "${POWER_PID}" 2>/dev/null || true
  wait "${POWER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

run_case() {
  local name="$1"
  shift
  local start exec_end copy_end work log
  work="${WORK_ROOT}/${name}"
  log="${WORK_ROOT}/${name}.log"
  mkdir -p "${work}"
  local -a edits=(
    -e "s|/lustre/bio/DATA/Basedata/EXEC/initGSI_12C_bio.exec|${BASEDATA}/initGSI_12C_bio.exec|g"
    -e "s|/lustre/bio/DATA/Basedata|${BASEDATA}|g"
    -e "s|/lustre/bio/DATA/PatientData/GSI_HN/CAK257|${PATIENT_ROOT}|g"
    -e '/RBE_normaltissue_ab_2\.rbe/s|^|*|'
  )
  if [[ -n "${DOSE_FIELD}" ]]; then
    edits+=(-e "/^dose /s|$| field(${DOSE_FIELD})|")
  fi
  start=$(date +%s.%N)
  (
    cd "${work}"
    sed "${edits[@]}" "${EXEC_SRC}" | \
      apptainer exec --nv -B /lustre:/lustre -B "${WORK_ROOT}:${WORK_ROOT}" \
        "${CONTAINER}" env \
        OMP_NUM_THREADS="${CONTROL_THREADS}" \
        LD_LIBRARY_PATH="${RUNTIME_PATH}" \
        MODULAR_CRASH_REPORTING_ENABLED=false \
        "$@" "${TRIP_BINARY}"
  ) > "${log}" 2>&1
  exec_end=$(date +%s.%N)
  (
    cd "${work}"
    tar -cf "${OUTDIR}/${name}/rst.tar" -- ./*.rst
  )
  find "${work}" -maxdepth 1 -type f ! -name '*.rst' \
    -exec cp -p -t "${OUTDIR}/${name}" -- {} +
  cp "${log}" "${OUTDIR}/${name}.log"
  copy_end=$(date +%s.%N)
  if grep -Eq '<E>|<SYS>|cannot open file|No such file or directory' \
    "${log}"; then
    grep -nE '<E>|<SYS>|cannot open file|No such file or directory' \
      "${log}" | tail -40 >&2
    return 1
  fi
  awk -v name="${name}" -v start="${start}" -v exec_end="${exec_end}" \
    -v copy_end="${copy_end}" \
    'BEGIN {
       printf("%s_process_wall=%.3f\n", name, exec_end-start)
       printf("%s_output_copy_wall=%.3f\n", name, copy_end-exec_end)
       printf("%s_wall=%.3f\n", name, copy_end-start)
     }' | tee "${OUTDIR}/${name}.time"
  du -sb "${work}" | awk -v name="${name}" '{print name "_output_bytes=" $1}' | \
    tee -a "${OUTDIR}/${name}.time"
}

run_case reference
test "$(find "${WORK_ROOT}/reference" -maxdepth 1 -type f -name '*.rst' | wc -l)" -gt 0
test "$(tar -tf "${OUTDIR}/reference/rst.tar" | wc -l)" = \
     "$(find "${WORK_ROOT}/reference" -maxdepth 1 -type f -name '*.rst' | wc -l)"
test "$(find "${OUTDIR}/reference" -maxdepth 1 -type f -name 'NormalArcDose*' | wc -l)" -gt 0
mojo_env=(
  TRIP_FDCB_MOJO=1 TRIP_FDCB_MOJO_MATRIX=1 TRIP_FDCB_MOJO_DIRECT=1
  TRIP_FDCB_MOJO_DEVICE_BOOTSTRAP=1
  TRIP_FDCB_MOJO_PACK_THREADS="${PACK_THREADS}"
  TRIP_CLINICAL_DOSE_MOJO=1
  MODULAR_DEVICE_CONTEXT_SYNC_MODE=true
  MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=80
  MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=80
)
if [[ "${DIAG_NATIVE_DOSE}" == 1 ]]; then
  mkdir -p "${OUTDIR}/mojo_native_dose"
  run_case mojo_native_dose "${mojo_env[@]:0:5}"
fi
if [[ "${DOSE_BACKEND}" == accelerator ]]; then
  mojo_env+=(TRIP_CLINICAL_DOSE_MOJO_ACCELERATOR=1)
fi
run_case mojo "${mojo_env[@]}"
grep -q 'FDCB Mojo optimize' "${OUTDIR}/mojo.log"
grep -q "Mojo clinical dose: backend=${DOSE_BACKEND} states=1" "${OUTDIR}/mojo.log"
test "$(find "${WORK_ROOT}/mojo" -maxdepth 1 -type f -name '*.rst' | wc -l)" -gt 0
test "$(tar -tf "${OUTDIR}/mojo/rst.tar" | wc -l)" = \
     "$(find "${WORK_ROOT}/mojo" -maxdepth 1 -type f -name '*.rst' | wc -l)"
test "$(find "${OUTDIR}/mojo" -maxdepth 1 -type f -name 'NormalArcDose*' | wc -l)" -gt 0

cleanup
trap - EXIT

grep -E 'FDCB Mojo physical matrix|FDCB Mojo pack|FDCB Mojo optimize|FDCB Mojo:|Mojo clinical dose:|OptVoxelSetup:|OptCmd:|DoseCmd:' \
  "${OUTDIR}/reference.log" "${OUTDIR}/mojo.log" | tail -80

matched=0
different=0
while IFS= read -r native; do
  candidate="${WORK_ROOT}/mojo/$(basename "${native}")"
  test -f "${candidate}" || continue
  matched=$((matched + 1))
  cmp -s "${native}" "${candidate}" || different=$((different + 1))
done < <(find "${WORK_ROOT}/reference" -maxdepth 1 -type f -name '*.rst' | sort)
echo "rst_files=${matched} rst_byte_different=${different}"

for kind in phys bio; do
  gzip -cd "${OUTDIR}/reference/NormalArcDose.${kind}.dos.gz" > \
    "${OUTDIR}/reference.${kind}.dos"
  gzip -cd "${OUTDIR}/mojo/NormalArcDose.${kind}.dos.gz" > \
    "${OUTDIR}/mojo.${kind}.dos"
  printf '%s ' "${kind}"
  "${MOJO_BUILD}/compare_float_cubes" \
    "${OUTDIR}/reference.${kind}.dos" "${OUTDIR}/mojo.${kind}.dos"
done
if [[ "${DIAG_NATIVE_DOSE}" == 1 ]]; then
  gzip -cd "${OUTDIR}/mojo_native_dose/NormalArcDose.phys.dos.gz" > \
    "${OUTDIR}/mojo_native_dose.phys.dos"
  printf 'reference_vs_mojo_native_dose '
  "${MOJO_BUILD}/compare_float_cubes" \
    "${OUTDIR}/reference.phys.dos" "${OUTDIR}/mojo_native_dose.phys.dos"
  printf 'mojo_native_dose_vs_mojo '
  "${MOJO_BUILD}/compare_float_cubes" \
    "${OUTDIR}/mojo_native_dose.phys.dos" "${OUTDIR}/mojo.phys.dos"
fi
gzip -cd "${OUTDIR}/reference/NormalArcDose.dosemlet.dos.gz" > \
  "${OUTDIR}/reference.dosemlet.dos"
gzip -cd "${OUTDIR}/mojo/NormalArcDose.dosemlet.dos.gz" > \
  "${OUTDIR}/mojo.dosemlet.dos"
printf 'dosemeanlet '
"${MOJO_BUILD}/compare_float_cubes" \
  "${OUTDIR}/reference.dosemlet.dos" "${OUTDIR}/mojo.dosemlet.dos"
if cmp -s "${OUTDIR}/reference/NormalAcDVH.dvh.gd" \
          "${OUTDIR}/mojo/NormalAcDVH.dvh.gd"; then
  echo "dvh_byte_identical=yes"
else
  echo "dvh_byte_identical=no"
fi

find "${OUTDIR}/reference" "${OUTDIR}/mojo" -maxdepth 1 -type f \
  \( -name 'NormalArcDose*' -o -name 'NormalAcDVH*' \) \
  -printf '%s %p\n' | sort
