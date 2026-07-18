#!/usr/bin/env bash
set -euo pipefail

usage() { echo "usage: $0 --vendor <nvidia|amd> [--gpus <1-3>] --plan <plan.exec>"; }
vendor_config() {
  case "$vendor" in
    nvidia) partition=nvidia_gpu constraint=h200 gres="gpu:nvidia_h200:$gpus"
      gpu_flag=--nv target=h200 rocm_path= ;;
    amd) partition=amd_gpu constraint=mi100 gres="gpu:$gpus"
      gpu_flag=--rocm target=mi100 rocm_path=/opt/rocm/lib:/opt/rocm/lib64: ;;
  esac
}
script=$(realpath "${BASH_SOURCE[0]}")
repo=$(dirname "$script")
if [[ ${1:-} == --run-job ]]; then
  vendor=$2 gpus=$3 plan=$4 apptainer=$5 repo=$6
fi
root=$(dirname "$repo")
trip="$root/trip_temp/build-mojo-h200/TRiP98"
container="$root/trip-dev.sif"
profiles="$root/PROFILES/trip-mojo"
if [[ ${1:-} == --run-job ]]; then
  vendor_config
  runtime=$(find "$repo/.venv/lib" -path '*/site-packages/modular/lib' -type d -print -quit)
  mkdir -p "$profiles/$SLURM_JOB_ID"
  cd "$profiles/$SLURM_JOB_ID"
  exec "$apptainer" exec "$gpu_flag" -B "$root:$root" "$container" env \
    OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" \
    LD_LIBRARY_PATH="$rocm_path$repo/build/$target:$runtime:/.singularity.d/libs" \
    "$trip" < "$plan"
fi
vendor= gpus=1 plan=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --vendor) vendor=$2; shift 2 ;;
    --gpus) gpus=$2; shift 2 ;;
    --plan) plan=$2; shift 2 ;;
    -h|--help) usage; exit ;;
    *) usage; exit 2 ;;
  esac
done
[[ $vendor =~ ^(nvidia|amd)$ && $gpus =~ ^[1-3]$ && -f $plan ]] || { usage; exit 2; }
plan=$(realpath "$plan")
apptainer=$(command -v apptainer)
vendor_config
mkdir -p "$profiles"
job=$(sbatch --parsable --export=NONE --job-name="trip-mojo-$vendor" \
  --nodes=1 --cpus-per-task=32 --mem=160G --time=00:10:00 \
  --partition="$partition" --constraint="$constraint" --gres="$gres" \
  --output="$profiles/%j.out" \
  "$script" --run-job "$vendor" "$gpus" "$plan" "$apptainer" "$repo")
echo "submitted $job -> $profiles/${job%%;*}.out"
