#!/usr/bin/env bash
set -euo pipefail

script=$(realpath "${BASH_SOURCE[0]}")
repo=$(dirname "$script")
root=$(dirname "$repo")
arc=0
case ${1:-} in
  "") ;;
  --arc) arc=1 ;;
  -h|--help) echo "usage: $0 [--arc]"; exit ;;
  *) echo "usage: $0 [--arc]" >&2; exit 2 ;;
esac

trip="$root/trip_temp"
mojo_build="$repo/build/h200"
trip_build="$trip/build-mojo-h200"
trip_compile=$trip_build
commit=1fb423f62b76a18b13b4ffd43e8dde55d004e9b5
cmake_probe_flags=
if ((arc)); then
  trip="$root/trip_arc"
  mojo_build="$repo/build/h200-arc"
  trip_build="$trip/build-mojo-h200"
  trip_compile=${SLURM_TMPDIR:-$trip_build}/trip-mojo-arc-build
  commit=347494d4ebe6286c5ba06107e4cf848f93209795
  cmake_probe_flags='-DCMAKE_C_COMPILER_WORKS=TRUE -DCMAKE_CXX_COMPILER_WORKS=TRUE -DCMAKE_C_ABI_COMPILED=TRUE -DCMAKE_CXX_ABI_COMPILED=TRUE'
fi

test "$(git -C "$trip" rev-parse HEAD)" = "$commit"
if ((arc)); then
  git -C "$trip" apply --reverse --check --exclude=CMakeLists.txt \
    --exclude=trpopt.c "$repo/integration/trip_temp/trip_temp_mojo.patch"
  for patch in trip_arc_dev_mojo.patch trip_arc_field_setup.patch \
    trip_temp_clinical_dose.patch; do
    git -C "$trip" apply --reverse --check \
      "$repo/integration/trip_temp/$patch"
  done
else
  for patch in trip_temp_mojo.patch trip_temp_clinical_dose.patch \
    trip_temp_robust_scenarios.patch; do
    git -C "$trip" apply --reverse --check --ignore-space-change \
      --ignore-whitespace --whitespace=nowarn \
      "$repo/integration/trip_temp/$patch"
  done
fi

mkdir -p "$mojo_build" "$trip_build" "$trip_compile" "$root/.tmp" \
  "$root/.uv-python" "$root/.ccache-trip-mojo"

apptainer exec -B "$root:$root" "$root/trip-dev.sif" env \
  UV_CACHE_DIR="$root/.uv-cache" \
  UV_PYTHON_INSTALL_DIR="$root/.uv-python" TMPDIR="$root/.tmp" \
  CCACHE_DIR="$root/.ccache-trip-mojo" \
  bash --noprofile --norc -lc "
    set -euo pipefail
    cd '$repo'
    '$root/bin/uv' sync --frozen
    .venv/bin/mojo build -I . -O3 -g1 \
      --target-accelerator sm_90a --emit shared-lib trip_abi.mojo \
      -Xlinker -lm -o '$mojo_build/libtrip_mojo.so'
    cmake -S '$trip' -B '$trip_compile' \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      $cmake_probe_flags \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE='-O3 -DNDEBUG' \
      -DTRIP_MOJO_ROOT='$repo' \
      -DTRIP_MOJO_LIBRARY='$mojo_build/libtrip_mojo.so'
    cmake --build '$trip_compile' -j12
    if [[ '$trip_compile' != '$trip_build' ]]; then
      cp -p '$trip_compile/TRiP98' '$trip_build/TRiP98'
    fi
    cc -O2 '$repo/tools/compare_rst_particles.c' -lm \
      -o '$mojo_build/compare_rst_particles'
    cc -O2 '$repo/tools/compare_float_cubes.c' -lm \
      -o '$mojo_build/compare_float_cubes'
  "

test -f "$mojo_build/libtrip_mojo.so"
test -x "$trip_build/TRiP98"
printf '%s\n' "$mojo_build/libtrip_mojo.so" "$trip_build/TRiP98"
