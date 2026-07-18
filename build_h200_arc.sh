#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:?set ROOT to your cluster scratch directory (e.g. /scratch/$USER/trip)}"
REPO="${REPO:-${ROOT}/trip-mojo}"
TRIP="${TRIP:-${ROOT}/trip_arc}"
CONTAINER="${CONTAINER:-${ROOT}/trip-dev.sif}"
UV="${UV:-${ROOT}/bin/uv}"
THREADS="${THREADS:-12}"
BUILD_MOJO="${BUILD_MOJO:-1}"
BUILD_TRIP="${BUILD_TRIP:-1}"
MOJO_BUILD="${REPO}/build/h200-arc"
TRIP_BUILD="${TRIP}/build-mojo-h200"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  TRIP_COMPILE_DIR="${TRIP_COMPILE_DIR:-${SLURM_TMPDIR}/trip-mojo-arc-build}"
else
  TRIP_COMPILE_DIR="${TRIP_COMPILE_DIR:-${TRIP_BUILD}}"
fi

test "$(git -C "${TRIP}" rev-parse HEAD)" = \
  "347494d4ebe6286c5ba06107e4cf848f93209795"
git -C "${TRIP}" apply --reverse --check --exclude=CMakeLists.txt \
  --exclude=trpopt.c "${REPO}/integration/trip_temp/trip_temp_mojo.patch"
git -C "${TRIP}" apply --reverse --check \
  "${REPO}/integration/trip_temp/trip_arc_dev_mojo.patch"
git -C "${TRIP}" apply --reverse --check \
  "${REPO}/integration/trip_temp/trip_arc_field_setup.patch"
git -C "${TRIP}" apply --reverse --check \
  "${REPO}/integration/trip_temp/trip_temp_clinical_dose.patch"
mkdir -p "${MOJO_BUILD}" "${TRIP_BUILD}" "${TRIP_COMPILE_DIR}" \
  "${ROOT}/.tmp" "${ROOT}/.uv-python"

apptainer exec -B "${BIND:-/lustre}:${BIND:-/lustre}" "${CONTAINER}" env \
  UV_CACHE_DIR="${ROOT}/.uv-cache" \
  UV_PYTHON_INSTALL_DIR="${ROOT}/.uv-python" TMPDIR="${ROOT}/.tmp" \
  CCACHE_DISABLE=1 \
  bash --noprofile --norc -lc "
    set -euo pipefail
    cd '${REPO}'
    if [[ '${BUILD_MOJO}' == 1 ]]; then
      '${UV}' sync --frozen
      .venv/bin/mojo build -I . -O3 -g1 \
        --target-accelerator sm_90a --emit shared-lib fdcb_abi.mojo \
        -Xlinker -lm -o '${MOJO_BUILD}/libtrip_fdcb_mojo.so'
    fi
    test -f '${MOJO_BUILD}/libtrip_fdcb_mojo.so'
    if [[ '${BUILD_TRIP}' == 1 ]]; then
      cmake -S '${TRIP}' -B '${TRIP_COMPILE_DIR}' \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_C_COMPILER_WORKS=TRUE -DCMAKE_CXX_COMPILER_WORKS=TRUE \
        -DCMAKE_C_ABI_COMPILED=TRUE -DCMAKE_CXX_ABI_COMPILED=TRUE \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE='-O3 -DNDEBUG' \
        -DTRIP_MOJO_ROOT='${REPO}' \
        -DTRIP_MOJO_LIBRARY='${MOJO_BUILD}/libtrip_fdcb_mojo.so'
      cmake --build '${TRIP_COMPILE_DIR}' -j'${THREADS}'
      if [[ '${TRIP_COMPILE_DIR}' != '${TRIP_BUILD}' ]]; then
        cp -p '${TRIP_COMPILE_DIR}/TRiP98' '${TRIP_BUILD}/TRiP98'
      fi
    fi
    test -x '${TRIP_BUILD}/TRiP98'
    cc -O2 '${REPO}/tools/compare_rst_particles.c' -lm \
      -o '${MOJO_BUILD}/compare_rst_particles'
    cc -O2 '${REPO}/tools/compare_float_cubes.c' -lm \
      -o '${MOJO_BUILD}/compare_float_cubes'
  "

stat -c 'artifact %y %s %n' \
  "${MOJO_BUILD}/libtrip_fdcb_mojo.so" "${TRIP_BUILD}/TRiP98"
