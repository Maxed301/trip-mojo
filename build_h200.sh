#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:?set ROOT to your cluster scratch directory (e.g. /scratch/$USER/trip)}"
REPO="${REPO:-${ROOT}/trip-mojo}"
TRIP="${TRIP:-${ROOT}/trip_temp}"
CONTAINER="${CONTAINER:-${ROOT}/trip-dev.sif}"
UV="${UV:-${ROOT}/bin/uv}"
THREADS="${THREADS:-12}"
MOJO_BUILD="${REPO}/build/h200"
TRIP_BUILD="${TRIP}/build-mojo-h200"
PATCH="${REPO}/integration/trip_temp/trip_temp_mojo.patch"
DOSE_PATCH="${REPO}/integration/trip_temp/trip_temp_clinical_dose.patch"
ROBUST_PATCH="${REPO}/integration/trip_temp/trip_temp_robust_scenarios.patch"

test "$(git -C "${TRIP}" rev-parse HEAD)" = \
  "1fb423f62b76a18b13b4ffd43e8dde55d004e9b5"
git -C "${TRIP}" apply --reverse --check --ignore-space-change \
  --ignore-whitespace --whitespace=nowarn "${PATCH}"
git -C "${TRIP}" apply --reverse --check --ignore-space-change \
  --ignore-whitespace --whitespace=nowarn "${DOSE_PATCH}"
git -C "${TRIP}" apply --reverse --check --ignore-space-change \
  --ignore-whitespace --whitespace=nowarn "${ROBUST_PATCH}"
mkdir -p "${MOJO_BUILD}" "${ROOT}/.tmp" "${ROOT}/.uv-python" \
  "${ROOT}/.ccache-trip-mojo"

apptainer exec -B "${BIND:-/lustre}:${BIND:-/lustre}" "${CONTAINER}" env \
  UV_CACHE_DIR="${ROOT}/.uv-cache" \
  UV_PYTHON_INSTALL_DIR="${ROOT}/.uv-python" TMPDIR="${ROOT}/.tmp" \
  CCACHE_DIR="${ROOT}/.ccache-trip-mojo" \
  bash --noprofile --norc -lc "
    set -euo pipefail
    cd '${REPO}'
    '${UV}' sync --frozen
    .venv/bin/mojo build -I . -O3 -g1 \
      --target-accelerator sm_90a --emit shared-lib fdcb_abi.mojo \
      -Xlinker -lm -o '${MOJO_BUILD}/libtrip_fdcb_mojo.so'
    cmake -S '${TRIP}' -B '${TRIP_BUILD}' \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_FLAGS_RELEASE='-O3 -DNDEBUG' \
      -DTRIP_MOJO_ROOT='${REPO}' \
      -DTRIP_MOJO_LIBRARY='${MOJO_BUILD}/libtrip_fdcb_mojo.so'
    cmake --build '${TRIP_BUILD}' -j'${THREADS}'
    cc -O2 '${REPO}/tools/compare_rst_particles.c' -lm \
      -o '${MOJO_BUILD}/compare_rst_particles'
    cc -O2 '${REPO}/tools/compare_float_cubes.c' -lm \
      -o '${MOJO_BUILD}/compare_float_cubes'
  "

stat -c 'artifact %y %s %n' \
  "${MOJO_BUILD}/libtrip_fdcb_mojo.so" "${TRIP_BUILD}/TRiP98"
