# trip-mojo

Portable Mojo backends for TRiP's FDCB optimizer and prepared clinical-dose
calculation. This is not a TRiP rewrite: TRiP keeps parsing, geometry, VOIs,
WET/setup, dose storage and output.

The sole correctness and performance reference is the clean
`~/Projects/trip_temp` checkout at commit `1fb423f`. See `AGENTS.md` for the
canonical local and Hydra paths.

## Status

| Path | Current validation |
|---|---|
| CPU FDCB | P101 3D robust biological plan stops at iteration 271 and writes byte-identical RSTs. |
| NVIDIA FDCB | P101 ten-state, nine-scenario 4D robust plan stops at iteration 120 and writes byte-identical RSTs on H200. |
| CPU dose | Full static P101 physical/biological cubes match within one/two Float32 output ULPs. |
| NVIDIA dose | Same complete static case and support validated on H200. |
| AMD | The shared kernels compile for gfx908 with the custom toolchain; queued MI100 runtime validation is not yet a parity claim. |
| Apple | Kernel lowering has been probed. Runtime validation remains experimental and lowest priority. |

Canonical measurements use 12 optimizer threads. The latest H200 4D run used
32 setup threads and 16 metadata-packing threads, completed in 27.841 s process
wall time versus 476.751 s for `trip_temp`, and produced two byte-identical
RSTs. Its full Mojo optimization call took 6.764 s. Full details are in
[`docs/4d-robust.md`](docs/4d-robust.md).

For the static dose case, CPU `DoseCmd` fell from 109.20 s to 63.29 s. H200
`DoseCmd` measured 15.52--19.31 s versus 236.39 s for the server CPU reference.
Physical and biological relative L2 differences are `5.88e-11` and `3.53e-11`;
maximum absolute errors are `1.49e-8` and `2.98e-8` Gy. Nonzero support is
identical.

## Build and test

```bash
uv sync --frozen
MOJO=.venv/bin/mojo
mkdir -p build
$MOJO build -I . -O3 -g1 -D FDCB_CPU_THREADS=12 --emit shared-lib \
  fdcb_abi.mojo -Xlinker -lm -o build/libtrip_fdcb_mojo.so

for test in test_fdcb_problem test_fdcb_cpu test_fdcb_optimize \
  test_fdcb_min_particles test_fdcb_accelerator; do
  $MOJO run -I . -O3 -D FDCB_CPU_THREADS=12 tests/$test.mojo
done
$MOJO run -I . -O3 -D FDCB_CPU_THREADS=12 -D FDCB_MIXED32=true \
  tests/test_fdcb_accelerator_mixed32.mojo
$MOJO run -I . -O3 tests/test_exec_parser.mojo

cc -O2 -Iinclude tests/ffi/test_fdcb_abi.c -Lbuild -ltrip_fdcb_mojo \
  -Wl,-rpath,'$ORIGIN' -o build/test_fdcb_abi
cc -O2 -Iinclude tests/ffi/test_clinical_dose_abi.c -Lbuild \
  -ltrip_fdcb_mojo -Wl,-rpath,'$ORIGIN' -o build/test_clinical_dose_abi
build/test_fdcb_abi && build/test_clinical_dose_abi
```

Hydra uses one persistently patched `trip_temp` build. Apply
`integration/trip_temp/trip_temp_mojo.patch` once with Git's
`--ignore-space-change --ignore-whitespace` options because the canonical tree
contains CRLF files. Use `build_h200.sh` after source changes and submit
`run_h200_4d.sh` for run-only validation. The runner does not copy or rebuild
either repository.

## Optimizer boundary

`fdcb_problem.mojo` owns the backend-neutral `FDCBProblemV1`. Its arrays are
flat and contiguous, and boundary indices have fixed widths:

- field slices partition the global particle vector;
- voxel/scenario records partition the slice array;
- slices reference contiguous coefficient ranges;
- UInt16 point indices are relative to a field slice.

The layout is identical for 3D and 4D. TRiP flattens selected motion-state
voxels before packing; robust scenario remains the inner numeric axis. Motion
loading, deformation and state-aware output stay in TRiP.

`fdcb_packing.mojo` converts convenient native test models into this numeric
layout. Production C integration avoids another full copy: Mojo allocates the
arrays, returns temporary writable pointers, and owns them until storage
destruction. The direct matrix boundary keeps coefficients and UInt16 indices
on the accelerator and transfers only compact metadata into the optimizer.

The CPU and accelerator evaluators share the host iteration controller,
stopping rules, Fletcher-Reeves updates, backtracking and host-side
minimum-particle policy. `complexminp` uses TRiP's captured 31-word libc RNG
state. The initialization update is not counted as a regular iteration.

Reference mode is Float64 throughout, matching the configured `trip_temp`
matrix and optimizer. `FDCB_MIXED32=true` is an explicit experimental build;
it cannot silently narrow a reference problem.

## Clinical-dose boundary

TRiP supplies prepared grid, CT/HLUT, transforms, raster points, DDD and
biology tables. Mojo computes Siddon WET, interpolation, divergent
double-Gaussian transport, and Float64 accumulation. TRiP stores and writes the
resulting cubes.

The validated dose boundary is static, one CT state, `ms`/`msdb`, physical or
low-dose biological. `tools/compare_float_cubes.c` reports raw Float32 cube
support and error without scaling or output shaping.

## Explicit gaps

- 4D clinical-dose deformation and accumulation
- robust-RBE and OER models
- arc fields, lung modulation and specialized LET outputs
- all-plan, master-field and static-field optimizer modes
- native MI100 and Apple runtime parity

Unsupported production modes must be rejected by the TRiP adapter, never
silently approximated. `.exec` parsing in this repository is only a fail-closed
setup probe; production command handling remains in TRiP.

MI100 toolchain details are in [`docs/mi100.md`](docs/mi100.md).
