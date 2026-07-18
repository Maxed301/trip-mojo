# trip-mojo

Portable Mojo backends for TRiP's optimizer and prepared clinical-dose
calculation. This is not a TRiP rewrite: TRiP keeps parsing, geometry, VOIs,
WET/setup, dose storage and output.

The sole correctness and performance reference is the clean
`~/Projects/trip_temp` checkout at commit `1fb423f`. See `AGENTS.md` for the
canonical local and cluster paths.

## Status

| Path | Current validation |
|---|---|
| CPU optimizer | P101 3D robust biological plan stops at iteration 271 and writes byte-identical RSTs. |
| NVIDIA optimizer | P101 ten-state, nine-scenario 4D robust plan stops at iteration 120 and writes byte-identical RSTs on H200. |
| NVIDIA multi-GPU | Direct coefficient-balanced shards remain byte-identical; oversized shards automatically retain topology and reconstruct exact Float64 coefficients. |
| CPU dose | Full static P101 physical/biological cubes match within one/two Float32 output ULPs. |
| NVIDIA dose | Same complete static case and support validated on H200. |
| 4D dose | P101 ten-state perfect-rescan physical/biological cubes are byte-identical to `trip_temp` on H200. |
| ARC | The head-and-neck ARC case stops at iteration 70 and writes 180 byte-identical RSTs on H200; physical and biological dose now agree at Float64 roundoff. |
| AMD | P101 3D stops at iteration 271 with byte-identical RSTs and DVH on MI100; dose agrees at Float32 output-rounding level. |
| Apple | Kernel lowering has been probed. Runtime validation remains experimental and lowest priority. |

Canonical measurements use 12 optimizer threads. The validated H200 runner
reserves 32 threads for TRiP setup and uses 16 for metadata packing. Direct
matrix construction now shards whole voxels across one, two or three devices
without staging the coefficient matrix in host memory. Full details are in
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
$MOJO build -I . -O3 -g1 --emit shared-lib \
  trip_abi.mojo -Xlinker -lm -o build/libtrip_mojo.so

# CPU tests run without an accelerator:
for test in test_optimization_problem test_cpu_backend test_optimizer \
  test_minimum_particles; do
  $MOJO run -I . -O3 tests/$test.mojo
done
# Requires one visible accelerator:
$MOJO run -I . -O3 tests/test_device_backend.mojo
# Require two and three visible accelerators, respectively:
$MOJO run -I . -O3 tests/test_two_devices.mojo
$MOJO run -I . -O3 tests/test_three_devices.mojo

cc -O2 -Iinclude tests/ffi/test_optimizer_abi.c -Lbuild -ltrip_mojo \
  -Wl,-rpath,'$ORIGIN' -o build/test_optimizer_abi
cc -O2 -Iinclude tests/ffi/test_clinical_dose_abi.c -Lbuild \
  -ltrip_mojo -Wl,-rpath,'$ORIGIN' -o build/test_clinical_dose_abi
build/test_optimizer_abi && build/test_clinical_dose_abi
```

On the cluster, a prebuilt Mojo-linked TRiP plan is submitted with no backend feature
flags:

```bash
./submit.sh --vendor amd --gpus 2 --plan plan.exec
```

`--gpus` defaults to one. `submit.sh` is the only job entry point; plans select
the TRiP workload and Slurm's visible devices select the accelerator count.
The requested devices are discovered from Slurm's standard visibility. The
Mojo-linked executable always uses the Mojo optimizer and clinical-dose
backend. NVIDIA uses direct accelerator matrix storage, including per-device
matrix ownership for multi-GPU jobs. AMD retains TRiP's canonical CPU-built
Float64 matrix. Native comparisons require a separate unpatched `trip_temp`
executable.

The cluster uses one persistently patched `trip_temp` build. Apply
`integration/trip_temp/trip_temp_mojo.patch` and
`integration/trip_temp/trip_temp_clinical_dose.patch`, followed by
`integration/trip_temp/trip_temp_robust_scenarios.patch`, once with Git's
`--ignore-space-change --ignore-whitespace` options because the canonical tree
contains CRLF files. Use `build_h200.sh` after source changes, then submit every
plan through `submit.sh`. Submission never copies or rebuilds either repository.

ARC validation uses a linked `trip_temp` `Arc-dev` worktree at commit
`347494d4`. Apply the common optimizer patch excluding `CMakeLists.txt` and
`trpopt.c`, then apply `trip_arc_dev_mojo.patch`,
`trip_arc_field_setup.patch` and the clinical-dose patch.
`build_h200_arc.sh` builds it. ARC plans and patient paths are supplied by the
operator and submitted through `submit.sh`; the Mojo-linked executable has no
runtime native fallback. A native comparison requires a separately built,
unpatched Arc-dev executable.
To avoid one Lustre metadata operation per arc field, each result directory
stores the 180 RSTs losslessly in `rst.tar`; dose and DVH files remain directly
accessible. The reported wall time includes this publication step. H200 job 22380 completed
the full reference in 104.529 s and Mojo in 49.179 s; those totals include
variable Lustre I/O from publishing each RST separately. With archived RST
publication, H200 job 22383 measured 21.664 s for the complete Mojo process and
2.954 s for publishing all results, or 24.617 s total. Cached Float64 HU-to-path tables reduced ARC `FieldCmd`
from 14.29/14.27 s in job 22379 to 2.77/2.78 s for reference/Mojo. All 180 RSTs
and the reference physical, biological and dose-mean-LET cubes remained
byte-identical to job 22379. Packed dose reduced Mojo `DoseCmd` from about 36 s
to about 2 s. Full-cube physical and biological relative L2 differences are
`1.39e-13` and `2.81e-11`; nonzero support is identical and the DVH is
byte-identical.

Dose-mean LET differs by `9.72e-4` relative L2 in 5,883 of 1,912,855 nonzero
voxels. This is isolated to Arc-dev retaining the previous
`sMFDCP.pdG[0]` value when a spectrum depth is out of range while zeroing the
other biological terms. Mojo deliberately returns zero for every out-of-range
table term instead of reproducing that stale-state behavior.

## Optimizer boundary

`optimization_problem.mojo` owns the backend-neutral `OptimizationProblem`. Its arrays are
flat and contiguous, and boundary indices have fixed widths:

- field slices partition the global particle vector;
- voxel/scenario records partition the slice array;
- slices reference contiguous coefficient ranges;
- UInt16 point indices are relative to a field slice.

The layout is identical for 3D and 4D. TRiP flattens selected motion-state
voxels before packing; robust scenario remains the inner numeric axis. Motion
loading, deformation and state-aware output stay in TRiP.

`problem_packing.mojo` converts convenient native test models into this numeric
layout. Production C integration avoids another full copy: Mojo allocates the
arrays, returns temporary writable pointers, and owns them until storage
destruction. The direct matrix boundary keeps UInt16 topology on the
accelerator and transfers only compact metadata into the optimizer. Shards up
to ten billion entries retain every Float64 coefficient; larger shards retain
geometry and reconstruct the same double-Gaussian coefficient on demand. After
accelerator setup, packed voxel/scenario/slice arrays that no iteration kernel
uses are released from host memory.

The CPU and accelerator evaluators share the host iteration controller,
stopping rules, Fletcher-Reeves updates, backtracking and host-side
minimum-particle policy. `complexminp` uses TRiP's captured 31-word libc RNG
state. The initialization update is not counted as a regular iteration.

The two-device evaluator partitions contiguous whole-voxel ranges near half
the sparse coefficient count. It replicates particles and field metadata,
keeps all robust scenarios for one voxel on one device, and sums partial
gradients, chi-square terms and exact-step terms on the host each iteration.
Sparse coefficient and slice arrays are not duplicated across devices.

MI100 reference mode packs TRiP's canonical CPU-built Float64 matrix and then
runs regular optimizer evaluations and clinical dose on gfx908. This avoids
vendor-specific device `exp` differences in matrix construction while keeping
the numeric backend shared. Hydra job 22559 reached the canonical 271-iteration
stop, wrote byte-identical RSTs and DVH, and measured 35.097 s for optimization,
17.47 s for `DoseCmd`, and 111.845 s process wall time. Full dose metrics are in
[`docs/mi100.md`](docs/mi100.md).

The implementation is Float64 throughout, matching the configured `trip_temp`
matrix and optimizer. Production builds cannot select a narrower precision.

## Clinical-dose boundary

TRiP supplies prepared grid, CT/HLUT, transforms, raster points, DDD and
biology tables. Mojo computes Siddon WET, interpolation, divergent
double-Gaussian transport, and Float64 accumulation. TRiP stores and writes the
resulting cubes.

The packed boundary supports static and state-major 4D input, `ms`/`msdb`, and
physical or low-dose biological dose. For 4D, TRiP transforms every reference
voxel into each state and supplies that state's CT and RST range. Mojo sums raw
Float64 dose terms in state order; TRiP then performs its unchanged dose
storage. Static P101 and the ten-state P101 perfect-rescan path are validated.
`tools/compare_float_cubes.c` reports raw Float32 cube support and error without
scaling or output shaping.

H200 job 22528 revalidated the current static path after restoring TRiP's field
dose-extension cutoff. Physical and biological support was identical; relative
L2 errors were `2.69e-10` and `1.26e-10`, with only 14 and 15 Float32 voxels
differing by one output ULP. The DVH was byte-identical. The clinical ABI also
carries its struct size, so a stale TRiP executable is rejected before any
shifted descriptor can be interpreted.

H200 job 22069 evaluated 41,446 packed reference-grid voxels over ten states and
20 flattened state fields. Both 39,059,456-voxel output cubes had identical
11,764-voxel support and were byte-identical to `trip_temp`. `DoseCmd` measured
10.62 s for Mojo and 8.17 s for `trip_temp`; this small Tumor-only workload is
currently dominated by packing and accelerator-transfer overhead.

## Explicit gaps

- deformable P101 4D-dose validation and smoothed/per-state dose output
- robust-RBE and OER models
- Arc-dev's stale out-of-range dose-mean-LET behavior, dynamic `arcangles` dose reconstruction, lung modulation and MLET output
- all-plan, master-field and static-field optimizer modes
- deterministic device matrix construction/bootstrap on MI100 and Apple runtime parity

Unsupported production modes must be rejected by the TRiP adapter, never
silently approximated. `.exec` parsing in this repository is only a fail-closed
setup probe; production command handling remains in TRiP.

MI100 toolchain details are in [`docs/mi100.md`](docs/mi100.md).
