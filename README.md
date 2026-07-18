# trip-mojo

Portable Mojo numeric backends for TRiP optimization and prepared clinical-dose
calculation. TRiP remains responsible for parsing, geometry, VOIs, WET, matrix
setup, storage, and output.

The sole CPU correctness reference is the clean `~/Projects/trip_temp`
checkout at commit `1fb423f`. ARC validation uses its linked `Arc-dev`
worktree at `347494d4`.

## Submit a plan

```bash
./submit.sh --vendor nvidia --plan plan.exec
./submit.sh --vendor amd --gpus 2 --plan plan.exec
```

`--gpus` defaults to one. The plan selects the TRiP workload and Slurm exposes
the requested accelerator count. Output is written to
`PROFILES/trip-mojo/<job-id>.out`.

## Build

```bash
./build.sh
./build.sh --arc
```

The build script derives the repository, cluster root, container, TRiP
worktree, compiler, thread count, and output paths from the canonical layout.

For a local shared library:

```bash
uv sync --frozen
mkdir -p build
.venv/bin/mojo build -I . -O3 -g1 --emit shared-lib trip_abi.mojo \
  -Xlinker -lm -o build/libtrip_mojo.so
```

## Test

```bash
for test in test_optimization_problem test_cpu_backend test_optimizer \
  test_minimum_particles; do
  .venv/bin/mojo run -I . -O3 tests/$test.mojo
done

# One visible accelerator:
.venv/bin/mojo run -I . -O3 tests/test_device_backend.mojo

# Two and three visible accelerators:
.venv/bin/mojo run -I . -O3 tests/test_two_devices.mojo
.venv/bin/mojo run -I . -O3 tests/test_three_devices.mojo

cc -O2 -Iinclude tests/ffi/test_optimizer_abi.c -Lbuild -ltrip_mojo \
  -Wl,-rpath,'$ORIGIN' -o build/test_optimizer_abi
cc -O2 -Iinclude tests/ffi/test_clinical_dose_abi.c -Lbuild -ltrip_mojo \
  -Wl,-rpath,'$ORIGIN' -o build/test_clinical_dose_abi
build/test_optimizer_abi
build/test_clinical_dose_abi
```

## Code map

| File | Responsibility |
|---|---|
| `optimization_problem.mojo` | Flat backend-neutral problem model |
| `cpu_backend.mojo` | Float64 correctness backend |
| `device_backend.mojo` | Shared NVIDIA/AMD optimizer kernels |
| `matrix_builder.mojo` | Direct accelerator matrix construction |
| `optimizer.mojo` | Shared iteration controller |
| `minimum_particles.mojo` | TRiP minimum-particle semantics |
| `clinical_dose*.mojo` | Packed clinical dose calculation |
| `trip_abi.mojo` | Thin C boundary |
| `tests/support/` | Synthetic reference models and problem builders |

## Validated paths

- CPU optimizer parity against `trip_temp`, including native stopping,
  particle numbers, and byte-identical RST output.
- H200 optimizer and clinical dose, including one to three devices.
- MI100 optimizer using canonical CPU-built Float64 matrices.
- Ten-state P101 4D robust optimization and perfect-rescan dose.
- Fixed-field ARC optimization through the normal optimizer backend.

Canonical optimizer comparisons use 12 threads. The H200 runner reserves 32
CPU threads for TRiP setup and uses 16 for metadata packing.

Detailed measurements and limitations are recorded in
[`docs/4d-robust.md`](docs/4d-robust.md) and
[`docs/mi100.md`](docs/mi100.md).

## TRiP integration

The cluster keeps persistent patched builds. Apply the common optimizer,
clinical-dose, and robust-scenario patches under `integration/trip_temp/` once
to the canonical master checkout. Apply the ARC patches only to the linked
`Arc-dev` worktree. Production command handling remains in TRiP.
