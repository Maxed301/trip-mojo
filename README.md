# trip-mojo

`trip-mojo` is a portable, backend-neutral implementation of the TRiP FDCB
optimizer. It is not a rewrite of the complete TRiP application.

The intended production boundary keeps TRiP's existing C control plane and its
geometry, VOI, WET, dose-matrix preparation, parsing, and output code. That code
will pack one numeric `FDCBProblemV1` and pass it through a thin C ABI. Mojo owns
the optimizer math: first a CPU correctness backend, then shared accelerator
kernels for NVIDIA and AMD. Apple Metal remains an experimental, lowest-priority
target.

## Development

The toolchain is pinned in `uv.lock`; do not copy `.venv` or `build` between
machines.

```bash
uv sync --frozen
MOJO=.venv/bin/mojo
mkdir -p build
$MOJO build -I . -O3 -g1 -D FDCB_CPU_THREADS=12 --emit shared-lib \
  fdcb_abi.mojo -Xlinker -lm -o build/libtrip_fdcb_mojo.so
```

The focused milestone suite is:

```bash
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

The repository contains the backend and versioned C headers, not a patched TRiP
tree. The current end-to-end P101 measurements used a disposable adapter in a
scratch copy of `trip_temp`; production integration remains the next thin C
control-plane step.

## Backend boundary

`fdcb_problem.mojo` defines the owned V1 numeric contract. All variable-length
data is stored in flat contiguous arrays and all indices crossing the boundary
have explicit widths. Field slices own contiguous point ranges; voxel/scenario
records own contiguous slice ranges; each slice references a contiguous
coefficient range, but those ranges need not be globally monotonic because
TRiP-GPU builds them field-first. UInt16 coefficient point indices are relative
to their field slice. Each field slice also carries the raster row stride used
only by the host `complexminp` neighbor policy. No device pointer, runtime
handle, or vendor-specific assumption is present.

Point arrays include accepted particles, the current gradient, and search
direction. For an all-zero plan, TRiP packs its host-computed master-voxel
bootstrap direction, which is distinct from robust scenario 0. Native tests
without a master matrix use a target-only scenario-0 fallback.

`include/fdcb_abi_v1.h` is the matching C view. TRiP retains ownership of every
input buffer for the duration of a call. The V1 Mojo entry points validate and
copy that borrowed view into the owned model, then return only scalar results
and caller-sized particle or gradient arrays. This copy-first implementation is
the correctness boundary; borrowed backend buffers are a later optimization
that does not change the ABI or numeric model.

`trip_fdcb_optimize_v1` always selects the CPU reference backend.
`trip_fdcb_optimize_accelerator_v1` selects the shared GPU backend when the
library is built with `-D FDCB_ABI_ACCELERATOR=true` and an accelerator target;
otherwise it returns `-3`. The packed view and output layout are identical for
both entry points.

Clinical dose follows the same explicit selection rule:
`trip_clinical_dose_compute_v1` remains the CPU reference entry point and
`trip_clinical_dose_compute_accelerator_v1` uses the shared accelerator backend
or returns `-3` when that backend was not compiled. Both consume the same
`ClinicalDoseProblemViewV1` and produce the same Float64 output records.

The C views are fail-closed about numeric mode. The host must identify FDCB,
`ms` or `msdb`, no biology or low-dose biology, and the effective CPU thread
count. A mismatched algorithm, biology model, precision mode, thread count, or
unknown flag is rejected before numeric work. Clinical dose uses its requested
runtime thread limit; the FDCB CPU library currently requires the effective
thread count selected at build time.

The legacy native `.exec` parser is a setup probe, not the production control
plane. It now accounts for every option in the P101 two-field file and rejects
unknown options and standalone flags. Its executable refuses plans requesting
optimization, clinical dose, RST output, or DVH instead of silently pretending
to execute them. Production `.exec` handling remains in TRiP C.

Reference precision matches the unmodified `trip_temp` build: sparse lateral
coefficients, per-slice static factors, particles, dose, gradients, biological
state, chi2, residuals, and reductions are all `Float64`. The precision mode is
versioned. `-D FDCB_MIXED32=true` explicitly selects an experimental
accelerator mode that narrows packed coefficients and dynamic device buffers to
`Float32` during transfer. Such a problem must be tagged
`FDCB_PRECISION_MIXED32`; CPU and reference accelerator entry points require
`FDCB_PRECISION_REFERENCE`, so no backend can silently downgrade reference
arithmetic.

`fdcb_packing.mojo` is the explicit adapter from the convenient physical and
biological scenario models. A small native field-slice descriptor partitions
global spot indices into grouped field/beam ranges; packing converts them to
UInt16 slice-local indices and emits the voxel/scenario/field-slice table.
Biological entries in one native slice must share their packed Float64 static
coefficients, matching TRiP's energy-slice semantics. The eventual TRiP C
adapter will populate the same contract directly without introducing another
numeric model.

Minimum-particle behavior is a host policy. `complexminp` accepts either an
explicit seed or the exact 31-word optimizer-entry RNG snapshot produced by
TRiP setup. RNG state and stochastic decisions do not belong to numeric
backends.

## Current parity gaps

The packed CPU path and thin C ABI cover physical and biological sparse forward
dose, robust minimum/maximum selection, chi2, LET objectives,
gradient/backprojection, exact step sizing, full iteration/backtracking and
convergence control, and seeded host-side `complexminp`. V1 deliberately matches
the validated `TRIPFDCBCudaDirectInput` case: robust biological FDCB without
robust-RBE, all-plan, static-field, or master-field modes, which `trip-gpu`
rejects before packing. OER tables and scenario-specific RBE models would need a
future versioned numeric contract; they are not silently approximated in V1.

Live CPU production parity is established against the sole reference,
unmodified `trip_temp`, for the 12-thread P101 3D robust biological plan. The
packed case has 2 fields, 48 field-energy slices, 40,889 points, 36,879 voxels,
9 scenarios, 12,928,841 slices, and 868,731,767 Float64 coefficients. Both
implementations accept regular iteration 271 and reject the next candidate
because chi2 no longer decreases. Mojo reports 271 iterations like TRiP; the
separate initialization update is deliberately excluded. Its accepted-plan
chi2 is 36.90306299652. TRiP leaves `psOpt->dChi2` holding the rejected
candidate even after restoring iteration-271 particles, so that stale scalar is
not the final plan chi2.

The paired in-memory comparison after TRiP restored the accepted plan had
particle RMS error 1.30e-8 and maximum error 1.21e-7. Independent writes from
Mojo-only and native-TRiP runs produced byte-identical RST files for both fields;
parsed particle RMS and maximum error were exactly zero. No iteration limit,
stopping criterion, tolerance, or output formatting was changed. Mojo's
Float64 numeric optimizer took 142.81--150.62 s in two runs. The Mojo-only
`OptCmd` took 172.70 s end to end; the clean native `trip_temp` measurement was
192.91 s.

A full-run CPU profile attributes 67.7% of sampled cycles to biological state
evaluation, 11.3% to exact-step response, and 6.5% to backprojection. Copying
the borrowed ABI problem into Mojo accounts for only 0.8%. A block-cyclic state
schedule preserved byte-identical RSTs but improved the best runtime by only
0.6%, so it was not retained.

For a faithful dose-only comparison, both implementations read the same two
optimized RST files and use the same 12-thread P101 setup. Unmodified
`trip_temp` took 109.20 s for `DoseCmd`. The current all-Float64 Mojo reference
path took 61.11 s in the numeric kernel and 63.29 s for the command, about 42%
faster end to end. Hardware-counter profiling showed low cache and branch-miss
rates; about 30% of the original Mojo cycles were in its native Float64
exponential. Calling the platform `libm exp`, which is also what the TRiP CPU
reference calls, removed that bottleneck without an approximation or
precision-mode change. Profiling then showed that point traversal dominated,
at 1.95 instructions per cycle with 0.78% branch misses and 0.30% last-level
cache misses. An in-memory raster-row index rejects whole out-of-support rows
while preserving the original contributing point order; finer row scheduling
also reduces tail imbalance. CPU shared-library builds pass `-Xlinker -lm` so
the platform-math dependency is explicit.

The clinical-dose boundary keeps CT, VOI selection, field/RST preparation, RBE
tables, normalization, cube storage, and output in TRiP. Mojo receives the flat
prepared grid, CT/HLUT data, transforms, raster points, DDD tables, and biology
tables. It computes Siddon WET, DDD and biology interpolation, divergent
double-Gaussian transport, and Float64 physical/biological accumulation over
dose-grid row work units. The reference ABI keeps geometry, static tables,
particle numbers, and accumulators in Float64; any later mixed32 dose mode must
be explicit and may not alter this reference path.

Using the byte-identical optimized RST files, the complete 512 x 512 x 149
physical cube differs from fresh `trip_temp` output by 5.88e-11 relative L2
with 1.49e-8 Gy maximum absolute error. The biological cube differs by 3.53e-11
relative L2 with 2.98e-8 Gy maximum absolute error. Both have exactly the same
1,654,902 nonzero voxels. These maxima are one and two Float32 output ULPs; no
output scaling, tolerance change, or shaping is applied. The current CPU entry
point deliberately supports the validated static one-CT-state `ms`/`msdb`
physical and low-dose biological path. 4D deformation/state output, OER, arc
fields, lung modulation, and specialized LET outputs remain explicit parity
gaps.

`tools/compare_float_cubes.c` compares raw Float32 dose cubes without a
tolerance or output shaping and reports nonzero support, relative L2 error, and
maximum absolute error. Comparing complete runs across `trip_temp` and
Mojo now uses byte-identical optimized RSTs, so these same-particle cube metrics
also describe the validated end-to-end optimizer-to-dose result. A tested broad
per-energy spatial bin was not retained: P101's
conservative `f2_max` made the bins coarse and reordered point traversal lost
DDD interpolation locality. The retained raster-row index is narrower: it only
skips a row when every point in that row fails the existing y-support bound, so
the final distance gate and accumulation order are unchanged. Lazy WET
evaluation was also byte-exact but slower and was reverted. Further dose work
should profile the indexed kernel before adding tighter depth-dependent support.

A bounded direct comparison against `trip-gpu` uses the same packed two-scenario
robust biological case in both implementations. Sparse dose, robust scenario
selection, weighted prescription, and chi2 match exactly. After the
initialization update and four regular updates, final chi2 differs by 2e-15 and
the single particle number by 1.78e-15. The raw gradient differs by 3.21e-7
because validated `trip-gpu` stores its per-slice gradient scale as Float32,
whereas Mojo reference mode follows the Float64 gradient precision policy.
With `complexminp` enabled, deletion decisions, deletion count, and final
particles match. `trip-gpu`'s result counter omits failed first draws; Mojo
reports every RNG draw actually consumed.

A production-data NVIDIA comparison first used an in-memory 23,000-voxel slice of
P101 with 9 robust scenarios, 40,889 particle variables, 9,936,000 slices, and
563,162,639 coefficients. The current `trip-gpu` direct entry point must first
run dose/gradient once: otherwise it invokes exact-step reduction with
uninitialized selected-scenario and min/max-dose buffers. With that state
initialized, a current `complexminp` rerun after the full-width metadata
regression fix deletes the same 20 spots; update-9 chi2 differs by
1.93e-8, particle RMS by 1.87e-5, and maximum particle number by 2.11e-4.
Longer trajectories separate because `trip-gpu` quantizes every
slice-gradient scale to Float32 while Mojo retains the required Float64
reference gradient. A subsequent complete P101 run on H200 accepted iteration
271, stopped on the same chi2 increase, and produced byte-identical RST files
for both fields. The Mojo accelerator ABI took 10.05 s versus 70.97 s for
unmodified `trip_temp` on the same server. The live device-buffer footprint was
8.51 GiB: 1.21 GiB of indices, 6.47 GiB of coefficients, and the remaining
workspaces. Mojo's default memory manager initially reserved about 129 GiB of
the H200, although that memory was not live problem data. The supplied H200
runner caps the allocator arena at 10% and uses 10% growth chunks. A repeated
full optimization then peaked at 14,875 MiB, retained the 10.03 s optimizer
time, stopped at iteration 271, and again produced byte-identical RST files.

Shared accelerator kernels keep the packed matrix resident,
perform sparse slice dots with Float64 warp/wave reductions, accumulate the five
scenario moments, convert biological dose when requested, select robust min/max
scenarios, evaluate physical or LET/dose objectives including robust Dmax, and
perform Float64 physical or biological gradient/backprojection with
active/zero-particle suppression. The full iteration controller supports both
packed modes through the same resident evaluator. The kernels run on NVIDIA and
compile unchanged for AMD. Cross-device
Float64 checks use a fixed IEEE-754 epsilon bound because accelerator
contraction can differ by one ULP. The explicit mixed32 build runs the focused
NVIDIA physical and biological forward/objective/gradient smoke tests and
compiles for AMD gfx1100 (32-lane RDNA3) and gfx942 (64-lane CDNA3). All eight
shared kernels also lower to Apple M3 Metal LLVM without device-side Float64;
final metallib generation and runtime validation require an Apple toolchain and
hardware and remain pending. The existing full host iteration controller can
use the resident accelerator evaluator and sparse exact-step response without
duplicating convergence, backtracking, Fletcher-Reeves, or minimum-particle
logic. Gradient scatter first compacts the selected nonzero slices, avoiding a
warp launch for every robust scenario slice. Chi2, weighted prescription, and
gradient-norm reductions are performed in device blocks with only compact
partials folded on the host. Optimizer evaluations suppress optional diagnostic
copies; gradients, directions, and particle candidates remain host-visible
because reference `complexminp` makes deterministic host RNG decisions over
them. The two multi-gigabyte coefficient arrays are staged sequentially during
accelerator construction; for the complete P101 case this lowers peak setup
host memory by about 1.62 GiB. Forward slice dots and slice-gradient scales
share one device workspace because their lifetimes do not overlap. Device-only
slice metadata packs the
full UInt32 field-slice index and UInt32 count together while retaining the full
UInt64 coefficient offset. It therefore preserves non-monotonic V1 ranges and
does not narrow or alter the packed V1/C ABI contract. A future policy-specific
fast path may move updates for cases without
`complexminp`; VOI processing, `.exec` expansion, and replay/export files remain
outside this backend.

The clinical-dose accelerator maps independent prepared dose-grid voxels to
device threads and stages the packed V1 records byte-for-byte. Grid and beam
geometry, HLUT, DDD and biology tables, Siddon WET, divergent double-Gaussian
transport, and all six output accumulators remain Float64. On the complete P101
direct-dose workload, the NVIDIA output has exact nonzero support versus a fresh
`trip_temp` run; physical and biological relative L2 differences are
`5.88e-11` and `3.53e-11`, with maxima of one and two Float32 output ULPs. On
H200 the kernel took 11.49--11.54 s and 11.75 s including packing and output
copy; the full TRiP dose command took 15.52--19.31 s, versus 236.39 s for the
unmodified CPU reference on the server. The focused physical and biological C
ABI cases pass on NVIDIA sm_75, and the complete P101 case passes on sm_90a.
The shared code compiles for AMD gfx942, but the pinned Mojo compiler rejects
MI100's gfx908 target, so MI100 runtime support remains a toolchain limitation
rather than a claimed backend.
