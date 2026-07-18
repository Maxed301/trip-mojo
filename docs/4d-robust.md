# 4D robust optimizer

TRiP owns CT states, deformation, masks, VOIs, WET, field setup and RST output.
It appends every selected state's optimization voxels to one flat table before
optimizer starts. Mojo therefore needs no motion-state branch: each flat voxel owns
the same robust-scenario range used by the 3D layout.

```text
voxel 0 -> scenarios 0..N-1 -> sparse slices -> coefficients
voxel 1 -> scenarios 0..N-1 -> sparse slices -> coefficients
...
```

Mojo owns the packed metadata arrays. The matrix builder always creates the
slice-local UInt16 topology on the accelerator. A shard with at most ten
billion entries also stores its Float64 coefficients. A larger shard retains
the geometry and exact double-Gaussian parameters instead, then reconstructs
each coefficient in the optimizer with the same Float64 expression used during
matrix construction. This selection is automatic and has no environment or
command-line flag. Distinct robust scenario matrices are retained. TRiP
releases completed source matrices while packing, and Mojo releases
matrix-build host buffers as soon as TRiP imports their values. Once
accelerator setup is complete, Mojo also releases packed voxel, scenario, state
and slice arrays that the iteration loop no longer uses.

## Canonical case

The reference exec is:

```text
${ROOT}/TRIP_DATA/P101_4Dopt/exec/P101_ITV_full4DITVplan.lustre.exec
```

It uses ten CT states, nine robust scenarios, biological low-dose optimization with MSDB and
deterministic `complexminp`. The packed problem contains 251,790 voxels and
5,546,317,539 coefficients. Reference and Mojo runs use the same exec, RNG
state, Float64 arithmetic and stopping criteria.

The clean `trip_temp` reference accepted iteration 120 at relative chi-square
change `4.54912e-6`, below the unchanged `1e-5` limit. Final residual was
1.53519%. It measured:

- 392.87 s through optimization
- 412.26 s for `OptCmd`
- 476.751 s process wall time
- 81,877,096 KiB peak RSS

H200 job 22017 accepted the same iteration and produced two byte-identical RST
files with zero parsed particle error. It measured:

- 3.323 s physical matrix build
- 0.603 s optimizer metadata packing
- 6.764 s complete Mojo optimization call
- 17.41 s for `OptCmd`
- 27.841 s process wall time
- 22,475,804 KiB peak RSS

The numerical optimizer contract is 12 threads. This H200 runner reserves 32
threads for TRiP setup and uses 16 for metadata packing. A 12-thread setup probe
took 30.461 s; shared-node variation remains visible.

## Multi-device memory scaling

`submit.sh --vendor nvidia --gpus N --plan ...` partitions contiguous whole
voxels near equal sparse-coefficient work. Particles and small field metadata
are replicated; slices, scenario state and Float64 coefficients are local to
one device. Forward passes and backprojections are queued on every device
before their partial gradients and scalar reductions are merged on the host.

Direct multi-device construction gives every device its own matrix shard and
does not create a complete host coefficient matrix. The canonical nine-scenario
case remains byte-identical on one, two and three H200s. Current Slurm walltimes
are 30 s, 28 s and 28 s respectively; optimizer time is 7.161 s on one H200,
4.570 s on two and 3.690 s on three.

The procedural representation changes the dominant entry payload from one
UInt16 index plus one Float64 coefficient (10 bytes) to the UInt16 index alone
(2 bytes), an exact 80% reduction. It additionally retains compact geometry and
88 bytes of parameters per matrix slice. The 21-scenario P101 plan contains
15,296,352,715 matrix entries, so its entry payload falls from 152.96 GB to
30.59 GB, saving 122.37 GB before the retained slice metadata. H200 job 23228
therefore ran on one device where the materialized matrix did not fit. It
stopped at iteration 208, wrote RSTs byte-identical to the materialized
two-device job 23210, and measured 58.784 s for optimization and 1:35 walltime.

The 81-scenario plan contains 56,514,021,756 packed coefficient references.
Job 23230 split 58,812,585,971 matrix entries into three procedural shards,
completed 148 iterations in 66.308 s, and measured 3:04 walltime with
196,652,932 KiB peak host RSS. This is a capacity and scaling result; there is
no materialized 81-scenario result against which to claim parity. The current
exact recomputation is about twice the extrapolated materialized cost per
device. A row-wise separable Gaussian cache is a possible next optimization,
but it must preserve the optimizer trajectory before replacing the exact path.

## Integration

The cluster keeps one `trip_temp` checkout at commit `1fb423f`, patched once
with the integration patches under `integration/trip_temp`. `build.sh`
updates the persistent Mojo library and linked TRiP executable. `submit.sh`
runs an existing plan without copying sources or rebuilding.

## 4D clinical dose

The dose ABI flattens state field ranges, CT descriptors and transformed voxel
positions into contiguous state-major arrays. TRiP owns motion transforms and
maps reference-grid positions into every state. Mojo evaluates each state with
its CT and RST, reduces the six raw Float64 dose terms in state order, and
returns one reference-grid result for normal TRiP storage.

The canonical optimizer exec stops after writing RSTs and the case data has no
deformation field. A 4D dose validation plan must explicitly expand the static
delivery into state RSTs with TRiP's `perfectrescan` and load a transform. This
is useful for exact `trip_temp` versus Mojo implementation comparison under the
same motion model, but is not evidence of deformable clinical-dose parity.
Smoothed 4D, per-state output, arc, oxygen and lung modulation are rejected
explicitly.

H200 job 22069 confirmed that the accelerator adapter received ten states,
41,446 reference-grid voxels and 20 flattened state fields. The physical and
biological Float32 cubes each contained 39,059,456 voxels with identical
11,764-voxel support; both were byte-identical to `trip_temp` (`relative_l2=0`,
`max_abs=0`). `DoseCmd` took 8.17 s in `trip_temp` and 10.62 s through Mojo. The
current small Tumor-only workload does not amortize host packing and GPU
transfer, so this is a correctness result rather than a speedup claim.
