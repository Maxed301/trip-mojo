# 4D robust FDCB

TRiP owns CT states, deformation, masks, VOIs, WET, field setup and RST output.
It appends every selected state's optimization voxels to one flat table before
FDCB starts. Mojo therefore needs no motion-state branch: each flat voxel owns
the same robust-scenario range used by the 3D layout.

```text
voxel 0 -> scenarios 0..N-1 -> sparse slices -> coefficients
voxel 1 -> scenarios 0..N-1 -> sparse slices -> coefficients
...
```

Mojo owns the packed metadata arrays. The matrix builder creates Float64
coefficients and slice-local UInt16 point indices on the accelerator; the
optimizer consumes them without a host coefficient copy. Distinct robust
scenario matrices are retained. TRiP releases completed source matrices while
packing so the nested and packed forms are not both fully resident.

## Canonical case

The reference exec is:

```text
/lustre/bio/mdick/CUDA/TRIP_DATA/P101_4Dopt/exec/P101_ITV_full4DITVplan.lustre.exec
```

It uses ten CT states, nine robust scenarios, biological low-dose FDCB/MSDB and
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

## Two-device memory scaling

`run_h200_4d_2gpu.sh` partitions contiguous whole voxels near half the sparse
coefficient count. Particles and small field metadata are replicated; slices,
scenario state and Float64 coefficients are local to one device. Both forward
passes and both backprojections are queued before their partial gradients and
scalar reductions are merged on the host.

H200 job 22598 used the canonical CPU-built matrix and measured 32,605 MiB of
reserved VRAM per GPU, 13.604 s optimization, 38.80 s `OptCmd` and 49.311 s
process wall time. It stopped at iteration 120 with the canonical objective and
wrote both RSTs byte-identically. This path trades speed and host memory for
lower per-device VRAM; the direct one-H200 path above remains the performance
reference. Peak host RSS was 139,720,980 KiB because TRiP's source matrices and
the packed host coefficient arrays overlap; removing that overlap is separate
from device sharding.

## Integration

Hydra keeps one `trip_temp` checkout at commit `1fb423f`, patched once with the
optimizer and clinical-dose patches under `integration/trip_temp`.
`build_h200.sh` updates the two persistent binaries. `run_h200_4d.sh` only
prepares the exec, runs the case and compares both RSTs; it performs no source
copying or build.

## 4D clinical dose

The dose ABI flattens state field ranges, CT descriptors and transformed voxel
positions into contiguous state-major arrays. TRiP owns motion transforms and
maps reference-grid positions into every state. Mojo evaluates each state with
its CT and RST, reduces the six raw Float64 dose terms in state order, and
returns one reference-grid result for normal TRiP storage.

The canonical optimizer exec stops after writing RSTs and the case data has no
deformation field. `run_h200_4d_dose.sh` therefore creates a separate transient
dose exec, expands the optimized static delivery into equal-weight state RSTs
with TRiP's `perfectrescan`, and generates a Tumor center-of-mass translation.
This is useful for exact `trip_temp` versus Mojo implementation comparison
under the same motion model, but is not evidence of deformable clinical-dose
parity. Smoothed 4D, per-state output, arc, oxygen and lung modulation are
rejected explicitly.

H200 job 22069 confirmed that the accelerator adapter received ten states,
41,446 reference-grid voxels and 20 flattened state fields. The physical and
biological Float32 cubes each contained 39,059,456 voxels with identical
11,764-voxel support; both were byte-identical to `trip_temp` (`relative_l2=0`,
`max_abs=0`). `DoseCmd` took 8.17 s in `trip_temp` and 10.62 s through Mojo. The
current small Tumor-only workload does not amortize host packing and GPU
transfer, so this is a correctness result rather than a speedup claim.
