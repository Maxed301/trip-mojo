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

## Integration

Hydra keeps one `trip_temp` checkout at commit `1fb423f`, patched once with
`integration/trip_temp/trip_temp_mojo.patch`. `build_h200.sh` updates the two
persistent binaries. `run_h200_4d.sh` only prepares the exec, runs the case and
compares both RSTs; it performs no source copying or build.

The direct boundary currently covers robust optimization and RST ownership.
4D clinical-dose deformation and accumulation remain outside Mojo.
