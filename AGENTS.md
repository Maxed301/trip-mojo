# Agent Instructions

This repo is a portable Mojo implementation of the TRiP FDCB optimizer.

## Project goals

- Build a backend-neutral Mojo FDCB optimizer, not a full TRiP rewrite.
- Use `~/Projects/trip_temp` as the sole unmodified CPU correctness and parity
  reference. Match its `TRPFLDDOSE_DBLPREC` Float64 dose-matrix coefficients,
  optimizer trajectory, native stopping iteration, particle numbers, and RST
  output before making parity claims.
- Use `~/Projects/trip-gpu` as the validated CUDA and performance reference.
- Keep TRiP's C control plane, geometry, VOI, WET, dose-matrix setup, parsing, and output outside this repository.
- Accept one flat packed FDCB problem through a thin C ABI and keep the numeric implementation backend-neutral.
- Keep clinical direct-dose setup and storage in TRiP, with a separate packed
  boundary for Mojo WET, transport, and biological accumulation.
- Do not port unrelated TRiP functionality or duplicate its `.inc` integration.

## Engineering priorities

- Performance first where it matters: hot numeric kernels, memory layout, CPU/GPU transfer boundaries, sparse/dense dose operations, and optimizer loops.
- Readability always matters: code should be straightforward, explicit, and easy to audit.
- Prefer simple data-oriented design over deep abstraction.
- Keep host/control code clean and boring; optimize only measured hot paths.
- Avoid speculative generality. Add flexibility only when a real case needs it.
- Make CPU and GPU backends share the same problem model and math contract.

## Mojo/GPU direction

- Use current Mojo syntax and idioms.
- Prefer Mojo-native code over inline Python. Do not add inline Python or Python interop for parsing, file I/O, setup, validation, reporting, math, or tests; implement those pieces natively in Mojo.
- GPU code must be Mojo-native, not CUDA-shaped code translated mechanically.
- Keep parsing, file I/O, setup, VOI ownership, and reporting in TRiP's existing CPU control plane.
- Put GPU effort into dose accumulation, objective/residual evaluation, gradient/backprojection, spot updates, and reductions.
- Maintain a CPU backend as the correctness/debug reference.

## Validation

- Run tests and parity/debug commands with 12 threads unless a narrower single-thread probe is explicitly required.
- Compare CPU correctness only against `trip_temp`. Do not use `trip4d` results
  as evidence of optimizer parity.
- Compare GPU performance and kernel behavior against `trip-gpu` where useful.
- Track correctness with objective/chi2, residuals, particle numbers, dose summaries, and relevant dose-volume outputs.
- Never use fudge factors, hidden normalization constants, output shaping, or tolerance loosening to make results look better. These hide bugs and usually indicate missing or incorrect physics, geometry, beam modeling, data parsing, or optimization logic. Any empirical/debug-only scale must be explicitly labeled, isolated, and removed before claiming parity.
- Do not trade correctness for speed.

## Scope control

- Start from the smallest useful vertical slice: packed FDCB model → CPU parity backend → shared NVIDIA/AMD kernels → thin C ABI.
- Do not reintroduce legacy compatibility unless explicitly requested.
- Treat Apple Metal as a later experimental target.
