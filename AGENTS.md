# Agent Instructions

This repo is a portable Mojo implementation of the TRiP FDCB optimizer.

## Project goals

- Build a backend-neutral Mojo FDCB optimizer, not a full TRiP rewrite.
- Use `~/Projects/trip_temp` as the sole unmodified CPU correctness and parity
  reference. Match its `TRPFLDDOSE_DBLPREC` Float64 dose-matrix coefficients,
  optimizer trajectory, native stopping iteration, particle numbers, and RST
  output before making parity claims.
- `~/Projects/trip_temp` is the only TRiP source reference for correctness,
  semantics, integration, and performance comparisons. Do not use or search for
  `trip-gpu`, `trip4d`, or another TRiP checkout, even if an older prompt,
  document, patch name, or repository history mentions one.
- Keep TRiP's C control plane, geometry, VOI, WET, dose-matrix setup, parsing, and output outside this repository.
- Accept one flat packed FDCB problem through a thin C ABI and keep the numeric implementation backend-neutral.
- Keep clinical direct-dose setup and storage in TRiP, with a separate packed
  boundary for Mojo WET, transport, and biological accumulation.
- Do not port unrelated TRiP functionality or duplicate its `.inc` integration.

## Canonical directories

- Sole TRiP source and CPU truth: the clean local `~/Projects/trip_temp`
  checkout at commit
  `1fb423f`. Hydra's `/lustre/bio/mdick/CUDA/trip_temp` uses that same commit
  plus only `integration/trip_temp/trip_temp_mojo.patch`, applied once for the
  persistent headless Mojo-linked build. Never inspect or use `trip4d` or
  `trip-gpu`; if `trip_temp` is missing or unsuitable, stop and report that
  problem instead of substituting another tree.
- Clinical/case inputs on Hydra: `/lustre/bio/mdick/CUDA/TRIP_DATA`. The P101
  4D robust case lives under `TRIP_DATA/P101_4Dopt`; do not read case inputs
  from a TRiP source checkout.
- Portable implementation: `~/Projects/trip-mojo` locally and
  `/lustre/bio/mdick/CUDA/trip-mojo` on Hydra.

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
- The pinned Mojo package does not expose gfx908. MI100 work uses the custom
  Modular source build with the added CDNA1/gfx908 target; distinguish compiler
  support from ROCm container/runtime availability in status reports.
- Keep parsing, file I/O, setup, VOI ownership, and reporting in TRiP's existing CPU control plane.
- Put GPU effort into dose accumulation, objective/residual evaluation, gradient/backprojection, spot updates, and reductions.
- Maintain a CPU backend as the correctness/debug reference.

## Validation

- Keep the numerical optimizer and CPU parity contract at 12 threads unless a
  narrower single-thread probe is required. The validated H200 runner reserves
  32 CPU threads for TRiP setup and uses 16 for metadata packing; always report
  those separately from optimizer threads.
- Compare CPU correctness only against `trip_temp`. Do not use `trip4d` results
  as evidence of optimizer parity.
- Compare every implementation and benchmark only against `trip_temp` and its
  recorded canonical outputs. Never cite `trip-gpu` or `trip4d` as evidence.
- Track correctness with objective/chi2, residuals, particle numbers, dose summaries, and relevant dose-volume outputs.
- Never use fudge factors, hidden normalization constants, output shaping, or tolerance loosening to make results look better. These hide bugs and usually indicate missing or incorrect physics, geometry, beam modeling, data parsing, or optimization logic. Any empirical/debug-only scale must be explicitly labeled, isolated, and removed before claiming parity.
- Do not trade correctness for speed.

## Scope control

- Start from the smallest useful vertical slice: packed FDCB model → CPU parity backend → shared NVIDIA/AMD kernels → thin C ABI.
- Do not reintroduce legacy compatibility unless explicitly requested.
- Treat Apple Metal as a later experimental target.
