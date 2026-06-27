# Agent Instructions

This repo is a ground-up Mojo rewrite of the TRiP dose optimization MVP.

## Project goals

- Build a clean Mojo-native dose optimization system, not a TRiP98 port.
- Use `~/Projects/trip4d` as the canonical unmodified CPU/reference implementation.
- Use `~/Projects/trip-gpu` as the CUDA MVP and performance reference only.
- Do not preserve legacy TRiP98 architecture, command dispatch, global state, C adapters, or CUDA integration debt.
- Keep `.exec` support as a narrow import adapter for known benchmark cases, not as the native API or a full scripting engine.
- Prefer a clean native case/problem model for geometry, fields, spots, objectives, optimizer settings, and results.

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
- Keep parsing, file I/O, setup, validation, and reporting on CPU.
- Put GPU effort into dose accumulation, objective/residual evaluation, gradient/backprojection, spot updates, and reductions.
- Maintain a CPU backend as the correctness/debug reference.

## Validation

- Run tests and parity/debug commands with 12 threads unless a narrower single-thread probe is explicitly required.
- Compare correctness against `trip4d` before expanding scope.
- Compare GPU performance and kernel behavior against `trip-gpu` where useful.
- Track correctness with objective/chi2, residuals, particle numbers, dose summaries, and relevant dose-volume outputs.
- Never use fudge factors, hidden normalization constants, output shaping, or tolerance loosening to make results look better. These hide bugs and usually indicate missing or incorrect physics, geometry, beam modeling, data parsing, or optimization logic. Any empirical/debug-only scale must be explicitly labeled, isolated, and removed before claiming parity.
- Do not trade correctness for speed.

## Scope control

- Start from the smallest useful vertical slice: clean case model → CPU optimizer → GPU optimizer → optional `.exec` importer.
- Do not reintroduce legacy compatibility unless explicitly requested.
- Unsupported `.exec` features should fail clearly rather than silently approximating old behavior.
