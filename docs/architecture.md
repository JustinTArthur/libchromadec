# Architecture

libchromadec decodes 4×fsc-sampled composite analog video (CVBS / TBC) into
component Y'CbCr. This page describes how the library is laid out. For the
decode workflow see the integration guide, and for exact symbols see the C API
reference.

## Public surface

The public interface is a pure C ABI exposed under `<chromadec/...>`, with
opaque handles, the `chd_status_t` return-code enum, and a thread-local
last-error string (`chd_last_error`). Every public symbol is prefixed `chd_`.
`chromadec.h` is the umbrella header; callers may also include the individual
headers (`errors.h`, `log.h`, `types.h`, `video.h`, `decoder.h`, `frame.h`,
`nn.h`, `dropout.h`, `pipeline.h`).

Failures and diagnostics travel separate channels, and the library owns no
console of its own. A failure comes back on the return path, as a
`chd_status_t` plus the detail string the layer that detected it recorded.
Everything else, the running commentary a decode produces, goes to a
process-wide sink the consumer installs through `log.h`; with no sink
installed, nothing is emitted anywhere. Internally `chd::log::fail()` both
emits at error level and records the detail, so a reason found deep in a
reader or a metadata parser survives the trip up through the `bool` returns
between it and the ABI shim. Those messages carry `CHD_LOG_F_RETURNED` on the
sink, which is how a consumer tells the copy of a returned failure from an
error that reaches it no other way. A site whose condition an intermediate
layer recovers from is an `error()` rather than a `fail()`, so the flag stays a
promise about the return path rather than a hint.

## Internal implementation

The implementation is C++17 under `src/`. The C ABI lives only in `src/abi/`, a
set of thin `c_api_*.cpp` shims that translate opaque handles, validate
arguments, and catch C++ exceptions so they never cross the ABI boundary. There
is no Qt anywhere.

Module layout under `src/`:

- `reader/`: source readers behind a single `ISource` interface: ld-decode
  `.tbc` (with `.tbc.db` / `.tbc.json` metadata sidecars), CVBS `.cvbs`, and
  dual-file YC (`.cvbsy` + `.cvbsc`, or luma + chroma `.tbc`).
- `metadata/`, `format/`: sidecar parsing and source-format detection.
- `decoders/`: the decoder implementations, the decoder registry, the
  cross-system chroma filter, and shared FIR/IIR filter helpers.
- `nn/`: the ONNX Runtime engine, the native CoreML engine, and execution
  provider selection.
- `dropout/`: dropout detection and concealment.
- `output/`: pixel-format conversion and output framing.
- `pipeline/`: ties a source, decoder, and output format into a decode run.
- `common/`: shared utilities.
- `abi/`: the C ABI shims described above.
- `legacy/`: ld-decode history preserved via `git filter-repo`, kept for
  reference and incremental porting rather than built into the library.

## Decoders

- Mono (luma only).
- NTSC comb: 1D / 2D / 3D and adaptive.
- PAL: PALcolour and Transform-PAL 2D / 3D.
- SECAM: line-sequential FM chroma via block-FFT analytic signal, exact
  inverses of both Table 4 pre-corrections, a near-DC differentiating-FIR
  discriminator, and porch-calibrated carriers that recentre the band and
  bell masks on the measured pair; 4:4:0 output only.
- nnTransform3D: neural 3D decode, fronted by an FFT stage (see NN inference).
- ldzeug2: color_cnn and luma_sep (per-field and per-frame), pure convolutional
  neural networks (CNNs).

## NN inference

The neural decoders (ldzeug2 and nnTransform3D) hold a backend-agnostic
`InferenceEngine` rather than reaching for runtime types directly, so the same
decoder code runs on any backend. Both implementations serve every neural
decoder:

- ONNX Runtime (`OrtEngine`): one process-wide `Ort::Env` (an RAII singleton)
  and one `Ort::Session` per `chd_nn_model_t`, shared across worker threads
  since sessions are thread-safe upstream. Execution-provider selection walks a
  per-OS chain (TensorRT/CUDA/DirectML on Windows, CUDA/MIGraphX on Linux,
  CoreML on macOS) with CPU as the always-available leaf.
- Native CoreML (`CoreMLEngine`, macOS only): runs an `.mlpackage` directly,
  no ONNX Runtime.

nnTransform3D additionally has a device-resident FFT front-end for its
transform stage: FFTW on CPU, cuFFT on CUDA, hipFFT on ROCm, MPSGraph FFT under
CoreML. On the GPU paths it binds device pointers into the ORT session through
`Ort::IoBinding`, so those FFT pipelines stay nnTransform3D-specific. The
ldzeug2 CNNs have no transform stage and run entirely through the engine's
`run()` call, so they pick up whichever backend and provider is active with no
decoder-specific code.

## Build system

Meson is the primary build. The library ships a pkg-config `.pc` file and a
CMake package config, so CMake consumers can
`find_package(chromadec CONFIG REQUIRED)`. Optional backends are gated by build
options: `with_onnxruntime`, `with_cuda`, `with_rocm`, and `with_coreml`.
