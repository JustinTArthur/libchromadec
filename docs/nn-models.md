# Neural-network models

The neural decoder kinds load an external **ONNX** model at runtime. libchromadec
ships no weights of its own; you supply a model file and attach it to the
decoder. NN support exists only when the library was built with it
(`chd_has_feature("nn")` returns 1).

Two model families are supported, both originating as NTSC composite-video
research projects:

- **nnTransform3D** (the `chroma_net` model), by asdfqazsnbb.
- **ldzeug2** (color-CNN and luma-separation models), by jsaowji.

## Applicability: NTSC only

!!! warning "These models are for 525-line NTSC, not PAL"
    Every model below was trained and designed for **525-line NTSC**
    (NTSC-1953 / SMPTE ST 170 and close relatives). They are **not** applicable to
    PAL, and that includes 525-line **PAL-M**. They exist precisely because NTSC
    never had a frequency-domain ("transform") decoder the way PAL has Transform
    PAL; the neural decoders fill that gap for NTSC.

Because of this, [`CHD_DEC_AUTO`](api-reference.md#chd_decoder_create) never
resolves to a neural decoder (it picks `CHD_DEC_NTSC_2D` / `CHD_DEC_PAL_2D` by
standard). You must select a neural kind explicitly, and only for NTSC content.

## Loading and attaching a model

Load with
[`chd_nn_model_load_from_file`](api-reference.md#chd_nn_model_load_from_file) (or
[`chd_nn_model_load_from_memory`](api-reference.md#chd_nn_model_load_from_file) for an
embedded model), then attach
to a neural decoder with
[`chd_decoder_set_nn_model`](api-reference.md#chd_decoder_set_nn_model) before
`commit`. The model is borrowed, so keep it alive for the decoder's lifetime.
[`chd_shutdown`](api-reference.md#chd_shutdown) offers optional deterministic
teardown. See the [integration guide](integration-guide.md#neural-decoders)
for the full sequence.

### Backends

The backend is chosen via `opts.backend` on
[`chd_nn_session_opts_t`](api-reference.md#session-options). The default
`CHD_NN_BACKEND_AUTO` infers it from the artifact (`.onnx` → ONNX Runtime,
`.mlpackage`/`.mlmodelc` → native CoreML). `CHD_NN_ORT_AUTO` forces ONNX Runtime
and tries a platform-specific EP chain, using the first that loads:

| Platform | Auto chain |
|---|---|
| Windows | TensorRT, CUDA, DirectML, CPU |
| macOS | CoreML, CPU |
| Linux / other | TensorRT, CUDA, MIGraphX, CPU |

A chain only reaches the providers the linked ONNX Runtime was built with. On
Windows that is a packaging choice rather than a hardware one: the release
archives carry CPU alone (`onnxruntime-win-<arch>`) or CUDA plus TensorRT
(`onnxruntime-win-x64-gpu`), while DirectML is published only as the
`Microsoft.ML.OnnxRuntime.DirectML` NuGet package, and none of them carries
both CUDA and DirectML. `-Donnxruntime_root=` takes either layout, the
archive's `lib/` plus `include/` or the NuGet package directory as restored.
See [shipping on Windows](integration-guide.md#shipping-on-windows) for which
DLLs then have to travel with your binary.

Pinning a specific `CHD_NN_ORT_*` provider tries only that one and returns
`CHD_E_NN_BACKEND_UNAVAILABLE` if it is not present. Probe ahead of time with
[`chd_nn_backend_is_available`](api-reference.md#chd_nn_backend_is_available),
and read the backend actually chosen with
[`chd_nn_model_get_active_backend`](api-reference.md#chd_nn_model_get_active_backend)
— which also distinguishes the native `CHD_NN_COREML` backend from the ORT
CoreML EP (`CHD_NN_ORT_COREML`).

!!! note "The CoreML *EP* covers only part of the graph"
    On macOS, the 3D-convolution operations in `chroma_net` are not supported
    by the ONNX Runtime CoreML execution provider; those nodes fall back to
    CPU regardless of the provider chain. The model still runs correctly, just
    without GPU/ANE acceleration. For full acceleration, use the **native
    CoreML** backend below instead of an ONNX Runtime backend.

For TensorRT and MIGraphX, set `engine_cache_dir` to avoid recompiling the
engine on every load (the first compile costs 15-30 s). See
[session options](api-reference.md#session-options).

### Native CoreML (macOS)

To run a model on GPU/ANE through CoreML directly — rather than through ONNX
Runtime's CoreML EP — load a `.mlpackage` with `opts.backend = CHD_NN_COREML`
(or just let `CHD_NN_BACKEND_AUTO` see the `.mlpackage` extension) via
[`chd_nn_model_load_from_file`](api-reference.md#chd_nn_model_load_from_file).
The handle binds to a decoder exactly like an ONNX-backed one and reports
`CHD_NN_COREML` as its active backend. This is the only route off CPU
on macOS for nnTransform3D (the EP gates out its 3D conv), and a large win for
ldzeug2 too: although the CoreML EP *can* run ldzeug2's 2D conv, it places only
part of the graph (≈ 74 of 120 nodes, across 7 partitions) on CoreML and bounces
the `Range`/`Gather`/`ScatterND`/`Slice` index machinery back to CPU at each
boundary, whereas coremltools compiles the whole graph for the GPU. Native
CoreML ran ldzeug2 `color_cnn` **~10× faster than the EP** (see Performance).

Generate the `.mlpackage` offline from the ONNX model with `coremltools`:

```bash
.venv/bin/pip install coremltools onnx2torch torch onnx
.venv/bin/python scripts/convert_coreml.py \
    --model-type nntransform3d --onnx chroma_net.onnx --out chroma_net.mlpackage
# ldzeug2 — one package per runtime shape (see fixed-shape note). color_cnn runs
# in field mode (NTSC 263×910); luma_sep has both a field and a frame export:
.venv/bin/python scripts/convert_coreml.py \
    --model-type ldzeug-colorcnn --onnx color_cnn.onnx --out color_cnn.mlpackage \
    --height 263 --width 910
.venv/bin/python scripts/convert_coreml.py \
    --model-type ldzeug-lumasep --onnx luma_sep.onnx --out luma_sep.mlpackage \
    --height 263 --width 910
```

The script converts at **fp32** by default (`--precision fp32`). At fp32,
nnTransform3D `chroma_net` and both ldzeug2 models (`color_cnn` and `luma_sep`)
match ONNX Runtime to rel-RMS ≈ 2e-6 at every tested shape, and the
GPU-resident Layer-2 nnTransform3D pipeline matches the FFTW Layer-1 pipeline
to ≈ 3e-6.

`--precision fp16` is a **per-model** choice, valid only for weights whose
input contract keeps every tensor inside fp16's ±65504 range:

- **`chroma_net` v2 (the ÷128-scale weights) converts safely at fp16**: on
  real-capture spectra its masks stay within rel-RMS ≈ 7e-4 of the fp32
  reference and the decoded chroma within ≈ 2 LSB of 65535 — below tape
  noise. An fp16 program is also what makes the model **ANE-eligible** (the
  ANE only executes fp16), which is the fastest macOS configuration; see
  [compute units](api-reference.md#coreml-compute-units) and Performance
  below.
- **v1-series `chroma_net` must stay fp32**: its contract feeds unscaled
  magnitudes, which overflow fp16 and produce NaN masks.
- **Both ldzeug2 models must stay fp32** regardless of scale: fp16 wrecks
  `color_cnn`'s `Range`/`Gather`/`ScatterND` index math (rel-RMS ≈ 0.9).

Either way the package presents fp32 inputs and outputs (the script pins the
I/O dtypes, keeping the casts inside the program), so an fp16 package is a
drop-in replacement at the engine boundary.

The conversion route is ONNX → `onnx2torch` → TorchScript → `coremltools`. The
script first applies two pre-passes `onnx2torch` needs: folding `Constant`-node
weights into initializers (`luma_sep` exports its conv weights as `Constant`
outputs, which `onnx2torch`'s Conv converter can't otherwise resolve) and
rewriting `SAME_UPPER` convolution padding to explicit pads.

The `.mlpackage` artifacts are not committed; regenerate them per machine / in
CI. Availability is reported by `chd_has_feature("coreml")` (false on non-Apple
builds and when configured with `-Dwith_coreml=disabled`). The native engine
defaults to CPU+GPU compute units; `CHD_NN_COREML_ALL` adds the ANE, which
engages only for fp16-converted packages (the ANE is an fp16-only device — an
fp32 program is ineligible for it wholesale, whatever its ops). The engine
falls back to CPU-only if a GPU predict fails.

Native CoreML is a self-contained backend and does **not** require ONNX Runtime.
A macOS build with `-Dwith_onnxruntime=false -Dwith_coreml=enabled` ships the
ldzeug and nnTransform3D decoders running entirely on CoreML, with no ORT
dependency linked. When both backends are built (the macOS default) they coexist:
load a `.onnx` for ORT or a `.mlpackage` for native CoreML, and
[`chd_nn_model_get_active_backend`](api-reference.md#chd_nn_model_get_active_backend)
distinguishes them at runtime.

!!! important "Convert at a fixed input shape, one package per shape"
    Every package only runs at the shape it was converted at: the TorchScript
    trace bakes the example dims into the model's dynamic-shape ops, so a
    flexible (`RangeDim`) shape either runs far slower or fails at runtime.

    - **nnTransform3D** fixes the input batch at 256 (`NNT3D_BATCH`, matching
      `kBatchBlocks` in `nntransform3d_pipeline_coreml.mm`); a dynamic batch ran
      **~30× slower** (CoreML processes it tile-by-tile). `runCoreMLPipeline`
      pads its last partial chunk up to this size — conv is per-batch-element
      independent, so the padding doesn't affect the real tiles. If you change
      `NNT3D_BATCH`, change `kBatchBlocks` to match and reconvert.
    - **ldzeug2** must be converted once per `[1,C,height,width]` you decode
      (`--height`/`--width` = the decoder's `modelHeight` × `fieldWidth`).
      `color_cnn` runs in field mode, so for NTSC that is `263×910`. A flexible
      `RangeDim` package fails at any non-default shape with *"Error in
      dynamically resizing for sequence length (error -7)"* — `color_cnn`'s
      `Range`/`ScatterND`/`Expand` ops don't re-derive under a resized input.

**Performance** (one M-series GPU). Two views: the full decoder pipeline and the
NN inference in isolation. Both are **single-threaded** — in production the
[`DecoderPool`](api-reference.md) runs frames across all cores (inter-frame
parallelism), which speeds the CPU paths up roughly by the core count and
narrows the gaps below. Native-backend frames match the ONNX Runtime CPU path to
**≤ 1 LSB out of 65535** (rel-RMS ≈ 1e-6) for every model.

Full pipeline, per frame (I/O → demod → NN → YUV):

| model | native CoreML | ORT CPU (1 thread) | ORT CoreML EP |
|---|---|---|---|
| nnTransform3D `chroma_net` | ≈ 0.78 s | ≈ 25 s | ≈ 27 s |
| ldzeug2 `color_cnn` (field) | ≈ 18 ms | ≈ 1.5 s | ≈ 157 ms |
| ldzeug2 `luma_sep` (field) | ≈ 42 ms | ≈ 5.9 s | ≈ 314 ms |

(A frame is two fields for the field-mode ldzeug2 decoders.) The ORT CoreML EP
doesn't help nnTransform3D — it runs the 3D conv on CPU anyway and adds
partition overhead, landing slightly *slower* than plain CPU.

For `chroma_net` v2, an **fp16 package with `CHD_NN_COREML_ALL`** moves the
convolution to the ANE and is the fastest macOS configuration measured — with
the default FFTW FFT, ≈ 0.52 s/frame vs ≈ 0.77 s for the fp32/GPU package on
one M4 Max (chips differ: the ANE-to-GPU ratio varies by generation, so
benchmark per machine, e.g. `examples/nn_benchmark -b coreml_native -M
chroma_net_fp16.mlpackage -s 128 -u all`). On the **GPU**, fp16 is slightly
*slower* than fp32 (the boundary casts cost more than the kernels save), so
an fp16 package without `ALL` buys nothing.

NN inference in isolation, one ldzeug2 field `[1,C,263,910]`:

| backend | color_cnn | luma_sep |
|---|---|---|
| native CoreML | ≈ 8.3 ms | ≈ 19 ms |
| ORT CoreML EP | ≈ 79 ms | ≈ 157 ms |
| ORT CPU, all cores | ≈ 110 ms | ≈ 392 ms |
| ORT CPU, 1 thread | ≈ 782 ms | ≈ 3014 ms |

For these fp32 packages, `CpuAndGpu` and `All` measure identically (an fp32
program can't be placed on the fp16-only ANE), so the default `CpuAndGpu`
loses nothing.

For nnTransform3D, the 3D FFT/window wrapping the model stay on the CPU
(double-precision FFTW) by default — only the convolution moves to GPU/ANE. An
experimental **GPU-resident** path (macOS 14+) is available by setting
`CHD_NNTRANSFORM3D_COREML_FFT=mps`: the spectrum stays in shared
unified-memory Metal buffers across the MPSGraph device FFT, the magnitude and
mask Metal kernels, and a `cpuAndGPU` CoreML convolution (written in place via
`outputBackings`), avoiding the per-tile host round-trip. It falls back to the
FFTW FFT on older macOS, when no Metal device is present, or on any
setup/runtime failure. In practice it's **not worth enabling**: the run is
conv-bound (the FFT is ~50 ms of the ~770 ms frame), so Layer 2 measured within
1% of the FFTW default while running the FFT at lower precision (f32 vs f64).
Do **not** combine it with `CHD_NN_COREML_ALL` and an fp16 package: a conv
scheduled on the ANE has to sync/copy out of the Metal-resident buffers every
chunk, and the combination measures slower than the plain FFTW path (the
library warns when it detects this pairing).

## nnTransform3D (`chroma_net`)

**Decoder kind:** `CHD_DEC_NN_TRANSFORM3D`. **Model file:** `chroma_net.onnx`.

In the designer's words, nnTransform3D "uses a neural network to process the
amplitude spectrum after performing a 3D FFT (16x16x4) for the Y/C separation."
It replaces the classic NTSC-3D decoder: each 16×16×4 block of the composite
signal is transformed, and a small CNN predicts a separation mask in the
frequency domain.

### Tensor interface

The ONNX graph (opset 11; `Conv` / `LeakyRelu` / `Add` / `Sigmoid`) is:

| | Shape | Dtype | Meaning |
|---|---|---|---|
| input | `[N, 2, 4, 16, 16]` | float32 | Per-block amplitude spectrum; channel 0 is the spectrum, channel 1 a reference (point-reflected) copy. `N` = number of blocks (batched). |
| output | `[N, 1, 4, 16, 16]` | float32 | Single-channel separation mask (sigmoid, 0..1). |

### Model series and the magnitude scale

What matters when you load a `chroma_net` model is its **series**, which fixes
the input magnitude scale:

| Series | Compute precision | `CHD_OPT_NN_INPUT_MAGNITUDE_SCALE` |
|---|---|---|
| v1 series | FP32 | `1.0` |
| v2 | FP16 | `128.0` |

The **v1 series** covers the original `chroma_net` release plus at least one later
retraining — distinct weights, same runtime contract: both feed the raw amplitude
spectrum to the model. **v2** is a separate retraining that runs inference in FP16.

The [`CHD_OPT_NN_INPUT_MAGNITUDE_SCALE`](api-reference.md#option-registry) option
exists specifically to support the v2 FP16 flow. Per the author: "Since the v2
model uses FP16 computation, you need to divide the amplitude spectrum by 128 at
the input to prevent overflow." Set `1.0` for any v1-series model and `128.0`
for v2; a mismatched scale produces garbage.

All series share the same ONNX interface and architecture, and the weights are
stored as FP32 either way (the v2 "FP16" is a runtime-compute choice, not the
stored precision), so a model file does **not** announce its series. Match the
scale to the release the weights came from.

> **Filenames are not authoritative.** Some integrations ship a v1-series
> (scale `1.0`) model under a `chroma_net_v2.onnx` name; applying `128.0` to it
> produces garbage. When unsure, treat an unidentified model as v1-series
> (`1.0`); only use `128.0` for weights you know came from the v2 release.

### Provenance and training

nnTransform3D was trained on 87 interlaced NTSC videos (10 frames each),
software-encoded with `ld-chroma-encoder`, deliberately chosen for heavy motion
and high-frequency detail, with no film source material. The canonical sources
of truth for the model are the author's own harnesses and notes (the v1
modified-ld-chroma-decoder harness, the v2 standalone CUDA harness `main.cu`,
and the author's release quotes).

## ldzeug2 models

The ldzeug2 project targets composite video from NTSC(-J) laserdiscs. Its models
come in two groups; the canonical source of truth is the ldzeug2 project itself.

### Color CNN { #color-cnn }

**Decoder kind:** `CHD_DEC_LDZEUG_COLOR_CNN`. A full colour-decoding network.

| | Shape | Dtype | Meaning |
|---|---|---|---|
| input | `[N, 3, H, W]` | float32 | Three per-field planes: the composite samples, plus synthesized I-carrier (cosine) and Q-carrier (sine) reference planes keyed to the field's subcarrier phase. |
| output | `[N, 3, H, W]` | float32 | Decoded colour. |

Published variants include `color_cnn` v1 and v2 and a denoising variant.

The carrier planes are always keyed to the nominal field/line phase. With
[`CHD_OPT_PHASE_COMPENSATION`](api-reference.md#phase-compensation) the
network's I/Q output is rotated onto each line's measured burst phase, the
same correction the luma-separation kinds apply to their demodulated output.
The correction stays on the output side because the network handles a
composite sitting off the carrier phase as a clean rotation of its output,
whereas carrier planes rotated off the ±1 lattice that training presented
measurably desaturate the decoded chroma.

### Luma separation

**Decoder kinds:** `CHD_DEC_LDZEUG_LUMA_SEP` (per field) and
`CHD_DEC_LDZEUG_LUMA_SEP_FRAME` (per frame). A grayscale Y/C separator.

| | Shape | Dtype | Meaning |
|---|---|---|---|
| input | `[N, 1, H, W]` | float32 | Single grayscale (composite) plane. |
| output | `[N, 1, H, W]` | float32 | Separated luma. |

The per-field model is, in the project's own assessment, "about as good as 3D"
with no ghosting on fast motion but less detail on stationary areas; the
per-frame model is the alternative. The
[`CHD_OPT_NN_CHROMA_BANDPASS`](api-reference.md#option-registry) option applies
to the luma-separation kinds.

## Model-to-decoder summary

| Decoder kind | Model | Input → output | Magnitude scale |
|---|---|---|---|
| `CHD_DEC_NN_TRANSFORM3D` | `chroma_net` (v1 or v2) | `[N,2,4,16,16]` → `[N,1,4,16,16]` | `1.0` (v1) / `128.0` (v2) |
| `CHD_DEC_LDZEUG_COLOR_CNN` | ldzeug2 `color_cnn` | `[N,3,H,W]` → `[N,3,H,W]` | n/a |
| `CHD_DEC_LDZEUG_LUMA_SEP` | ldzeug2 `luma_sep` (field) | `[N,1,H,W]` → `[N,1,H,W]` | n/a |
| `CHD_DEC_LDZEUG_LUMA_SEP_FRAME` | ldzeug2 `luma_sep` (frame) | `[N,1,H,W]` → `[N,1,H,W]` | n/a |

## Obtaining the models

Model weights are distributed by their upstream projects (nnTransform3D by
asdfqazsnbb; ldzeug2 releases by jsaowji), not bundled with libchromadec. Both
sets are released by their authors without usage restrictions and are generally
treated as public domain; confirm the upstream terms for your own use.

<!-- Specific download locations and canonical filenames to be added. -->