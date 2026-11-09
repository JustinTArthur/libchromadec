// SPDX-License-Identifier: GPL-3.0-or-later
//
// Native CoreML pipeline for nnTransform3D, the macOS counterpart to the
// CUDA cuFFT / HIP hipFFT pipelines. The ORT CoreML execution provider
// gates out the model's 3D convolution and silently runs it on CPU; this
// path instead drives an offline-converted `.mlpackage` through CoreML so
// the conv reaches GPU/ANE.
//
// The 3D FFT and Hann window wrapping the model are DSP outside the ONNX
// graph (as in the CPU/CUDA paths). Two FFT strategies exist behind one
// seam, decided once and cached:
//
//   Layer 1 (default) — double-precision FFTW on CPU around the CoreML
//                        conv. No macOS-version floor.
//   Layer 2 (opt-in)  — GPU-resident: the spectrum stays in shared
//                        (unified-memory) Metal buffers across MPSGraph FFT →
//                        magnitude (Metal kernel) → cpuAndGPU CoreML conv (via
//                        outputBackings) → mask (Metal kernel) → inverse FFT.
//                        macOS 14.0+. Falls back to Layer 1 on older macOS, a
//                        headless/no-Metal device, or any setup/runtime
//                        failure. Selected only when
//                        CHD_NNTRANSFORM3D_COREML_FFT=mps is set, because the
//                        FFT is f32 (vs the FFTW path's f64) and the
//                        precision/perf trade is per-hardware and must be
//                        measured (see notes/ml-runtime-ep-vs-native.md).

#ifndef CHD_DECODERS_NNTRANSFORM3D_PIPELINE_COREML_H
#define CHD_DECODERS_NNTRANSFORM3D_PIPELINE_COREML_H

#include <cstdint>
#include <mutex>
#include <vector>

#include "nntransform3d_window.h"
#include "../../metadata/core.h"

namespace chd::nn { class CoreMLEngine; }

namespace chd::decoders::nntransform3d {

// Run the nnTransform3D 3D-FFT + CNN-mask + IFFT pipeline through native
// CoreML. Parameter semantics match runCudaPipeline (see
// nntransform3d_pipeline_cuda.h) — `engine` must wrap the nnTransform3D
// `.mlpackage`. Returns true on success; on any CoreML / FFT failure
// returns false and the caller falls back to the 2D chroma path.
bool runCoreMLPipeline(
    const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters,
    const TileGrid &grid,
    const uint16_t *currentRaw,
    const uint16_t *nextRaw,
    bool padF0,  // currentRaw is black padding, not picture
    bool padF1,  // likewise for nextRaw
    std::vector<std::vector<double>> &currentAccChroma,
    std::vector<std::vector<double>> &currentWeightSum,
    std::vector<std::vector<double>> &nextAccChroma,
    std::vector<std::vector<double>> &nextWeightSum,
    chd::nn::CoreMLEngine &engine,
    std::mutex &runMutex,
    double inputMagnitudeScale);

}  // namespace chd::decoders::nntransform3d

#endif  // CHD_DECODERS_NNTRANSFORM3D_PIPELINE_COREML_H
