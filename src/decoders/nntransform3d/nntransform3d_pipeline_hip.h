// SPDX-License-Identifier: GPL-3.0-or-later
//
// HIP + hipFFT + ORT-IoBinding pipeline for nnTransform3D, the AMD ROCm
// counterpart to the CUDA cuFFT pipeline. Kernels and host-side
// orchestration mirror the CUDA path almost line for line; hipFFT
// deliberately tracks the cuFFT API (literal s/cufft/hipfft/g) so the
// only material difference is the runtime headers + the
// `MemoryInfo("Hip", ...)` token the ORT IoBinding uses to route inputs
// and outputs through device memory.
//
// hipFFT is preferred over rocFFT-direct here because: (a) it gives us
// one shape of code for both vendors, (b) on AMD it dispatches into
// rocFFT under the hood with no measurable overhead, and (c) ORT's
// ROCm and MIGraphX execution providers both consume IoBinding's
// "Hip" memory token regardless of which FFT library wrote the input.

#ifndef CHD_DECODERS_NNTRANSFORM3D_PIPELINE_HIP_H
#define CHD_DECODERS_NNTRANSFORM3D_PIPELINE_HIP_H

#include <cstdint>
#include <mutex>
#include <vector>

#include "nntransform3d_window.h"
#include "../../metadata/core.h"

namespace chd::nn { class OrtSession; }

namespace chd::decoders::nntransform3d {

// Run the nnTransform3D 3D-FFT + CNN-mask + IFFT pipeline on an AMD GPU
// via HIP + hipFFT. See nntransform3d_pipeline_cuda.h for the parameter
// semantics — they're identical. `session` must have been built with
// the ROCm or MIGraphX execution provider attached, on HIP device
// `deviceId`; the pipeline's buffers and plans live on that same device
// so the bound tensors need no copy.
//
// Returns true on success. On any HIP / hipFFT / ORT failure, returns
// false and the caller falls back to the 2D chroma path.
bool runHipPipeline(
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
    chd::nn::OrtSession &session,
    std::mutex &runMutex,
    double inputMagnitudeScale,
    int deviceId);

}  // namespace chd::decoders::nntransform3d

#endif  // CHD_DECODERS_NNTRANSFORM3D_PIPELINE_HIP_H