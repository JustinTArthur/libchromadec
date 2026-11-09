// SPDX-License-Identifier: GPL-3.0-or-later
//
// CUDA cuFFT + ORT-IoBinding pipeline for nnTransform3D. The entry point is
// a plain C++ function callable from comb.cpp; the implementation
// (.cu file) contains all CUDA / cuFFT / ORT IoBinding code so that the
// rest of the library can still be compiled by a stock C++ compiler when
// CUDA is not present.
//
// Algorithm + model authorship: asdfqazsnbb (originally Discord-distributed,
// v2 later open-sourced as the standalone nnTransform3D harness).
//
// Ported by inspection from asdfqazsnbb's standalone main.cu (607 LOC,
// the canonical reference) with parameterisation for arbitrary video
// standards (PAL frame dimensions + active-line ranges); the original
// hardcodes NTSC's 910×525 + active range 40-525.

#ifndef CHD_DECODERS_NNTRANSFORM3D_PIPELINE_CUDA_H
#define CHD_DECODERS_NNTRANSFORM3D_PIPELINE_CUDA_H

#include <cstdint>
#include <mutex>
#include <vector>

#include "nntransform3d_window.h"
#include "../../metadata/core.h"

namespace chd::nn { class OrtSession; }

namespace chd::decoders::nntransform3d {

// Run the nnTransform3D 3D-FFT + CNN-mask + IFFT pipeline on a CUDA device.
//
// `currentRaw` / `nextRaw` point at the interlaced baseband samples for
// frame N and frame N+1 (sized `videoParameters.fieldWidth *
// frameHeight`, where frameHeight = videoParameters.fieldHeight * 2 - 1
// laid out as in Comb::FrameBuffer::rawbuffer).
//
// `current{AccChroma,WeightSum}` are the overlap-add accumulators for
// frame N; `next{AccChroma,WeightSum}` are for frame N+1. Both are
// passed by reference and start zeroed, because each tile pass is kept
// separate until the frame is finalized. On return, all four hold this
// call's contributions, brought back to host memory.
//
// `session` must have been built with the CUDA or TensorRT execution
// provider attached. `runMutex` serialises concurrent IoBinding Run()
// calls on the same session per the same tbc-tools-derived locking
// rule the CPU path uses; this implementation also holds `runMutex`
// across the surrounding kernel launches because it uses the CUDA
// default stream — without the lock, worker threads would interleave
// commands on the same device command queue.
//
// `inputMagnitudeScale` is the model's per-channel normalisation
// constant (1.0 for chroma_net v1; 128.0 for v2).
//
// Returns true on success. On any CUDA / cuFFT / ORT failure, returns
// false; the caller falls back to the 2D chroma path (matching the
// CPU body's behaviour on ORT exceptions). All device resources are
// released before returning either way.
bool runCudaPipeline(
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

#endif  // CHD_DECODERS_NNTRANSFORM3D_PIPELINE_CUDA_H
