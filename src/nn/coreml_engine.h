// SPDX-License-Identifier: GPL-3.0-or-later
//
// Native CoreML implementation of InferenceEngine (macOS only). Drives an
// offline-converted `.mlpackage` through MLModel / MLMultiArray, the route
// that reaches GPU/ANE for models the ORT CoreML execution provider rejects
// — notably nnTransform3D's 3D convolution, which the EP gates out and runs
// on CPU.
//
// The implementation lives in coreml_engine.mm (Objective-C++). This header
// is pure C++ (the Obj-C payload hides behind a pimpl) so plain .cpp / .cu
// translation units can hold a CoreMLEngine pointer.

#ifndef CHD_NN_COREML_ENGINE_H
#define CHD_NN_COREML_ENGINE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "inference_engine.h"

namespace chd::nn {

// Which compute units CoreML may schedule the model on.
//   CpuAndGpu — CPU + GPU, no ANE. The default: the ANE cannot run
//               nnTransform3D's 3D convolution (CoreML fails to "prepare" the
//               model under MLComputeUnitsAll), and the Layer-2 GPU-resident
//               FFT path must exclude the ANE anyway (residency relies on a
//               shared Metal buffer the ANE can't consume).
//   All       — CoreML picks (CPU / GPU / ANE). Faster for ANE-friendly
//               models (e.g. ldzeug2's 2D conv) but unusable for nnTransform3D.
//   CpuOnly   — diagnostics / the graceful fallback when GPU predict fails.
enum class CoreMLComputeUnits { CpuAndGpu, All, CpuOnly };

class CoreMLEngine : public InferenceEngine {
public:
    // Load a `.mlpackage` (or precompiled `.mlmodelc`) from disk. Compiles
    // it if needed and instantiates the MLModel. Throws std::runtime_error
    // with a descriptive message on any failure.
    explicit CoreMLEngine(const std::string &modelPath,
                          CoreMLComputeUnits units = CoreMLComputeUnits::CpuAndGpu);
    ~CoreMLEngine() override;

    CoreMLEngine(const CoreMLEngine &)            = delete;
    CoreMLEngine &operator=(const CoreMLEngine &) = delete;

    std::unique_ptr<RunResult> run(
        const std::vector<TensorSpec>  &inputs,
        const std::vector<std::string> &outputNames) override;

    chd_nn_backend_t activeBackend() const override { return CHD_NN_COREML; }
    uint64_t outputGeneration() const override { return outputGeneration_.load(std::memory_order_acquire); }
    const std::vector<std::string> &inputNames()  const override { return inputNames_; }
    const std::vector<std::string> &outputNames() const override { return outputNames_; }

    CoreMLComputeUnits computeUnits() const { return units_; }

    // GPU-resident predict for the nnTransform3D Layer-2 path: runs a
    // lazily-built cpuAndGPU model that reads `input` and writes results
    // directly into `output` (via CoreML outputBackings), with no internal
    // allocation/copy of the output. `input`/`output` are float32 pointers —
    // typically the .contents of shared (unified-memory) MTLBuffers the
    // MPSGraph FFT and Metal kernels also touch, so the spectrum stays
    // GPU-resident across FFT → conv → IFFT. Thread-safe; returns false on
    // failure so the caller can fall back. Single input / single output.
    bool predictResident(const float *input, const std::vector<int64_t> &inputShape,
                         float *output, const std::vector<int64_t> &outputShape);

private:
    struct Impl;                       // Obj-C++ payload (MLModel + names)
    std::unique_ptr<Impl>    impl_;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    CoreMLComputeUnits       units_;
    std::atomic<uint64_t>    outputGeneration_{0};
};

}  // namespace chd::nn

#endif  // CHD_NN_COREML_ENGINE_H
