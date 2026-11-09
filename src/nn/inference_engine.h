// SPDX-License-Identifier: GPL-3.0-or-later
//
// Backend-agnostic neural-network inference interface. The NN decoders
// (ldzeug2 colour/luma-sep, nnTransform3D) hold an InferenceEngine rather
// than reaching for ONNX Runtime types directly, so a non-ORT backend
// (native CoreML) can be dropped in as a second implementation.
//
// Two implementations exist:
//   - OrtEngine    — ONNX Runtime, every execution provider.
//   - CoreMLEngine — native CoreML .mlpackage (macOS only).
//
// The device-resident GPU pipelines for nnTransform3D (CUDA cuFFT / HIP
// hipFFT) bind device pointers through Ort::IoBinding and so remain ORT
// specific; they reach the raw session through OrtEngine::ortSession().
// Every other inference call goes through run() below.

#ifndef CHD_NN_INFERENCE_ENGINE_H
#define CHD_NN_INFERENCE_ENGINE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <chromadec/nn.h>

namespace chd::nn {

// One host-resident float32 input tensor handed to InferenceEngine::run.
// `name` selects the graph input by name; leave empty to bind positionally
// (the i-th input in the model's declared order).
struct TensorSpec {
    const float         *data = nullptr;
    std::vector<int64_t>  shape;
    std::string           name;
};

// Owns the native output objects of one run() so the caller can read them
// as host float spans. The returned pointers stay valid until the
// RunResult is destroyed. Backends return a concrete subclass; callers use
// it only through this interface.
class RunResult {
public:
    virtual ~RunResult() = default;
    virtual size_t               count()             const = 0;
    virtual const float         *data(size_t index)  const = 0;
    virtual std::vector<int64_t> shape(size_t index) const = 0;
};

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    // Run inference over host float tensors. `outputNames` selects which
    // graph outputs to fetch; empty → all outputs in declared order.
    // Throws on backend failure (callers that fall back catch it).
    virtual std::unique_ptr<RunResult> run(
        const std::vector<TensorSpec>  &inputs,
        const std::vector<std::string> &outputNames = {}) = 0;

    // The backend actually in use (after fallback).
    virtual chd_nn_backend_t activeBackend() const = 0;

    // Changes whenever this engine starts producing numerically different
    // output for the same input, so results from before aren't mixed with
    // results from after.
    virtual uint64_t outputGeneration() const { return 0; }

    // Graph input / output names in declared order.
    virtual const std::vector<std::string> &inputNames()  const = 0;
    virtual const std::vector<std::string> &outputNames() const = 0;
};

}  // namespace chd::nn

#endif  // CHD_NN_INFERENCE_ENGINE_H
