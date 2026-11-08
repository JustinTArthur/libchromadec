// SPDX-License-Identifier: GPL-3.0-or-later
//
// ONNX Runtime implementation of InferenceEngine. Wraps an OrtSession and
// translates the host-tensor run() contract into Ort::Value / Ort::Session
// calls. This is the first (and on every non-Apple platform, the only)
// InferenceEngine implementation.

#ifndef CHD_NN_ORT_ENGINE_H
#define CHD_NN_ORT_ENGINE_H

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "inference_engine.h"
#include "ort_session.h"

namespace chd::nn {

class OrtEngine : public InferenceEngine {
public:
    // Takes shared ownership of an already-built session. Input/output names
    // are snapshotted here (at load time, single-threaded) so the per-call
    // accessors are race-free when workers share the engine.
    explicit OrtEngine(std::shared_ptr<OrtSession> session);

    std::unique_ptr<RunResult> run(
        const std::vector<TensorSpec>  &inputs,
        const std::vector<std::string> &outputNames) override;

    chd_nn_backend_t activeBackend() const override { return session_->activeBackend(); }
    int32_t deviceId() const { return session_->deviceId(); }
    const std::vector<std::string> &inputNames()  const override { return inputNames_; }
    const std::vector<std::string> &outputNames() const override { return outputNames_; }

    // ORT-specific escape hatch for the device-resident GPU pipelines
    // (CUDA cuFFT / HIP hipFFT) that bind device pointers via
    // Ort::IoBinding, and for the comb nnTransform3D CPU fallback body.
    // Only valid on an OrtEngine — recover it with dynamic_cast.
    OrtSession &ortSession() { return *session_; }

private:
    std::shared_ptr<OrtSession> session_;
    std::vector<std::string>    inputNames_;
    std::vector<std::string>    outputNames_;
    Ort::MemoryInfo             memInfo_;
};

}  // namespace chd::nn

#endif  // CHD_NN_ORT_ENGINE_H
