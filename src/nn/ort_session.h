// SPDX-License-Identifier: GPL-3.0-or-later
//
// RAII wrapper around one Ort::Session — the internal payload of a
// chd_nn_model_t handle.
//
// Ort::Session is thread-safe; multiple worker threads call
// session->Run(...) concurrently with their own per-call input/output
// tensors. The session is built once and shared across the decoder pool;
// switching models means loading a new chd_nn_model_t and calling
// chd_decoder_set_nn_model again — the old session goes away by RAII once
// all decoders release their handles.

#ifndef CHD_NN_ORT_SESSION_H
#define CHD_NN_ORT_SESSION_H

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <chromadec/nn.h>

namespace chd::nn {

// Options the C ABI gives us when loading a model. Mirrors
// chd_nn_session_opts_t but in C++ form, with defaults already applied.
struct SessionOptions {
    chd_nn_backend_t  requestedProvider = CHD_NN_ORT_AUTO;
    int32_t           deviceId          = 0;
    bool              enableGraphOptim  = true;
    bool              enableMemPattern  = true;
    int32_t           interOpThreads    = 0;   // 0 = ORT default
    int32_t           intraOpThreads    = 1;   // 1 = avoid oversubscription
    // Cache directory for compiled EP engines. nullopt = caller passed
    // NULL on the C ABI -> auto-pick a per-user cache dir. Empty string
    // = caller passed "" -> disable caching entirely.
    std::optional<std::string> engineCacheDir;
    chd_nn_compute_precision_t precision = CHD_NN_PRECISION_FP32;
};

class OrtSession {
public:
    // Load an ONNX model from `modelPath`, attaching execution providers
    // per `opts.requestedProvider` (using the auto chain when AUTO).
    // Throws std::runtime_error on failure with a detailed message.
    OrtSession(const std::string &modelPath, const SessionOptions &opts);

    // Load an ONNX model from `modelSize` bytes at `modelData` — for callers
    // that embed the model rather than shipping a file. ORT copies the bytes
    // during construction, so the buffer need not outlive this call.
    // Throws std::runtime_error on failure with a detailed message.
    OrtSession(const void *modelData, size_t modelSize, const SessionOptions &opts);

    OrtSession(const OrtSession &) = delete;
    OrtSession &operator=(const OrtSession &) = delete;

    // Access the underlying Ort::Session for inference. Thread-safe to
    // call Run() concurrently per ORT docs.
    Ort::Session &session() { return *session_; }

    // The backend ORT actually ended up using (after fallback) — always an
    // ORT-family value (e.g. CHD_NN_ORT_CPU when an AUTO chain fell back to
    // CPU because CUDA wasn't available).
    chd_nn_backend_t activeBackend() const { return activeBackend_; }

    // Best-effort accessor for input/output metadata. Empty until lazily
    // populated on first call.
    const std::vector<std::string> &inputNames();
    const std::vector<std::string> &outputNames();

private:
    // Shared construction prologue: force-inits the process-wide Ort::Env,
    // applies common options, and attaches the provider chain (setting
    // activeBackend_). `cacheModelPath` is only used for engine-cache
    // keying and may be empty for in-memory models. Returns the configured
    // SessionOptions ready to hand to an Ort::Session constructor.
    Ort::SessionOptions prepareSessionOptions(const SessionOptions &opts,
                                              const std::string &cacheModelPath);

    std::unique_ptr<Ort::Session> session_;
    chd_nn_backend_t              activeBackend_ = CHD_NN_ORT_CPU;
    std::vector<std::string>      inputNames_;
    std::vector<std::string>      outputNames_;
    bool                          ioNamesCached_ = false;
};

}  // namespace chd::nn

#endif  // CHD_NN_ORT_SESSION_H
