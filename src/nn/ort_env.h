// SPDX-License-Identifier: GPL-3.0-or-later
//
// Process-wide ONNX Runtime Ort::Env singleton.
//
// One Ort::Env per process — it's the heaviest ORT object,
// thread-safe for sharing, and ORT recommends using a single env across all
// sessions. The singleton is constructed lazily on first get() via
// std::call_once; shutdown() tears it down (only called from chd_shutdown(),
// which is opt-in to avoid Windows DLL-unload ordering
// hazards).
//
// This header is internal — the C ABI never exposes Ort types.

#ifndef CHD_NN_ORT_ENV_H
#define CHD_NN_ORT_ENV_H

#include <mutex>

#include <onnxruntime_cxx_api.h>

namespace chd::nn {

class OrtEnvSingleton {
public:
    // Returns the process-wide Ort::Env, constructing it on first call.
    // Thread-safe. Throws Ort::Exception (via std::runtime_error) if env
    // construction fails — caller should propagate as CHD_E_INTERNAL.
    static Ort::Env &get();

    // Tears down the env. Safe to call when env was never constructed.
    // After this returns, subsequent get() calls will reconstruct lazily
    // (matches ORT's recommendation that env can be recreated after
    // shutdown if needed).
    //
    // Only called from chd_shutdown(). This is intentionally NOT
    // wired to atexit() — ORT provider DLLs run their own static
    // destructors on unload and Windows ordering is fragile.
    static void shutdown();

    OrtEnvSingleton() = delete;

private:
    static Ort::Env  *env_;   // leaked unless chd_shutdown() runs; see .cpp
    static std::mutex envMutex_;
};

}  // namespace chd::nn

#endif  // CHD_NN_ORT_ENV_H
