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
// Plugin execution-provider registrations live and die with the env: ONNX
// Runtime unregisters them when it is destroyed. ONNX Runtime also keeps one
// refcounted env per process, so ours may be shared with any other user of the
// same libonnxruntime, and a registration may outlive our own shutdown().
//
// This header is internal — the C ABI never exposes Ort types.

#ifndef CHD_NN_ORT_ENV_H
#define CHD_NN_ORT_ENV_H

#include <cstdint>
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

    // Identifies the current env: a new value each time get() constructs
    // one, 0 while none exists. Lets callers tie per-env state, such as a
    // plugin registration, to the env it was made in. Does not construct.
    static uint64_t generation();

    OrtEnvSingleton() = delete;

private:
    static Ort::Env  *env_;   // leaked unless chd_shutdown() runs; see .cpp
    static std::mutex envMutex_;
    static uint64_t   generation_;
};

}  // namespace chd::nn

#endif  // CHD_NN_ORT_ENV_H
