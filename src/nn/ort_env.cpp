// SPDX-License-Identifier: GPL-3.0-or-later

#include "ort_env.h"

#include <stdexcept>

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#include "../common/log.h"

namespace chd::nn {

namespace {

// ORT's default logger writes to stderr on its own. Give it our logging
// function instead, so everything a consumer sees arrives through the one sink
// they installed. These never carry CHD_LOG_F_RETURNED: ORT hands them to us
// out of band, with no relation to whatever call is in progress.
void ORT_API_CALL ortLogSink(void *, OrtLoggingLevel severity, const char *category,
                             const char *logid, const char *code_location,
                             const char *message)
{
    chd::log::Level level = CHD_LOG_INFO;
    switch (severity) {
        case ORT_LOGGING_LEVEL_VERBOSE: level = CHD_LOG_DEBUG; break;
        case ORT_LOGGING_LEVEL_INFO:    level = CHD_LOG_INFO;  break;
        case ORT_LOGGING_LEVEL_WARNING: level = CHD_LOG_WARN;  break;
        case ORT_LOGGING_LEVEL_ERROR:
        case ORT_LOGGING_LEVEL_FATAL:   level = CHD_LOG_ERROR; break;
    }
    if (!chd::log::isEnabled(level)) return;
    chd::log::Stream(level).nospace()
        << "onnxruntime [" << (logid ? logid : "") << ":" << (category ? category : "")
        << " " << (code_location ? code_location : "") << "] "
        << (message ? message : "");
}

std::unique_ptr<Ort::Env> makeEnv()
{
    return std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "chromadec",
                                      ortLogSink, nullptr);
}

// Pin the module containing this library's code into the process for its
// remaining lifetime, so a host that unloads the library (plugin frameworks
// do, from their own exit handlers) cannot unmap it. Once an env exists,
// process-lifetime state points into this module and its dependency chain:
// the deliberately leaked env itself, exit handlers registered inside the
// ONNX Runtime this module links, and the CUDA runtime's registrations when
// the GPU pipeline is built in. Unmapping any of it turns process exit into
// a jump through a dangling handler into unmapped code. Pinning the module
// holds its dependencies with it, and a handle-based unload becomes a
// refcount drop that never unmaps.
void pinSelfModule()
{
#if defined(__linux__)
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&pinSelfModule), &info) != 0 &&
        info.dli_fname != nullptr) {
        dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
    }
#elif defined(_WIN32)
    HMODULE mod = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN |
                           GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                       reinterpret_cast<LPCWSTR>(&pinSelfModule), &mod);
#endif
}

}  // namespace

// A raw pointer, deliberately never deleted at static-destruction time. If
// the env were destroyed from a static destructor, its teardown
// (UnloadSharedProviders -> dlclose of the GPU provider libraries) would run
// inside _dl_fini, interleaved with the CUDA/TensorRT runtimes' own exit
// handlers — the CUDA driver then frees against a corrupted heap and the
// process aborts after main returns, with correct output already written.
// Leaking the env keeps ORT and its providers untouched during _dl_fini;
// chd_shutdown() remains the deterministic teardown for callers that want
// one, and running it before exit is proven safe.
Ort::Env  *OrtEnvSingleton::env_ = nullptr;
std::mutex OrtEnvSingleton::envMutex_;

Ort::Env &OrtEnvSingleton::get()
{
    std::lock_guard<std::mutex> lock(envMutex_);
    if (env_ == nullptr) {
        pinSelfModule();
        env_ = makeEnv().release();
    }
    return *env_;
}

void OrtEnvSingleton::shutdown()
{
    std::lock_guard<std::mutex> lock(envMutex_);
    delete env_;
    env_ = nullptr;
}

}  // namespace chd::nn
