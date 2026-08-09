// SPDX-License-Identifier: GPL-3.0-or-later

#include "ort_env.h"

#include <stdexcept>

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
