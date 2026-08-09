// SPDX-License-Identifier: GPL-3.0-or-later

#include "ort_session.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "ort_env.h"
#include "provider_select.h"
#include "../common/log.h"

namespace chd::nn {

namespace {

void applyCommonOptions(Ort::SessionOptions &so, const SessionOptions &opts)
{
    if (opts.intraOpThreads > 0) so.SetIntraOpNumThreads(opts.intraOpThreads);
    if (opts.interOpThreads > 0) so.SetInterOpNumThreads(opts.interOpThreads);
    so.SetGraphOptimizationLevel(opts.enableGraphOptim
                                     ? GraphOptimizationLevel::ORT_ENABLE_ALL
                                     : GraphOptimizationLevel::ORT_DISABLE_ALL);
    if (!opts.enableMemPattern) so.DisableMemPattern();
}

// OS-appropriate per-user cache directory, following each platform's
// conventional location. Falls back to /tmp on POSIX if $HOME is unset.
std::string defaultCacheDir()
{
#if defined(_WIN32)
    if (const char *appdata = std::getenv("LOCALAPPDATA"); appdata && *appdata) {
        return std::string(appdata) + "\\chromadec";
    }
    return "C:\\chromadec-cache";
#elif defined(__APPLE__)
    if (const char *home = std::getenv("HOME"); home && *home) {
        return std::string(home) + "/Library/Caches/chromadec";
    }
    return "/tmp/chromadec-cache";
#else
    if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
        return std::string(xdg) + "/chromadec";
    }
    if (const char *home = std::getenv("HOME"); home && *home) {
        return std::string(home) + "/.cache/chromadec";
    }
    return "/tmp/chromadec-cache";
#endif
}

// Resolve the caller's engine_cache_dir field per the nn.h contract.
// Returns empty string when caching should be disabled; otherwise an
// absolute path that has been mkdir'd. Failures to create the directory
// are logged and treated as "disabled" so a misconfigured cache never
// blocks model loading.
std::string resolveEngineCacheDir(const SessionOptions &opts)
{
    std::string path;
    if (!opts.engineCacheDir.has_value()) {
        path = defaultCacheDir();
    } else if (opts.engineCacheDir->empty()) {
        return {};  // explicit disable
    } else {
        path = *opts.engineCacheDir;
    }

    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
        chd::log::warn() << "chd_nn: failed to create engine cache dir" << path
                         << ":" << ec.message() << "(caching disabled)";
        return {};
    }
    return path;
}

}  // namespace

Ort::SessionOptions OrtSession::prepareSessionOptions(const SessionOptions &opts,
                                                      const std::string &cacheModelPath)
{
    // Force-init the process-wide Ort::Env BEFORE any provider attach
    // touches ORT internals. From ORT 1.25 onward, UpdateCUDAProviderOptions
    // (and likely other V2 provider-options mutators) log through the
    // default Logger that the Ort::Env constructor registers; without an
    // Env in scope they abort with "Attempt to use DefaultLogger but none
    // has been registered." Older ORT versions tolerated the reversed
    // order, which is why this was latent on the macOS CoreML path.
    (void)OrtEnvSingleton::get();

    Ort::SessionOptions sessionOptions;
    applyCommonOptions(sessionOptions, opts);

    EngineCacheConfig cache;
    cache.dir       = resolveEngineCacheDir(opts);
    cache.modelPath = cacheModelPath;

    const auto chain = buildAutoChain(opts.requestedProvider);
    std::string attachError;
    chd_nn_backend_t attached = CHD_NN_ORT_CPU;
    if (!attachProviderChain(sessionOptions, chain, cache, opts.precision,
                             &attached, &attachError)) {
        throw std::runtime_error("provider attach failed: " + attachError);
    }
    activeBackend_ = attached;
    return sessionOptions;
}

OrtSession::OrtSession(const std::string &modelPath, const SessionOptions &opts)
{
    Ort::SessionOptions sessionOptions = prepareSessionOptions(opts, modelPath);

    try {
#if defined(_WIN32)
        // ORT on Windows takes wide-char paths.
        std::wstring wpath(modelPath.begin(), modelPath.end());
        session_ = std::make_unique<Ort::Session>(OrtEnvSingleton::get(), wpath.c_str(),
                                                  sessionOptions);
#else
        session_ = std::make_unique<Ort::Session>(OrtEnvSingleton::get(), modelPath.c_str(),
                                                  sessionOptions);
#endif
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Ort::Session create: ") + e.what());
    }

    // A dlopen'd provider library must never be unmapped once a session has
    // used it (see pinProviderLibraries). Pin after creation, the first
    // point where the provider is guaranteed loaded.
    pinProviderLibraries(activeBackend_);
}

OrtSession::OrtSession(const void *modelData, size_t modelSize, const SessionOptions &opts)
{
    // In-memory models have no path to key the engine cache on; pass empty.
    Ort::SessionOptions sessionOptions = prepareSessionOptions(opts, std::string());

    try {
        session_ = std::make_unique<Ort::Session>(OrtEnvSingleton::get(), modelData,
                                                  modelSize, sessionOptions);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Ort::Session create from memory: ") + e.what());
    }

    pinProviderLibraries(activeBackend_);
}

const std::vector<std::string> &OrtSession::inputNames()
{
    if (!ioNamesCached_) {
        Ort::AllocatorWithDefaultOptions allocator;
        const size_t numInputs = session_->GetInputCount();
        for (size_t i = 0; i < numInputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            inputNames_.emplace_back(name.get());
        }
        const size_t numOutputs = session_->GetOutputCount();
        for (size_t i = 0; i < numOutputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            outputNames_.emplace_back(name.get());
        }
        ioNamesCached_ = true;
    }
    return inputNames_;
}

const std::vector<std::string> &OrtSession::outputNames()
{
    inputNames();  // triggers the lazy populate
    return outputNames_;
}

}  // namespace chd::nn
