// SPDX-License-Identifier: GPL-3.0-or-later

#include "provider_select.h"

#include <algorithm>
#include <filesystem>
#include <system_error>

#include "../common/log.h"
#include <mutex>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace chd::nn {

namespace {

const char *providerName(ProviderPreference p)
{
    switch (p) {
        case CHD_NN_ORT_AUTO:     return "auto";
        case CHD_NN_ORT_CPU:      return "CPU";
        case CHD_NN_ORT_CUDA:     return "CUDA";
        case CHD_NN_ORT_TENSORRT: return "TensorRT";
        case CHD_NN_ORT_COREML:   return "CoreML";
        case CHD_NN_ORT_DIRECTML: return "DirectML";
        case CHD_NN_ORT_MIGRAPHX: return "MIGraphX";
        default:                  return "unknown";  // non-ORT backend
    }
}

// ORT's name for each provider in Ort::GetAvailableProviders()'s list.
// Names taken from the ORT public API documentation; the C++ wrapper just
// reads the C-style strings out of the OrtApi vector.
const char *availabilityKey(ProviderPreference p)
{
    switch (p) {
        case CHD_NN_ORT_CPU:      return "CPUExecutionProvider";
        case CHD_NN_ORT_CUDA:     return "CUDAExecutionProvider";
        case CHD_NN_ORT_TENSORRT: return "TensorrtExecutionProvider";
        case CHD_NN_ORT_COREML:   return "CoreMLExecutionProvider";
        case CHD_NN_ORT_DIRECTML: return "DmlExecutionProvider";
        case CHD_NN_ORT_MIGRAPHX: return "MIGraphXExecutionProvider";
        default:                  return "";
    }
}

// ─── Linux CUDA driver loader ─────────────────────────────────────────────
// Some Linux distros ship libcuda.so.1 only in /usr/lib/x86_64-linux-gnu;
// some only via the linker's default path; some install the development
// stub libraries under /usr/local/cuda/lib64/stubs that are NOT a real
// driver. Probe in priority order, reject stubs by checking the resolved
// path of cuInit. Ported from tbc-tools/src/ld-chroma-decoder/comb.cpp's
// ensureCudaDriverLoaded (author: harrypm).
//
// macOS / Windows are no-ops — Windows has its own provider-loading
// pitfall (see ensureWindowsCudaProviderLoaded below), and macOS doesn't
// support CUDA at all.
bool ensureCudaDriverLoaded(std::string *outError)
{
#if defined(__linux__)
    static std::once_flag onceFlag;
    static bool         ready = false;
    static std::string  driverError;

    std::call_once(onceFlag, []() {
        auto tryLoad = [](const char *path, std::string &capturedError, bool rejectStubs) -> bool {
            dlerror();   // clear stale state
            void *handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
            if (handle != nullptr) {
                if (rejectStubs) {
                    void *cuInit = dlsym(handle, "cuInit");
                    if (cuInit != nullptr) {
                        Dl_info info{};
                        if (dladdr(cuInit, &info) != 0 && info.dli_fname != nullptr) {
                            const std::string loadedPath = info.dli_fname;
                            if (loadedPath.find("/stubs/") != std::string::npos) {
                                capturedError = std::string("resolved to CUDA stub library: ") + loadedPath;
                                dlclose(handle);
                                return false;
                            }
                        }
                    }
                }
                // Leak the handle on purpose — we want libcuda + the PTX
                // JIT compiler available for the lifetime of the process.
                return true;
            }
            const char *msg = dlerror();
            if (msg != nullptr) capturedError = msg;
            return false;
        };

        std::string libcudaError;
        if (!(tryLoad("/lib/x86_64-linux-gnu/libcuda.so.1",      libcudaError, true) ||
              tryLoad("/usr/lib/x86_64-linux-gnu/libcuda.so.1",  libcudaError, true) ||
              tryLoad("libcuda.so.1",                            libcudaError, true))) {
            if (libcudaError.empty()) libcudaError = "unable to locate libcuda.so.1";
            driverError = "CUDA driver load failed: " + libcudaError;
            return;
        }

        std::string ptxJitError;
        if (!(tryLoad("/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1",     ptxJitError, false) ||
              tryLoad("/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1", ptxJitError, false) ||
              tryLoad("libnvidia-ptxjitcompiler.so.1",                           ptxJitError, false))) {
            if (ptxJitError.empty()) ptxJitError = "unable to locate libnvidia-ptxjitcompiler.so.1";
            driverError = "CUDA PTX JIT compiler load failed: " + ptxJitError;
            return;
        }

        ready = true;
    });

    if (!ready) {
        if (outError) *outError = driverError;
        return false;
    }
#else
    (void)outError;
#endif
    return true;
}

// ─── Windows ORT CUDA provider probe ──────────────────────────────────────
// Deliberately a no-op, matching vapoursynth-analog's
// `tools/patch_tbc_tools.py` patch #5: we MUST NOT LoadLibrary the
// onnxruntime_providers_cuda.dll ourselves before ORT does — that
// triggers the provider's DllMain before ORT has set up its host
// function pointer table, causing the provider's overridden
// `operator new` to crash. Let ORT discover the provider on its own;
// `Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2` will
// dlopen the right things in the right order.
//
// Only declared on Windows so non-Windows builds don't see an unused
// function warning.
#if defined(_WIN32)
bool ensureWindowsCudaProviderLoaded(std::string *outError)
{
    (void)outError;
    return true;
}
#endif

}  // namespace

std::vector<ProviderPreference> buildAutoChain(ProviderPreference requested)
{
    std::vector<ProviderPreference> autoChain;
#if defined(_WIN32)
    autoChain = { CHD_NN_ORT_TENSORRT, CHD_NN_ORT_CUDA, CHD_NN_ORT_DIRECTML, CHD_NN_ORT_CPU };
#elif defined(__APPLE__)
    autoChain = { CHD_NN_ORT_COREML, CHD_NN_ORT_CPU };
#else
    // Linux + everything else. TensorRT leads because it runs chroma_net far
    // faster than the CUDA EP does — measured 6.3x on an A10G, same graph and
    // same frames — at the cost of a one-time engine build per model and GPU
    // architecture, which engine_cache_dir keeps to once per machine. Boxes
    // without TensorRT are unaffected: a failed attach falls through to CUDA.
    autoChain = { CHD_NN_ORT_TENSORRT, CHD_NN_ORT_CUDA, CHD_NN_ORT_MIGRAPHX,
                  CHD_NN_ORT_CPU };
#endif

    if (requested == CHD_NN_ORT_AUTO || requested == CHD_NN_BACKEND_AUTO) return autoChain;

    // Caller pinned a specific provider: try only that one, then fall through
    // to CPU as the universal fallback only if the caller pinned CPU
    // explicitly. The ABI contract is: "Explicit: try only that one;
    // CHD_E_NN_BACKEND_UNAVAILABLE on failure".
    return { requested };
}

bool providerIsAvailable(ProviderPreference provider)
{
    if (provider == CHD_NN_BACKEND_AUTO || provider == CHD_NN_ORT_AUTO) return true;
    if (provider == CHD_NN_ORT_CPU)  return true;  // always
    const std::string key = availabilityKey(provider);
    if (key.empty()) return false;
    try {
        const auto providers = Ort::GetAvailableProviders();
        for (const auto &name : providers) {
            if (name == key) return true;
        }
    } catch (...) {
        return false;
    }
    return false;
}

bool attachCpu(Ort::SessionOptions &, std::string *)
{
    // The CPU provider is ORT's default — nothing to append. Returning true
    // here is the universal "everything tried and we're using CPU" leaf.
    return true;
}

// ─── CUDA attach ──────────────────────────────────────────────────────────
// Ported from tbc-tools/src/ld-chroma-decoder/comb.cpp's
// appendCudaExecutionProvider (lines 368-507, author: harrypm).
// The preferred option set is the one tbc-tools ships with;
// the compatibility option set drops the four newest options
// (use_tf32, prefer_nhwc, the two tunable_op_*) for older ORT builds
// that don't recognise them yet. If even the compatibility set has an
// unsupported option, the regex-driven filter prunes it and retries.
bool attachCuda(Ort::SessionOptions &options, std::string *outError)
{
    std::string driverError;
    if (!ensureCudaDriverLoaded(&driverError)) {
        if (outError) *outError = driverError;
        return false;
    }
#if defined(_WIN32)
    std::string winError;
    if (!ensureWindowsCudaProviderLoaded(&winError)) {
        if (outError) *outError = winError;
        return false;
    }
#endif

    const auto &api = Ort::GetApi();
    OrtCUDAProviderOptionsV2 *cudaOptions = nullptr;
    OrtStatus *status = api.CreateCUDAProviderOptions(&cudaOptions);
    if (status != nullptr) {
        if (outError) *outError = std::string("CreateCUDAProviderOptions: ") +
                                  api.GetErrorMessage(status);
        api.ReleaseStatus(status);
        return false;
    }

    using KV = std::pair<const char *, const char *>;

    // Preferred set — what nnTransform3D was tuned for.
    std::vector<KV> preferred = {
        { "device_id",                    "0"          },
        { "cudnn_conv_algo_search",       "EXHAUSTIVE" },
        { "cudnn_conv_use_max_workspace", "0"          },
        { "do_copy_in_default_stream",    "1"          },
        { "tunable_op_enable",            "0"          },
        { "tunable_op_tuning_enable",     "0"          },
        { "use_tf32",                     "0"          },
        { "prefer_nhwc",                  "0"          },
    };

    // Compatibility set — drops the four newest options for older ORT.
    const std::vector<KV> compatibility = {
        { "device_id",                    "0"          },
        { "cudnn_conv_algo_search",       "EXHAUSTIVE" },
        { "cudnn_conv_use_max_workspace", "0"          },
        { "do_copy_in_default_stream",    "1"          },
        { "tunable_op_enable",            "0"          },
        { "tunable_op_tuning_enable",     "0"          },
    };

    auto applyEntries = [&api, cudaOptions](std::vector<KV> &entries,
                                            std::string *err) -> bool {
        // Loop: try to apply, on "Unknown provider option: \"X\"" filter
        // X out and retry. Caps at entries.size() iterations since each
        // failure prunes at least one entry.
        const std::regex unknownRe(R"re(Unknown provider option:\s*"([^"]+)")re");
        while (!entries.empty()) {
            std::vector<const char *> keys;
            std::vector<const char *> values;
            keys.reserve(entries.size());
            values.reserve(entries.size());
            for (const auto &e : entries) {
                keys.push_back(e.first);
                values.push_back(e.second);
            }
            OrtStatus *st = api.UpdateCUDAProviderOptions(
                cudaOptions, keys.data(), values.data(), keys.size());
            if (st == nullptr) return true;

            const std::string msg = api.GetErrorMessage(st);
            api.ReleaseStatus(st);

            std::smatch m;
            if (!std::regex_search(msg, m, unknownRe)) {
                if (err) *err = msg;
                return false;
            }
            const std::string unknown = m[1].str();
            const auto before = entries.size();
            entries.erase(std::remove_if(entries.begin(), entries.end(),
                              [&](const KV &e) { return unknown == e.first; }),
                          entries.end());
            if (entries.size() == before) {
                if (err) *err = "regex matched unknown option but couldn't remove it: " + unknown;
                return false;
            }
            // Loop again with the filtered list.
        }
        if (err) *err = "no supported CUDA provider options remained after filtering";
        return false;
    };

    std::string applyError;
    if (!applyEntries(preferred, &applyError)) {
        // Preferred set fully filtered out or hit an error we can't fix —
        // try the smaller compatibility set as a clean second attempt.
        std::vector<KV> compatCopy = compatibility;
        applyError.clear();
        if (!applyEntries(compatCopy, &applyError)) {
            if (outError) *outError = "UpdateCUDAProviderOptions: " + applyError;
            api.ReleaseCUDAProviderOptions(cudaOptions);
            return false;
        }
    }

    status = api.SessionOptionsAppendExecutionProvider_CUDA_V2(options, cudaOptions);
    api.ReleaseCUDAProviderOptions(cudaOptions);
    if (status != nullptr) {
        if (outError) *outError = std::string("SessionOptionsAppendExecutionProvider_CUDA_V2: ") +
                                  api.GetErrorMessage(status);
        api.ReleaseStatus(status);
        return false;
    }
    return true;
}

bool attachCoreML(Ort::SessionOptions &options, std::string *outError)
{
#if defined(__APPLE__)
    try {
        const std::unordered_map<std::string, std::string> coremlOpts = {
            { "MLComputeUnits", "ALL" },
            { "ModelFormat",    "MLProgram" },
        };
        options.AppendExecutionProvider("CoreML", coremlOpts);
        return true;
    } catch (const std::exception &e) {
        if (outError) *outError = e.what();
        return false;
    }
#else
    (void)options;
    if (outError) *outError = "CoreML execution provider is only available on macOS";
    return false;
#endif
}

// ─── DirectML attach (Windows only) ──────────────────────────────────────
// Uses the generic AppendExecutionProvider key-value entry point (same
// shape as our CoreML and MIGraphX attaches) rather than the named
// OrtSessionOptionsAppendExecutionProvider_DML export, which lives behind a
// header that only some ONNX Runtime packages ship. The generic path is
// wired up directly in onnxruntime.dll on every package we build against —
// both the classic DirectML NuGet package and the Windows ML NuGet package
// register "DML" through it — so the attach no longer depends on which
// package supplied the build-time headers, only on whether the linked
// onnxruntime.dll actually carries a DirectML.dll to delay-load beside it at
// runtime. Validated end-to-end (session creation, Run, and profiled node
// placement showing DmlExecutionProvider) against the Windows ML package.
bool attachDirectML(Ort::SessionOptions &options, std::string *outError)
{
#if defined(_WIN32)
    try {
        std::unordered_map<std::string, std::string> dmlOpts;
        dmlOpts["device_id"] = "0";
        options.AppendExecutionProvider("DML", dmlOpts);
        return true;
    } catch (const std::exception &e) {
        if (outError) *outError = e.what();
        return false;
    }
#else
    (void)options;
    if (outError) *outError = "DirectML execution provider is only available on Windows";
    return false;
#endif
}

// ─── TensorRT attach ──────────────────────────────────────────────────────
// V2 API. Honours the engine-cache configuration by setting
// trt_engine_cache_enable=1 and trt_engine_cache_path=<dir> on the
// provider options struct; subsequent invocations with the same model +
// input shapes load the cached engine and skip the multi-minute build.
// Requires the CUDA driver to be loaded since TensorRT runs on the same
// hardware as CUDA EP.
//
// CHD_NN_PRECISION_FP16_ALLOWED maps to trt_fp16_enable=1 (TensorRT builds
// the engine with fp16 kernels where they win, fp32 where they don't). The
// engine cache is segregated into an fp16/ subdirectory in that mode: the
// EP does not reliably invalidate a cached engine when the precision flags
// change, so sharing one directory can serve an fp32 engine to an fp16
// session or vice versa.
bool attachTensorRT(Ort::SessionOptions &options,
                    const EngineCacheConfig &cache,
                    chd_nn_compute_precision_t precision,
                    std::string *outError)
{
    std::string driverError;
    if (!ensureCudaDriverLoaded(&driverError)) {
        if (outError) *outError = "TensorRT requires CUDA driver: " + driverError;
        return false;
    }
    const bool fp16 = precision == CHD_NN_PRECISION_FP16_ALLOWED;

    std::string cacheDir = cache.dir;
    if (!cacheDir.empty() && fp16) {
        std::error_code ec;
        const auto sub = std::filesystem::path(cacheDir) / "fp16";
        std::filesystem::create_directories(sub, ec);
        if (ec) {
            chd::log::warn() << "chd_nn: failed to create fp16 engine cache dir"
                             << sub.string() << ":" << ec.message()
                             << "(TensorRT engine caching disabled)";
            cacheDir.clear();
        } else {
            cacheDir = sub.string();
        }
    }

    const auto &api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2 *trtOptions = nullptr;
    OrtStatus *status = api.CreateTensorRTProviderOptions(&trtOptions);
    if (status != nullptr) {
        if (outError) *outError = std::string("CreateTensorRTProviderOptions: ") +
                                  api.GetErrorMessage(status);
        api.ReleaseStatus(status);
        return false;
    }

    std::vector<const char *> keys;
    std::vector<const char *> values;
    if (fp16) {
        keys.push_back("trt_fp16_enable");
        values.push_back("1");
    }
    if (!cacheDir.empty()) {
        keys.push_back("trt_engine_cache_enable");
        values.push_back("1");
        keys.push_back("trt_engine_cache_path");
        values.push_back(cacheDir.c_str());
    }
    if (!keys.empty()) {
        status = api.UpdateTensorRTProviderOptions(trtOptions, keys.data(), values.data(),
                                                   keys.size());
        if (status != nullptr) {
            if (outError) *outError = std::string("UpdateTensorRTProviderOptions: ") +
                                      api.GetErrorMessage(status);
            api.ReleaseStatus(status);
            api.ReleaseTensorRTProviderOptions(trtOptions);
            return false;
        }
    }

    status = api.SessionOptionsAppendExecutionProvider_TensorRT_V2(options, trtOptions);
    api.ReleaseTensorRTProviderOptions(trtOptions);
    if (status != nullptr) {
        if (outError) *outError = std::string("SessionOptionsAppendExecutionProvider_TensorRT_V2: ") +
                                  api.GetErrorMessage(status);
        api.ReleaseStatus(status);
        return false;
    }
    return true;
}

// ─── MIGraphX attach (Linux / AMD) ───────────────────────────────────────
// Uses the generic AppendExecutionProvider key-value entry point (same
// shape as our CoreML attach) rather than the OrtMIGraphXProviderOptions
// struct. That struct's layout has shifted across ORT versions and AMD's
// pre-built wheels were built against a different layout than upstream's
// headers, so passing the struct produced "Failed to map enum value to
// name" failures from arena_extend_strategy reading off the end. The
// string-keyed map is ABI-stable across builds — unknown keys are
// silently ignored, recognised ones are parsed by the EP's own option
// reader.
bool attachMIGraphX(Ort::SessionOptions &options,
                    const EngineCacheConfig &cache,
                    std::string *outError)
{
#if defined(__linux__)
    try {
        std::unordered_map<std::string, std::string> mxOpts;
        mxOpts["device_id"] = "0";
        if (!cache.dir.empty()) {
            // migraphx_model_cache_dir is recognised by newer ORT MIGraphX
            // EPs as a directory that the provider manages files inside.
            // Older ORT builds reject unknown option keys, so we only set
            // this single key when caching is requested — the EP either
            // honours it or ignores it; in older builds where it's
            // unrecognised, the key-value parser fails the attach, which
            // would be a regression. If we encounter ORT versions that
            // reject this key, fall back to leaving cache.dir unused on
            // MIGraphX (TensorRT will still be cached via its own option
            // path).
            mxOpts["migraphx_model_cache_dir"] = cache.dir;
        }
        options.AppendExecutionProvider("MIGraphX", mxOpts);
        return true;
    } catch (const std::exception &e) {
        if (outError) *outError = e.what();
        return false;
    }
#else
    (void)options;
    (void)cache;
    if (outError) *outError = "MIGraphX execution provider is only available on Linux";
    return false;
#endif
}

// ORT loads GPU execution providers by dlopening their libraries
// (libonnxruntime_providers_{migraphx,cuda,tensorrt,shared}.so) and dlcloses
// them again when the last OrtEnv is released. Each provider's runtime stack
// queues work that outlives that unload: libmigraphx registers exit handlers
// (a static vector of migraphx::dynamic_loader whose destructor libc runs
// from exit()), and the CUDA/TensorRT providers leave atexit handlers and
// static destructors from cudart/cuDNN/nvinfer behind. Once the library is
// unmapped, those handlers run against unmapped code and the process takes a
// SIGSEGV after main returns — a crash whose backtrace shows a PC inside no
// loaded library and no valid caller frame. Promoting the provider library
// to RTLD_NODELETE turns ORT's dlclose into a refcount drop that never
// unmaps, so the handlers run against live code at exit.
void pinProviderLibraries(chd_nn_backend_t provider)
{
#if defined(__linux__)
    // The TensorRT provider layers on the CUDA one; both route dispatch
    // through the shared provider bridge, so each pin set includes its
    // full dependency chain. RTLD_NOLOAD makes a listed library that this
    // session never loaded a no-op rather than a fresh load.
    static const char *const cudaSet[]     = { "libonnxruntime_providers_cuda.so",
                                               "libonnxruntime_providers_shared.so" };
    static const char *const tensorrtSet[] = { "libonnxruntime_providers_tensorrt.so",
                                               "libonnxruntime_providers_cuda.so",
                                               "libonnxruntime_providers_shared.so" };
    static const char *const migraphxSet[] = { "libonnxruntime_providers_migraphx.so",
                                               "libonnxruntime_providers_shared.so" };

    const char *const *sonames = nullptr;
    size_t             count   = 0;
    switch (provider) {
        case CHD_NN_ORT_CUDA:     sonames = cudaSet;     count = 2; break;
        case CHD_NN_ORT_TENSORRT: sonames = tensorrtSet; count = 3; break;
        case CHD_NN_ORT_MIGRAPHX: sonames = migraphxSet; count = 2; break;
        default:                  return;  // no dlopen'd provider library
    }

    static std::mutex pinMutex;
    static std::unordered_set<std::string> pinned;
    std::lock_guard<std::mutex> lock(pinMutex);

    const int flags = RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE;

    // The providers sit beside whichever libonnxruntime this process loaded;
    // resolve that directory rather than trusting the search path to agree
    // with ORT's own lookup.
    Dl_info info{};
    const bool haveOrtDir =
        dladdr(reinterpret_cast<void *>(&OrtGetApiBase), &info) != 0 &&
        info.dli_fname != nullptr;

    if (haveOrtDir && pinned.insert(info.dli_fname).second) {
        // Pin the ORT core on the same terms. A provider's registry holds
        // pointers into the ORT instance that first loaded it, so the two
        // have to stay mapped together: a host that unloads libchromadec
        // and loads it again would otherwise reach a fresh ORT through the
        // surviving provider and fault in InitializeRegistry against the
        // previous ORT's dead state.
        dlopen(info.dli_fname, flags);
    }

    for (size_t i = 0; i < count; ++i) {
        if (!pinned.insert(sonames[i]).second) continue;
        if (haveOrtDir) {
            const std::string path =
                (std::filesystem::path(info.dli_fname).parent_path() / sonames[i]).string();
            if (dlopen(path.c_str(), flags) != nullptr) continue;
        }
        dlopen(sonames[i], flags);
    }
#else
    (void)provider;
#endif
}

bool attachProviderChain(Ort::SessionOptions &options,
                         const std::vector<ProviderPreference> &chain,
                         const EngineCacheConfig &cache,
                         chd_nn_compute_precision_t precision,
                         ProviderPreference *outAttached,
                         std::string *outError)
{
    std::ostringstream errs;
    for (const auto provider : chain) {
        if (!providerIsAvailable(provider)) {
            errs << providerName(provider) << ": not available in this ORT build\n";
            continue;
        }

        bool ok = false;
        std::string err;
        switch (provider) {
            case CHD_NN_ORT_CPU:      ok = attachCpu     (options, &err);        break;
            case CHD_NN_ORT_CUDA:     ok = attachCuda    (options, &err);        break;
            case CHD_NN_ORT_COREML:   ok = attachCoreML  (options, &err);        break;
            case CHD_NN_ORT_DIRECTML: ok = attachDirectML(options, &err);        break;
            case CHD_NN_ORT_TENSORRT: ok = attachTensorRT(options, cache, precision, &err); break;
            case CHD_NN_ORT_MIGRAPHX: ok = attachMIGraphX(options, cache, &err); break;
            default:
                // AUTO sentinels and non-ORT backends can't appear in a
                // resolved ORT chain.
                err = "not a concrete ORT execution provider";
                break;
        }
        if (ok) {
            if (outAttached) *outAttached = provider;
            return true;
        }
        errs << providerName(provider) << ": " << err << "\n";
    }

    if (outError) *outError = errs.str();
    return false;
}

}  // namespace chd::nn
