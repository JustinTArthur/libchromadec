// SPDX-License-Identifier: GPL-3.0-or-later

#include "plugin_ep.h"

#if defined(__linux__) && ORT_API_VERSION >= 22
#define CHD_PLUGIN_EP_SUPPORTED 1
#endif

#if defined(CHD_PLUGIN_EP_SUPPORTED)

#include <dlfcn.h>

#include <filesystem>
#include <iterator>
#include <mutex>
#include <sstream>
#include <system_error>

#include "ort_env.h"
#include "provider_select.h"
#include "../common/log.h"

namespace chd::nn::plugin {

namespace {

namespace fs = std::filesystem;

struct PluginEpSpec {
    chd_nn_backend_t   backend;
    const char        *displayName;
    // Both the registration name and the EpDevice_EpName the plugin reports:
    // plugins name their EP after the registration name they are given.
    const char        *epName;
    const char        *deviceKind;
    const char *const *fileNames;
    size_t             fileNameCount;
    // Tried relative to the directory of the loaded libonnxruntime.
    const char *const *relativeDirs;
    size_t             relativeDirCount;
};

// In site-packages the plugin wheel's onnxruntime_ep_migraphx/ sits two levels
// up from pip's onnxruntime/capi/libonnxruntime.so.*, and one level up from a
// wheel that bundles its own ONNX Runtime in a <package>.libs/ directory.
const char *const kMigraphxFiles[] = { "libmigraphx-ep.so" };
const char *const kMigraphxDirs[]  = { ".", "../../onnxruntime_ep_migraphx",
                                       "../onnxruntime_ep_migraphx" };

const PluginEpSpec kSpecs[] = {
    { CHD_NN_ORT_MIGRAPHX, "MIGraphX", "MIGraphXExecutionProvider", "AMD GPU",
      kMigraphxFiles, std::size(kMigraphxFiles), kMigraphxDirs, std::size(kMigraphxDirs) },
};

struct PluginState {
    std::string explicitPath;
    // Env generation the cached outcome belongs to; 0 means nothing cached.
    uint64_t    generation = 0;
    bool        registered = false;
    std::string error;
    std::string library;
    // The file this process registered, which can never be swapped out.
    std::string lockedPath;
};

std::mutex  gMutex;
PluginState gStates[std::size(kSpecs)];

const PluginEpSpec *findSpec(chd_nn_backend_t backend, size_t *outIndex = nullptr)
{
    for (size_t i = 0; i < std::size(kSpecs); ++i) {
        if (kSpecs[i].backend == backend) {
            if (outIndex) *outIndex = i;
            return &kSpecs[i];
        }
    }
    return nullptr;
}

std::string takeStatus(OrtStatus *status)
{
    const auto &api = Ort::GetApi();
    std::string msg = api.GetErrorMessage(status);
    api.ReleaseStatus(status);
    return msg;
}

bool fileExists(const fs::path &p)
{
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}

std::string resolveLocked(const PluginEpSpec &spec, const PluginState &st, std::string *outWhyNot)
{
    if (!st.explicitPath.empty()) {
        if (fileExists(st.explicitPath)) return st.explicitPath;
        if (outWhyNot) {
            *outWhyNot = "plugin library " + st.explicitPath +
                         " (set by chd_nn_backend_set_library_path) no longer exists";
        }
        return {};
    }

    std::vector<std::string> tried;
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&OrtGetApiBase), &info) != 0 &&
        info.dli_fname != nullptr) {
        const fs::path ortDir = fs::path(info.dli_fname).parent_path();
        for (size_t d = 0; d < spec.relativeDirCount; ++d) {
            for (size_t f = 0; f < spec.fileNameCount; ++f) {
                const fs::path candidate =
                    (ortDir / spec.relativeDirs[d] / spec.fileNames[f]).lexically_normal();
                if (fileExists(candidate)) return candidate.string();
                tried.push_back(candidate.string());
            }
        }
    }
    if (outWhyNot) {
        std::ostringstream why;
        why << "no plugin library found";
        if (tried.empty()) {
            why << " (the loaded ONNX Runtime's directory could not be resolved)";
        } else {
            why << "; tried";
            for (size_t i = 0; i < tried.size(); ++i) why << (i ? ", " : " ") << tried[i];
        }
        why << ". Name it with chd_nn_backend_set_library_path";
        *outWhyNot = why.str();
    }
    return {};
}

// EpDevices the plugin exposes (optionally GPUs only), in ONNX Runtime's
// enumeration order. GetEpDevices returns a view into the env's live list, so
// copy it out.
std::vector<const OrtEpDevice *> epDevicesLocked(OrtEnv *env, const PluginEpSpec &spec,
                                                 bool gpuOnly)
{
    const auto &api = Ort::GetApi();
    const OrtEpDevice *const *devices = nullptr;
    size_t count = 0;
    std::vector<const OrtEpDevice *> out;
    if (OrtStatus *st = api.GetEpDevices(env, &devices, &count); st != nullptr) {
        chd::log::warn() << "chd_nn: GetEpDevices failed:" << takeStatus(st);
        return out;
    }
    for (size_t i = 0; i < count; ++i) {
        const char *name = api.EpDevice_EpName(devices[i]);
        if (name == nullptr || std::string(name) != spec.epName) continue;
        const OrtHardwareDevice *hw = api.EpDevice_Device(devices[i]);
        if (gpuOnly && (hw == nullptr || api.HardwareDevice_Type(hw) != OrtHardwareDeviceType_GPU)) {
            continue;
        }
        out.push_back(devices[i]);
    }
    return out;
}

bool ensureRegisteredLocked(const PluginEpSpec &spec, PluginState &st, std::string *outError)
{
    OrtEnv *env = static_cast<OrtEnv *>(OrtEnvSingleton::get());
    const uint64_t generation = OrtEnvSingleton::generation();

    if (st.generation != generation) {
        st.generation = generation;
        st.registered = false;
        st.error.clear();
        st.library.clear();

        if (!epDevicesLocked(env, spec, false).empty()) {
            // Another user of this ONNX Runtime's shared env registered it
            // first; registering again would reload the library only to fail.
            st.registered = true;
            st.library    = "(registered by another user of this ONNX Runtime)";
            chd::log::info() << "chd_nn: reusing the" << spec.displayName
                             << "plugin already registered with ONNX Runtime";
        } else {
            std::string why;
            const std::string path = resolveLocked(spec, st, &why);
            if (path.empty()) {
                st.error = why;
            } else {
                // Pin before registering: a failed registration and env
                // teardown both dlclose the library, and the GPU runtime it
                // pulls in leaves exit handlers that must stay mapped.
                dlerror();
                if (dlopen(path.c_str(), RTLD_NOW | RTLD_NODELETE) == nullptr) {
                    const char *msg = dlerror();
                    st.error = "plugin " + path + ": " + (msg ? msg : "dlopen failed");
                } else if (OrtStatus *status = Ort::GetApi().RegisterExecutionProviderLibrary(
                               env, spec.epName, path.c_str());
                           status != nullptr) {
                    st.error = "plugin " + path + ": " + takeStatus(status);
                } else {
                    st.registered = true;
                    st.library    = path;
                    st.lockedPath = path;
                    // Plugins keep pointers into the ONNX Runtime that
                    // registered them, so it has to stay mapped as well.
                    pinProviderLibraries(spec.backend);
                    chd::log::info() << "chd_nn: registered the" << spec.displayName
                                     << "plugin from" << path;
                }
            }
        }
    }

    if (!st.registered && outError) *outError = st.error;
    return st.registered;
}

}  // namespace

bool hasPluginForm(chd_nn_backend_t backend)
{
    return findSpec(backend) != nullptr;
}

chd_status_t setLibraryPath(chd_nn_backend_t backend, const char *pathOrNull,
                            std::string *outError)
{
    size_t index = 0;
    const PluginEpSpec *spec = findSpec(backend, &index);
    if (spec == nullptr) {
        if (outError) *outError = "this backend has no plugin library form";
        return CHD_E_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> lock(gMutex);
    PluginState &st = gStates[index];

    std::string path;
    if (pathOrNull != nullptr && *pathOrNull != '\0') {
        std::error_code ec;
        fs::path abs = fs::absolute(pathOrNull, ec);
        if (ec) abs = pathOrNull;
        abs = abs.lexically_normal();
        if (!fileExists(abs)) {
            if (outError) *outError = "no file at " + abs.string();
            return CHD_E_FILE_NOT_FOUND;
        }
        if (!st.lockedPath.empty() && !fs::equivalent(abs, st.lockedPath, ec)) {
            if (outError) {
                *outError = std::string("the ") + spec->displayName + " plugin is already registered from " +
                            st.lockedPath + " and cannot be replaced for the life of the process";
            }
            return CHD_E_UNSUPPORTED;
        }
        path = abs.string();
    }

    st.explicitPath = path;
    // A cached failure may have come from the old path; let the next load retry.
    if (!st.registered) st.generation = 0;
    return CHD_OK;
}

bool explicitPathSet(chd_nn_backend_t backend)
{
    size_t index = 0;
    if (findSpec(backend, &index) == nullptr) return false;
    std::lock_guard<std::mutex> lock(gMutex);
    return !gStates[index].explicitPath.empty();
}

std::string resolveLibrary(chd_nn_backend_t backend, std::string *outWhyNot)
{
    size_t index = 0;
    const PluginEpSpec *spec = findSpec(backend, &index);
    if (spec == nullptr) {
        if (outWhyNot) *outWhyNot = "this backend has no plugin library form";
        return {};
    }
    std::lock_guard<std::mutex> lock(gMutex);
    return resolveLocked(*spec, gStates[index], outWhyNot);
}

bool isRegistered(chd_nn_backend_t backend)
{
    size_t index = 0;
    if (findSpec(backend, &index) == nullptr) return false;
    const uint64_t generation = OrtEnvSingleton::generation();
    std::lock_guard<std::mutex> lock(gMutex);
    const PluginState &st = gStates[index];
    return generation != 0 && st.generation == generation && st.registered;
}

bool ensureRegistered(chd_nn_backend_t backend, std::string *outError)
{
    size_t index = 0;
    const PluginEpSpec *spec = findSpec(backend, &index);
    if (spec == nullptr) {
        if (outError) *outError = "this backend has no plugin library form";
        return false;
    }
    std::lock_guard<std::mutex> lock(gMutex);
    return ensureRegisteredLocked(*spec, gStates[index], outError);
}

bool attach(Ort::SessionOptions &options, chd_nn_backend_t backend, int32_t deviceId,
            const std::vector<std::pair<std::string, std::string>> &epOptions,
            std::string *outLibrary, std::string *outError)
{
    size_t index = 0;
    const PluginEpSpec *spec = findSpec(backend, &index);
    if (spec == nullptr) {
        if (outError) *outError = "this backend has no plugin library form";
        return false;
    }

    std::vector<const OrtEpDevice *> devices;
    std::string library;
    OrtEnv *env = nullptr;
    {
        std::lock_guard<std::mutex> lock(gMutex);
        PluginState &st = gStates[index];
        if (!ensureRegisteredLocked(*spec, st, outError)) return false;
        env     = static_cast<OrtEnv *>(OrtEnvSingleton::get());
        devices = epDevicesLocked(env, *spec, true);
        library = st.library;
    }
    if (outLibrary) *outLibrary = library;

    if (devices.empty()) {
        if (outError) {
            *outError = "plugin registered from " + library + " but ONNX Runtime lists no " +
                        spec->deviceKind + " for it";
        }
        return false;
    }
    // The plugin pairs its EpDevices with GPU ordinals in this same order and
    // also reads device_id from the options, so the two agree.
    if (deviceId < 0 || static_cast<size_t>(deviceId) >= devices.size()) {
        if (outError) {
            *outError = "device_id " + std::to_string(deviceId) + " is out of range: the plugin lists " +
                        std::to_string(devices.size()) + " " + spec->deviceKind + "(s)";
        }
        return false;
    }

    std::vector<const char *> keys;
    std::vector<const char *> values;
    keys.reserve(epOptions.size());
    values.reserve(epOptions.size());
    for (const auto &kv : epOptions) {
        keys.push_back(kv.first.c_str());
        values.push_back(kv.second.c_str());
    }
    const OrtEpDevice *device = devices[static_cast<size_t>(deviceId)];
    if (OrtStatus *status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
            options, env, &device, 1, keys.data(), values.data(), keys.size());
        status != nullptr) {
        if (outError) {
            *outError = "SessionOptionsAppendExecutionProvider_V2 (plugin " + library +
                        "): " + takeStatus(status);
        }
        return false;
    }
    return true;
}

}  // namespace chd::nn::plugin

#else  // !CHD_PLUGIN_EP_SUPPORTED

namespace chd::nn::plugin {

namespace {
constexpr const char *kUnsupported =
    "plugin execution providers are supported on Linux builds against ONNX Runtime 1.22 or newer";
}

bool hasPluginForm(chd_nn_backend_t) { return false; }

chd_status_t setLibraryPath(chd_nn_backend_t, const char *, std::string *outError)
{
    if (outError) *outError = kUnsupported;
    return CHD_E_UNSUPPORTED;
}

bool explicitPathSet(chd_nn_backend_t) { return false; }

std::string resolveLibrary(chd_nn_backend_t, std::string *outWhyNot)
{
    if (outWhyNot) *outWhyNot = kUnsupported;
    return {};
}

bool isRegistered(chd_nn_backend_t) { return false; }

bool ensureRegistered(chd_nn_backend_t, std::string *outError)
{
    if (outError) *outError = kUnsupported;
    return false;
}

bool attach(Ort::SessionOptions &, chd_nn_backend_t, int32_t,
            const std::vector<std::pair<std::string, std::string>> &, std::string *,
            std::string *outError)
{
    if (outError) *outError = kUnsupported;
    return false;
}

}  // namespace chd::nn::plugin

#endif  // CHD_PLUGIN_EP_SUPPORTED
