// SPDX-License-Identifier: GPL-3.0-or-later
//
// ONNX Runtime plugin execution providers: EPs that ship as their own shared
// library and are registered with the env at run time, instead of being built
// into ONNX Runtime. A static table maps each backend that has a plugin form
// to the name it registers under and the file names to look for.
//
// A registration belongs to the env and lasts as long as the env does; ONNX
// Runtime unregisters every plugin when the env is destroyed. Nothing here
// unregisters, because doing so while a session still uses the EP is
// forbidden. Each library is pinned into the process before it is registered,
// so neither a failed registration nor env teardown can unmap it while exit
// handlers it queued are still pending (see pinProviderLibraries).
//
// Implemented on Linux against ONNX Runtime 1.22 or newer. Elsewhere no
// backend has a plugin form and every call reports that.

#ifndef CHD_NN_PLUGIN_EP_H
#define CHD_NN_PLUGIN_EP_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <chromadec/errors.h>
#include <chromadec/nn.h>

namespace chd::nn::plugin {

// True if `backend` can be attached through a plugin library in this build.
bool hasPluginForm(chd_nn_backend_t backend);

// Set (or, with NULL or "", clear) the library to register for `backend`.
// The file must exist. Once a library has been registered, naming a
// different file returns CHD_E_UNSUPPORTED. Returns CHD_E_UNSUPPORTED for a
// backend without a plugin form; `outError` explains any non-OK status.
chd_status_t setLibraryPath(chd_nn_backend_t backend, const char *pathOrNull,
                            std::string *outError);

// True if the caller named a library through setLibraryPath.
bool explicitPathSet(chd_nn_backend_t backend);

// The library registration would use, without loading anything: the explicit
// path, else the first candidate file beside the loaded ONNX Runtime. Empty
// when none exists, with `outWhyNot` listing what was tried.
std::string resolveLibrary(chd_nn_backend_t backend, std::string *outWhyNot);

// True if the plugin is registered with the current env. Never registers.
bool isRegistered(chd_nn_backend_t backend);

// Register the plugin with the process-wide env unless it already is.
// The outcome, success or failure, is cached for the life of that env.
bool ensureRegistered(chd_nn_backend_t backend, std::string *outError);

// Register if needed, then append the plugin EP for GPU ordinal `deviceId`
// to `options` with the given EP options. `outLibrary` receives the
// registered library's path for logging.
bool attach(Ort::SessionOptions &options, chd_nn_backend_t backend, int32_t deviceId,
            const std::vector<std::pair<std::string, std::string>> &epOptions,
            std::string *outLibrary, std::string *outError);

}  // namespace chd::nn::plugin

#endif  // CHD_NN_PLUGIN_EP_H
