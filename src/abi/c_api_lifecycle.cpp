// SPDX-License-Identifier: GPL-3.0-or-later
#include <chromadec/version.h>
#include <chromadec/video.h>   // chd_init / chd_shutdown declarations
#include <cstring>

#if defined(CHD_WITH_ORT)
#include "../nn/ort_env.h"
#endif

extern "C" {

chd_status_t chd_init(void)     { return CHD_OK; }

void chd_shutdown(void) {
#if defined(CHD_WITH_ORT)
    // chd_shutdown() tears down the Ort::Env singleton. This
    // is opt-in (never auto-registered with atexit),
    // since ORT provider DLLs run their own static destructors on unload
    // and the Windows ordering is fragile.
    chd::nn::OrtEnvSingleton::shutdown();
#endif
}

const char *chd_version_string(void) {
    return CHROMADEC_VERSION_STRING;
}

void chd_version(int *major, int *minor, int *patch) {
    if (major) *major = CHROMADEC_VERSION_MAJOR;
    if (minor) *minor = CHROMADEC_VERSION_MINOR;
    if (patch) *patch = CHROMADEC_VERSION_PATCH;
}

int chd_has_feature(const char *feature) {
    if (!feature) return 0;
    if (std::strcmp(feature, "nn") == 0) {
        // The NN framework is present if any inference backend is built.
#if defined(CHD_WITH_NN)
        return 1;
#else
        return 0;
#endif
    }
    if (std::strcmp(feature, "onnxruntime") == 0) {
#if defined(CHD_WITH_ORT)
        return 1;
#else
        return 0;
#endif
    }
    if (std::strcmp(feature, "cuda") == 0) {
#if defined(CHD_WITH_CUDA)
        return 1;
#else
        return 0;
#endif
    }
    if (std::strcmp(feature, "rocm") == 0) {
#if defined(CHD_WITH_ROCM)
        return 1;
#else
        return 0;
#endif
    }
    if (std::strcmp(feature, "coreml") == 0) {
#if defined(CHD_WITH_COREML)
        return 1;
#else
        return 0;
#endif
    }
    if (std::strcmp(feature, "fftw") == 0) {
#if defined(CHD_WITH_FFTW)
        return 1;
#else
        return 0;
#endif
    }
    if (std::strcmp(feature, "hvd") == 0) {
#if defined(CHD_WITH_HVD)
        return 1;
#else
        return 0;
#endif
    }
    if (std::strcmp(feature, "sqlite") == 0) {
        // SQLite is a hard dependency; if we got here, it's on.
        return 1;
    }
    return 0;
}

}  // extern "C"
