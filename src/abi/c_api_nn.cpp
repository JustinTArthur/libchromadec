// SPDX-License-Identifier: GPL-3.0-or-later
#include <chromadec/nn.h>

#include <cstring>
#include <memory>
#include <string>

#include "../common/error_state.h"

#if defined(CHD_WITH_NN)
#include "handles.h"
#if defined(CHD_WITH_ORT)
#include "../nn/ort_engine.h"
#include "../nn/ort_session.h"
#include "../nn/provider_select.h"
#endif
#if defined(CHD_WITH_COREML)
#include "../nn/coreml_engine.h"
#endif

namespace {

// True if `backend` names a native (non-ONNX-Runtime) backend.
bool is_native_backend(chd_nn_backend_t backend) {
    return backend == CHD_NN_COREML;
}

// True if `backend` can only be loaded from an on-disk artifact, not a
// contiguous in-memory buffer. Per-backend, not a blanket native check: ORT
// ingests a serialized ONNX byte array directly, and CHD_NN_COREML can't
// because a .mlpackage is a multi-file bundle compiled from a filesystem URL
// with no in-memory load API. A future native backend that accepts a model
// buffer would simply not be listed here.
bool backend_requires_ondisk_artifact(chd_nn_backend_t backend) {
    return backend == CHD_NN_COREML;
}

// True if `path` looks like a CoreML model bundle (.mlpackage / .mlmodelc),
// possibly with a trailing slash. Used to resolve CHD_NN_BACKEND_AUTO.
bool path_is_coreml_bundle(const char *path) {
    std::string p(path);
    while (!p.empty() && p.back() == '/') p.pop_back();
    auto ends_with = [&](const char *suffix) {
        const size_t n = std::strlen(suffix);
        return p.size() >= n && p.compare(p.size() - n, n, suffix) == 0;
    };
    return ends_with(".mlpackage") || ends_with(".mlmodelc");
}

#if defined(CHD_WITH_ORT)
// Translate the C option struct into the internal ORT form, applying the
// "NULL means defaults" contract. `ort_backend` is the resolved ORT-family
// preference (CHD_NN_ORT_AUTO when the caller asked for AUTO).
chd::nn::SessionOptions map_ort_opts(const chd_nn_session_opts_t *opts_or_null,
                                     chd_nn_backend_t ort_backend) {
    chd::nn::SessionOptions opts;
    opts.requestedProvider = ort_backend;
    if (opts_or_null != nullptr) {
        opts.deviceId         = opts_or_null->device_id;
        opts.enableGraphOptim = opts_or_null->enable_graph_optim != 0;
        opts.enableMemPattern = opts_or_null->enable_mem_pattern != 0;
        opts.interOpThreads   = opts_or_null->inter_op_threads;
        opts.intraOpThreads   = opts_or_null->intra_op_threads;
        if (opts_or_null->engine_cache_dir != nullptr) {
            opts.engineCacheDir = std::string(opts_or_null->engine_cache_dir);
        }
        opts.precision = opts_or_null->precision;
    }
    return opts;
}
#endif  // CHD_WITH_ORT

#if defined(CHD_WITH_COREML)
chd::nn::CoreMLComputeUnits map_coreml_compute(const chd_nn_session_opts_t *opts_or_null) {
    if (opts_or_null == nullptr) return chd::nn::CoreMLComputeUnits::CpuAndGpu;
    switch (opts_or_null->coreml_compute) {
        case CHD_NN_COREML_ALL:      return chd::nn::CoreMLComputeUnits::All;
        case CHD_NN_COREML_CPU_ONLY: return chd::nn::CoreMLComputeUnits::CpuOnly;
        case CHD_NN_COREML_CPU_AND_GPU:
        default:                     return chd::nn::CoreMLComputeUnits::CpuAndGpu;
    }
}
#endif

}  // namespace
#endif

extern "C" {

void chd_nn_session_opts_default(chd_nn_session_opts_t *out) {
    if (!out) return;
    out->backend = CHD_NN_BACKEND_AUTO;
    out->device_id = 0;
    out->enable_graph_optim = 1;
    out->enable_mem_pattern = 1;
    out->inter_op_threads = 0;
    /* Default 1: the library's DecoderPool already parallelises across frames;
     * intra-op > 1 oversubscribes CPU. */
    out->intra_op_threads = 1;
    /* NULL → auto-pick per-user cache dir; see chromadec/nn.h. */
    out->engine_cache_dir = nullptr;
    out->coreml_compute = CHD_NN_COREML_CPU_AND_GPU;
    out->precision = CHD_NN_PRECISION_FP32;
    for (size_t i = 0; i < sizeof(out->reserved)/sizeof(out->reserved[0]); ++i) {
        out->reserved[i] = nullptr;
    }
}

#if defined(CHD_WITH_NN)

chd_status_t chd_nn_model_load_from_file(const char *model_path,
                                         const chd_nn_session_opts_t *opts_or_null,
                                         chd_nn_model_t **out) {
    if (model_path == nullptr || out == nullptr) {
        chd::detail::set_last_error("chd_nn_model_load_from_file: null argument");
        return CHD_E_INVALID_ARG;
    }
    *out = nullptr;

    chd_nn_backend_t backend = opts_or_null ? opts_or_null->backend : CHD_NN_BACKEND_AUTO;

    // Resolve which runtime to use. AUTO infers from the artifact on disk;
    // a pinned backend forces its path (and needs the matching artifact).
    const bool want_native =
        is_native_backend(backend) ||
        (backend == CHD_NN_BACKEND_AUTO && path_is_coreml_bundle(model_path));

    if (want_native) {
#if defined(CHD_WITH_COREML)
        try {
            auto engine = std::make_shared<chd::nn::CoreMLEngine>(
                std::string(model_path), map_coreml_compute(opts_or_null));
            auto handle = std::make_unique<chd_nn_model>();
            handle->engine = std::move(engine);
            *out = handle.release();
            return CHD_OK;
        } catch (const std::exception &e) {
            chd::detail::set_last_error(std::string("chd_nn_model_load_from_file: ") + e.what());
            return CHD_E_NN_MODEL_LOAD;
        }
#else
        chd::detail::set_last_error(
            "chd_nn_model_load_from_file: native CoreML backend not built "
            "(requires a macOS build with with_coreml enabled)");
        return CHD_E_NN_BACKEND_UNAVAILABLE;
#endif
    }

#if defined(CHD_WITH_ORT)
    const chd_nn_backend_t ort_backend =
        (backend == CHD_NN_BACKEND_AUTO) ? CHD_NN_ORT_AUTO : backend;
    chd::nn::SessionOptions opts = map_ort_opts(opts_or_null, ort_backend);

    try {
        auto session = std::make_shared<chd::nn::OrtSession>(model_path, opts);
        auto handle = std::make_unique<chd_nn_model>();
        handle->engine = std::make_shared<chd::nn::OrtEngine>(std::move(session));
        *out = handle.release();
        return CHD_OK;
    } catch (const std::exception &e) {
        chd::detail::set_last_error(std::string("chd_nn_model_load_from_file: ") + e.what());
        return CHD_E_NN_MODEL_LOAD;
    }
#else
    chd::detail::set_last_error(
        "chd_nn_model_load_from_file: ONNX Runtime backend not built "
        "(this is a native-CoreML-only build); .onnx models are unavailable");
    return CHD_E_NN_BACKEND_UNAVAILABLE;
#endif
}

chd_status_t chd_nn_model_load_from_memory(const void *model_data,
                                           size_t model_size,
                                           const chd_nn_session_opts_t *opts_or_null,
                                           chd_nn_model_t **out) {
    if (model_data == nullptr || model_size == 0 || out == nullptr) {
        chd::detail::set_last_error("chd_nn_model_load_from_memory: null or empty argument");
        return CHD_E_INVALID_ARG;
    }
    *out = nullptr;

    chd_nn_backend_t backend = opts_or_null ? opts_or_null->backend : CHD_NN_BACKEND_AUTO;
    if (backend_requires_ondisk_artifact(backend)) {
        chd::detail::set_last_error(
            "chd_nn_model_load_from_memory: CHD_NN_COREML requires an on-disk "
            "artifact; use chd_nn_model_load_from_file");
        return CHD_E_INVALID_ARG;
    }

#if defined(CHD_WITH_ORT)
    const chd_nn_backend_t ort_backend =
        (backend == CHD_NN_BACKEND_AUTO) ? CHD_NN_ORT_AUTO : backend;
    chd::nn::SessionOptions opts = map_ort_opts(opts_or_null, ort_backend);

    try {
        auto session = std::make_shared<chd::nn::OrtSession>(model_data, model_size, opts);
        auto handle = std::make_unique<chd_nn_model>();
        handle->engine = std::make_shared<chd::nn::OrtEngine>(std::move(session));
        *out = handle.release();
        return CHD_OK;
    } catch (const std::exception &e) {
        chd::detail::set_last_error(std::string("chd_nn_model_load_from_memory: ") + e.what());
        return CHD_E_NN_MODEL_LOAD;
    }
#else
    chd::detail::set_last_error(
        "chd_nn_model_load_from_memory: ONNX Runtime backend not built "
        "(this is a native-CoreML-only build); in-memory ONNX is unavailable");
    return CHD_E_NN_BACKEND_UNAVAILABLE;
#endif
}

void chd_nn_model_free(chd_nn_model_t *m) {
    delete m;
}

chd_status_t chd_nn_model_get_active_backend(const chd_nn_model_t *m,
                                             chd_nn_backend_t *out) {
    if (m == nullptr || out == nullptr || m->engine == nullptr) {
        chd::detail::set_last_error("chd_nn_model_get_active_backend: null argument");
        return CHD_E_INVALID_ARG;
    }
    *out = m->engine->activeBackend();
    return CHD_OK;
}

int chd_nn_backend_is_available(chd_nn_backend_t b) {
    // CHD_NN_BACKEND_AUTO resolves to whichever backend is built. Reaching this
    // function at all means the framework is compiled in (CHD_WITH_NN), so at
    // least one backend exists — AUTO is therefore always available here.
    // (CHD_NN_ORT_AUTO is ORT-specific and handled by the ORT-family branch.)
    if (b == CHD_NN_BACKEND_AUTO) return 1;

    if (is_native_backend(b)) {
#if defined(CHD_WITH_COREML)
        return 1;
#else
        return 0;
#endif
    }
    // ORT-family backend (incl. CHD_NN_ORT_AUTO).
#if defined(CHD_WITH_ORT)
    return chd::nn::providerIsAvailable(b) ? 1 : 0;
#else
    return 0;
#endif
}

#else  /* !CHD_WITH_NN */

static chd_status_t nn_disabled(const char *what) {
    chd::detail::set_last_error(std::string("chromadec: ") + what +
                                " requires a build with with_nn=true");
    return CHD_E_INTERNAL;
}

chd_status_t chd_nn_model_load_from_file(const char *, const chd_nn_session_opts_t *,
                                         chd_nn_model_t **) {
    return nn_disabled("chd_nn_model_load_from_file");
}

chd_status_t chd_nn_model_load_from_memory(const void *, size_t,
                                           const chd_nn_session_opts_t *,
                                           chd_nn_model_t **) {
    return nn_disabled("chd_nn_model_load_from_memory");
}

void chd_nn_model_free(chd_nn_model_t *) {}

chd_status_t chd_nn_model_get_active_backend(const chd_nn_model_t *,
                                             chd_nn_backend_t *) {
    return nn_disabled("chd_nn_model_get_active_backend");
}

int chd_nn_backend_is_available(chd_nn_backend_t) {
    return 0;
}

#endif  /* CHD_WITH_NN */

}  // extern "C"