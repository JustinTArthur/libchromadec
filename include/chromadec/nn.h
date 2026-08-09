/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef CHROMADEC_NN_H
#define CHROMADEC_NN_H

#include <chromadec/enum.h>
#include <chromadec/errors.h>
#include <chromadec/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Inference backend. Family-grouped: AUTO (best across all backends), then the
 * ONNX Runtime execution providers (the CHD_NN_ORT_* infix), then native non-ORT
 * backends (CHD_NN_<name>, no ORT_ infix). Per-value semantics live in the docs
 * site. */
typedef CHD_ENUM(chd_nn_backend) {
    CHD_NN_BACKEND_AUTO  = 0,   /* best across all backends; inferred from artifact */

    CHD_NN_ORT_AUTO      = 10,  /* ONNX Runtime, per-OS EP fallback chain */
    CHD_NN_ORT_CPU       = 11,
    CHD_NN_ORT_CUDA      = 12,
    CHD_NN_ORT_TENSORRT  = 13,
    CHD_NN_ORT_COREML    = 14,  /* ONNX Runtime CoreML execution provider */
    CHD_NN_ORT_DIRECTML  = 15,
    CHD_NN_ORT_MIGRAPHX  = 16,

    CHD_NN_COREML        = 20   /* native CoreML .mlpackage via MLModel */
} chd_nn_backend_t;

/* Compute units for the native CoreML backend (CHD_NN_COREML); ignored by every
 * ONNX Runtime backend. */
typedef CHD_ENUM(chd_nn_coreml_compute) {
    CHD_NN_COREML_CPU_AND_GPU = 0,  /* default: CPU + GPU, no ANE */
    CHD_NN_COREML_ALL         = 1,  /* CPU + GPU + Apple Neural Engine */
    CHD_NN_COREML_CPU_ONLY    = 2   /* CPU only */
} chd_nn_coreml_compute_t;

/* Inference compute precision. FP16_ALLOWED permits a backend that compiles
 * the model into a device engine to build that engine with mixed fp16/fp32
 * kernels; backends without such a mode ignore it and run the model at its
 * stored precision. Only enable it for weights whose input contract keeps
 * fp16 in range (see the docs site's NN-models page). */
typedef CHD_ENUM(chd_nn_compute_precision) {
    CHD_NN_PRECISION_FP32         = 0,  /* default */
    CHD_NN_PRECISION_FP16_ALLOWED = 1
} chd_nn_compute_precision_t;

typedef struct chd_nn_session_opts {
    chd_nn_backend_t backend;       /* AUTO unless caller pins */
    int32_t device_id;              /* 0 unless multi-GPU */
    int     enable_graph_optim;     /* default 1 */
    int     enable_mem_pattern;     /* default 1 */
    /* Default 1 — our DecoderPool already parallelises across frames; setting
     * intra-op > 1 oversubscribes CPU. A consumer that decodes frames
     * serially has no such parallelism and should raise intra_op_threads
     * (or pass 0 for the ORT default) instead. */
    int32_t inter_op_threads;
    int32_t intra_op_threads;
    /* Native CoreML (CHD_NN_COREML) compute units; ignored otherwise. */
    chd_nn_coreml_compute_t coreml_compute;

    /* Optional directory for caching compiled EP engines/binaries (TensorRT
     * engine plans, MIGraphX compiled models). Avoids the 15-30 s graph
     * compile that runs on the first inference call.
     *   NULL  → library auto-picks a per-user cache dir:
     *             Linux  : $XDG_CACHE_HOME/chromadec or $HOME/.cache/chromadec
     *             macOS  : $HOME/Library/Caches/chromadec
     *             Windows: %LOCALAPPDATA%/chromadec
     *           Directory is created if absent. Auto-pick is the default.
     *   ""    → caching disabled (recompile every load, useful for CI /
     *           hermetic tests / cold-start benchmarking).
     *   path  → use this absolute path (created if it doesn't exist).
     * Currently honoured by the TensorRT and MIGraphX EPs; the CUDA EP
     * uses its own internal PTX cache that isn't configurable here. */
    const char *engine_cache_dir;

    /* Compute precision the backend may use for inference; see
     * chd_nn_compute_precision_t. */
    chd_nn_compute_precision_t precision;

    /* Reserved for future ABI extensions. Initialised to zero by
     * chd_nn_session_opts_default(); set to zero by callers building the
     * struct manually. Adding fields here keeps consumers source-compatible
     * across minor versions. */
    void *reserved[4];
} chd_nn_session_opts_t;

void chd_nn_session_opts_default(chd_nn_session_opts_t *out);

/* Load a model from a file on disk. The backend in `opts` selects the runtime:
 * CHD_NN_BACKEND_AUTO (the default) infers it from the artifact — a `.onnx`
 * loads through ONNX Runtime (EP via the auto chain), a `.mlpackage`/`.mlmodelc`
 * through the native CoreML backend. A pinned backend forces that path and
 * requires the matching artifact. The file only needs to outlive this call. */
chd_status_t chd_nn_model_load_from_file(const char *model_path,
                                         const chd_nn_session_opts_t *opts_or_null,
                                         chd_nn_model_t **out);

/* Load an ONNX model from an in-memory buffer — for callers that embed the
 * model as a compiled-in byte array and want no filesystem dependency.
 * `model_data` points to `model_size` bytes of serialized ONNX. The bytes
 * are consumed during this call and need not outlive it. A backend that can
 * ingest a model buffer is required: ONNX Runtime can; CHD_NN_COREML can't
 * (its .mlpackage is an on-disk bundle) and returns CHD_E_INVALID_ARG. */
chd_status_t chd_nn_model_load_from_memory(const void *model_data,
                                           size_t model_size,
                                           const chd_nn_session_opts_t *opts_or_null,
                                           chd_nn_model_t **out);

void chd_nn_model_free(chd_nn_model_t *m);

/* The backend actually in use after load (e.g. CHD_NN_ORT_CPU when an ORT
 * AUTO chain fell back to CPU, or CHD_NN_COREML for the native backend). */
chd_status_t chd_nn_model_get_active_backend(const chd_nn_model_t *m,
                                             chd_nn_backend_t *out);

int chd_nn_backend_is_available(chd_nn_backend_t b);

#ifdef __cplusplus
}
#endif
#endif
