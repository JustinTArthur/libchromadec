// SPDX-License-Identifier: GPL-3.0-or-later
//
// Smoke tests for the NN framework — Ort::Env singleton,
// provider availability, and error handling on the chd_nn_model_load_from_file
// and chd_nn_model_load_from_memory paths. Does NOT exercise inference on a
// real model; that lands in a
// follow-up with the dummy ONNX model fixture once nnTransform3D / ldzeug2
// have something to feed.
//
// The test compiles into a no-op pass when the library was built with
// with_nn=false (chd_has_feature("nn") returns 0 → we skip the body).

#include <chromadec/errors.h>
#include <chromadec/nn.h>
#include <chromadec/version.h>
#include <chromadec/video.h>

#include "nn_test_model.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond \
                  << " (last_error: " << (chd_last_error() ? chd_last_error() : "(none)") << ")\n"; \
        return 1; \
    } \
} while (0)

int testLifecycle() {
    // chd_init must be idempotent + safe to call multiple times.
    REQUIRE(chd_init() == CHD_OK);
    REQUIRE(chd_init() == CHD_OK);
    chd_shutdown();
    // After shutdown the env reconstructs lazily — should be safe to
    // re-init + reuse.
    REQUIRE(chd_init() == CHD_OK);
    chd_shutdown();
    return 0;
}

int testFeatureFlag() {
    // "nn" feature must report whatever the build said. The test only runs
    // body assertions when NN is on (we're built against ORT here).
    REQUIRE(chd_has_feature("nn") == 1);
    REQUIRE(chd_has_feature("sqlite") == 1);
    REQUIRE(chd_has_feature("nonsense-feature") == 0);
    REQUIRE(chd_has_feature(nullptr) == 0);
    return 0;
}

int testProviderAvailability() {
    // CHD_NN_BACKEND_AUTO is available whenever any backend is built — and the
    // test body only runs when chd_has_feature("nn") is 1, so it must hold here.
    REQUIRE(chd_nn_backend_is_available(CHD_NN_BACKEND_AUTO) == 1);

    // The CHD_NN_ORT_* backends and the ORT auto chain are ONNX-Runtime
    // concepts; only assert their availability when the ORT backend is built
    // (a native-CoreML-only build reports them unavailable).
    if (chd_has_feature("onnxruntime") == 1) {
        // CPU is always available; the ORT auto chain too.
        REQUIRE(chd_nn_backend_is_available(CHD_NN_ORT_CPU) == 1);
        REQUIRE(chd_nn_backend_is_available(CHD_NN_ORT_AUTO) == 1);
#if defined(__APPLE__)
        // CoreML EP should be present in any current macOS ORT build (Homebrew
        // 1.26.0 bundles it).
        REQUIRE(chd_nn_backend_is_available(CHD_NN_ORT_COREML) == 1);
        // The DirectML / TensorRT / MIGraphX providers are Windows-/Linux-only
        // and absent on macOS.
        REQUIRE(chd_nn_backend_is_available(CHD_NN_ORT_DIRECTML) == 0);
        REQUIRE(chd_nn_backend_is_available(CHD_NN_ORT_MIGRAPHX) == 0);
#endif

        // A CI leg that exists to cover one execution provider names it here,
        // so picking up the wrong ONNX Runtime package fails the job instead of
        // passing on a CPU-only runtime. Availability is a property of the
        // linked ORT build rather than of the hardware, so this holds on a
        // runner with no compatible device; whether that provider then attaches
        // is what CHD_TEST_EXPECT_NN_BACKEND covers, on hosts that have one.
        if (const char *require = std::getenv("CHD_TEST_REQUIRE_ORT_PROVIDER");
            require != nullptr && *require != '\0') {
            REQUIRE(chd_nn_backend_is_available(
                        static_cast<chd_nn_backend_t>(std::atoi(require))) == 1);
        }
    }

    // Native CoreML availability tracks its build feature flag, independent of
    // whether the ORT backend is present.
    REQUIRE(chd_nn_backend_is_available(CHD_NN_COREML) == (chd_has_feature("coreml") ? 1 : 0));

    // 99 names no enumerator but is still a valid value of the enum type
    // itself (fixed int32_t underlying type); it is simply never available.
    REQUIRE(chd_nn_backend_is_available(static_cast<chd_nn_backend_t>(99)) == 0);
    return 0;
}

int testSessionOptsDefault() {
    chd_nn_session_opts_t opts{};
    chd_nn_session_opts_default(&opts);
    REQUIRE(opts.backend            == CHD_NN_BACKEND_AUTO);
    REQUIRE(opts.device_id          == 0);
    REQUIRE(opts.enable_graph_optim == 1);
    REQUIRE(opts.enable_mem_pattern == 1);
    REQUIRE(opts.intra_op_threads   == 1);
    REQUIRE(opts.coreml_compute     == CHD_NN_COREML_CPU_AND_GPU);
    return 0;
}

int testLoadNonexistentModel() {
    // Passing a path that doesn't exist must return CHD_E_NN_MODEL_LOAD
    // (or CHD_E_INVALID_ARG for the null cases), never crash, and never
    // leak the out pointer.
    chd_nn_model_t *m = nullptr;
    REQUIRE(chd_nn_model_load_from_file(nullptr, nullptr, &m) == CHD_E_INVALID_ARG);
    REQUIRE(m == nullptr);

    // A `.onnx` path resolves to the ORT backend under AUTO. With ORT built it
    // fails to load (missing file → CHD_E_NN_MODEL_LOAD); in a CoreML-only build
    // the ORT backend is unavailable instead. Either way it must not crash, and
    // the error detail names the entry point.
    if (chd_has_feature("onnxruntime") == 1) {
        REQUIRE(chd_nn_model_load_from_file("/nonexistent/path/to/model.onnx", nullptr, &m)
                == CHD_E_NN_MODEL_LOAD);
    } else {
        REQUIRE(chd_nn_model_load_from_file("/nonexistent/path/to/model.onnx", nullptr, &m)
                == CHD_E_NN_BACKEND_UNAVAILABLE);
    }
    REQUIRE(m == nullptr);
    const char *err = chd_last_error();
    REQUIRE(err != nullptr);
    REQUIRE(std::strstr(err, "chd_nn_model_load_from_file") != nullptr);
    return 0;
}

int testLoadFromMemoryInvalid() {
    // The in-memory entry point must reject null/empty buffers and the null
    // out pointer with CHD_E_INVALID_ARG, never crash, never leak. Garbage
    // (non-ONNX) bytes must fail the load cleanly with CHD_E_NN_MODEL_LOAD.
    chd_nn_model_t *m = nullptr;
    const unsigned char dummy[4] = { 0, 1, 2, 3 };

    REQUIRE(chd_nn_model_load_from_memory(nullptr, 4, nullptr, &m) == CHD_E_INVALID_ARG);
    REQUIRE(m == nullptr);
    REQUIRE(chd_nn_model_load_from_memory(dummy, 0, nullptr, &m) == CHD_E_INVALID_ARG);
    REQUIRE(m == nullptr);
    REQUIRE(chd_nn_model_load_from_memory(dummy, 4, nullptr, nullptr) == CHD_E_INVALID_ARG);

    // Well-formed call, but the bytes aren't a valid ONNX model. With ORT built,
    // it rejects them (CHD_E_NN_MODEL_LOAD); in a CoreML-only build the in-memory
    // ORT path is unavailable (CHD_E_NN_BACKEND_UNAVAILABLE). Either way the
    // message names the entry point.
    if (chd_has_feature("onnxruntime") == 1) {
        REQUIRE(chd_nn_model_load_from_memory(dummy, sizeof(dummy), nullptr, &m)
                == CHD_E_NN_MODEL_LOAD);
    } else {
        REQUIRE(chd_nn_model_load_from_memory(dummy, sizeof(dummy), nullptr, &m)
                == CHD_E_NN_BACKEND_UNAVAILABLE);
    }
    REQUIRE(m == nullptr);
    const char *err = chd_last_error();
    REQUIRE(err != nullptr);
    REQUIRE(std::strstr(err, "chd_nn_model_load_from_memory") != nullptr);
    return 0;
}

int testLoadRealModel() {
    // Drive the PUBLIC C ABI loaders against a real ONNX model — both the
    // file and in-memory entry points must succeed, produce a usable handle,
    // and resolve the same execution provider (same model + same default
    // opts). This is the happy-path counterpart to the failure-only checks
    // above, and the only place the public chd_nn_model_load_from_memory
    // symbol is exercised against real weights. Skips when no model is
    // discoverable on this machine.
    // The committed fixture is ONNX, loaded through the ORT backend. Skip when
    // ORT isn't built (e.g. a native-CoreML-only build).
    if (chd_has_feature("onnxruntime") != 1) {
        std::cout << "Skipping real-model load test (ONNX Runtime backend not built).\n";
        return 0;
    }
    const std::string modelPath = chd_test::modelFromEnvOrFixture("CHD_TEST_NN_MODEL");
    if (modelPath.empty()) {
        std::cout << "Skipping real-model load test (no model fixture available).\n";
        return 0;
    }

    // Honour a pinned provider when CI asks for one; NULL opts (the AUTO chain)
    // otherwise.
    chd_nn_session_opts_t pinned{};
    chd_nn_session_opts_default(&pinned);
    const int wanted = chd_test::backendFromEnv();
    if (wanted != 0) pinned.backend = static_cast<chd_nn_backend_t>(wanted);
    const chd_nn_session_opts_t *opts = wanted != 0 ? &pinned : nullptr;

    // File loader.
    chd_nn_model_t *fileModel = nullptr;
    REQUIRE(chd_nn_model_load_from_file(modelPath.c_str(), opts, &fileModel) == CHD_OK);
    REQUIRE(fileModel != nullptr);
    chd_nn_backend_t fileBackend = CHD_NN_BACKEND_AUTO;
    REQUIRE(chd_nn_model_get_active_backend(fileModel, &fileBackend) == CHD_OK);

    // Memory loader: same bytes, via the in-memory entry point.
    std::vector<char> bytes = chd_test::readFileBytes(modelPath);
    REQUIRE(!bytes.empty());

    chd_nn_model_t *memModel = nullptr;
    REQUIRE(chd_nn_model_load_from_memory(bytes.data(), bytes.size(), opts, &memModel)
            == CHD_OK);
    REQUIRE(memModel != nullptr);
    chd_nn_backend_t memBackend = CHD_NN_BACKEND_AUTO;
    REQUIRE(chd_nn_model_get_active_backend(memModel, &memBackend) == CHD_OK);

    // Identical model + opts must resolve to the same backend regardless of
    // how the bytes were sourced.
    REQUIRE(memBackend == fileBackend);

    // A GPU provider that fails to attach falls back to CPU with the suite
    // still green, so pass/fail alone can't tell a GPU run from a silent CPU
    // one. CI sets CHD_TEST_EXPECT_NN_BACKEND to the chd_nn_backend_t value
    // the job exists to prove (12 ORT+CUDA, 13 ORT+TensorRT, 16 ORT+MIGraphX)
    // and this turns the fallback into a failure.
    if (const char *expect = std::getenv("CHD_TEST_EXPECT_NN_BACKEND");
        expect != nullptr && *expect != '\0') {
        REQUIRE(static_cast<int>(memBackend) == std::atoi(expect));
    }

    chd_nn_model_free(memModel);
    chd_nn_model_free(fileModel);
    std::cout << "Loaded real model via file + memory (" << bytes.size()
              << " bytes), backend " << static_cast<int>(memBackend) << ".\n";
    return 0;
}

int testCoreMLLoad() {
    // Native CoreML is now selected through the unified loader by pinning
    // opts.backend = CHD_NN_COREML (or by passing a .mlpackage under AUTO).
    chd_nn_session_opts_t opts{};
    chd_nn_session_opts_default(&opts);
    opts.backend = CHD_NN_COREML;

    // Argument validation is the same on every build.
    chd_nn_model_t *m = nullptr;
    REQUIRE(chd_nn_model_load_from_file(nullptr, &opts, &m) == CHD_E_INVALID_ARG);
    REQUIRE(m == nullptr);

    if (chd_has_feature("coreml") != 1) {
        // Built without native CoreML (non-Apple, or with_coreml=disabled):
        // pinning the native backend must report it unavailable, not crash.
        REQUIRE(chd_nn_model_load_from_file("/nonexistent.mlpackage", &opts, &m)
                == CHD_E_NN_BACKEND_UNAVAILABLE);
        REQUIRE(m == nullptr);
        std::cout << "Native CoreML not built; load returned UNAVAILABLE as expected.\n";
        return 0;
    }

    // Native CoreML built: a path that doesn't exist must fail the load
    // cleanly (compile/load error), not crash.
    REQUIRE(chd_nn_model_load_from_file("/nonexistent.mlpackage", &opts, &m)
            == CHD_E_NN_MODEL_LOAD);
    REQUIRE(m == nullptr);

    // Happy path against a real .mlpackage when one is provided (artifacts
    // aren't committed; convert with scripts/convert_coreml.py). Skips
    // otherwise. Loaded under AUTO to also exercise extension-based dispatch.
    const char *pkg = std::getenv("CHD_TEST_COREML_MODEL");
    if (pkg == nullptr || pkg[0] == '\0') {
        std::cout << "Skipping native-CoreML load happy path (set CHD_TEST_COREML_MODEL).\n";
        return 0;
    }
    chd_nn_model_t *model = nullptr;
    REQUIRE(chd_nn_model_load_from_file(pkg, nullptr, &model) == CHD_OK);
    REQUIRE(model != nullptr);
    chd_nn_backend_t backend = CHD_NN_BACKEND_AUTO;
    REQUIRE(chd_nn_model_get_active_backend(model, &backend) == CHD_OK);
    REQUIRE(backend == CHD_NN_COREML);
    chd_nn_model_free(model);
    std::cout << "Loaded native CoreML model " << pkg << " via AUTO dispatch.\n";
    return 0;
}

int testFreeNullHandle() {
    // free(nullptr) must be a no-op (matches free() / delete semantics).
    chd_nn_model_free(nullptr);
    return 0;
}

int testGetActiveProviderNull() {
    chd_nn_backend_t b = CHD_NN_BACKEND_AUTO;
    REQUIRE(chd_nn_model_get_active_backend(nullptr, &b) == CHD_E_INVALID_ARG);
    return 0;
}

}  // namespace

int main() {
    // Bail with a pass if the build doesn't include NN; the test still
    // confirms the feature flag works.
    if (chd_has_feature("nn") != 1) {
        std::cout << "Skipping NN tests (with_nn=false).\n";
        return 0;
    }
    int rc = 0;
    rc |= testLifecycle();
    rc |= testFeatureFlag();
    rc |= testProviderAvailability();
    rc |= testSessionOptsDefault();
    rc |= testLoadNonexistentModel();
    rc |= testLoadFromMemoryInvalid();
    rc |= testLoadRealModel();
    rc |= testCoreMLLoad();
    rc |= testFreeNullHandle();
    rc |= testGetActiveProviderNull();
    // Tear down the Ort::Env explicitly so its destructor runs before C++
    // static destruction reaches ORT's internal globals. Without this, the
    // test process aborts on exit with "mutex lock failed: Invalid argument"
    // because ORT's own static destructors race with the C++ runtime's. This
    // is exactly the destruction-order hazard noted above — chd_shutdown() is
    // intentionally opt-in.
    chd_shutdown();
    if (rc == 0) std::cout << "All NN framework tests passed.\n";
    return rc;
}
