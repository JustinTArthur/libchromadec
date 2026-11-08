// SPDX-License-Identifier: GPL-3.0-or-later
//
// Plugin execution-provider registration through the public ABI, against the
// stub plugin library built beside this test (nn_plugin_stub.c). The build
// passes the stub's absolute path in CHD_TEST_PLUGIN_STUB and runs this once
// per stub mode in CHD_TEST_PLUGIN_STUB_MODE. A successful registration lasts
// for the life of the process, which is why each mode gets its own process.

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

bool lastErrorHas(const std::string &needle)
{
    const char *err = chd_last_error();
    return err != nullptr && std::strstr(err, needle.c_str()) != nullptr;
}

chd_status_t loadPinned(const std::string &model, chd_nn_backend_t backend)
{
    chd_nn_session_opts_t opts{};
    chd_nn_session_opts_default(&opts);
    opts.backend = backend;
    chd_nn_model_t *m = nullptr;
    const chd_status_t st = chd_nn_model_load_from_file(model.c_str(), &opts, &m);
    chd_nn_model_free(m);
    return st;
}

// With nothing set and no plugin beside ONNX Runtime, a pinned load says why.
int testNotFound(const std::string &model)
{
    if (chd_nn_backend_is_available(CHD_NN_ORT_MIGRAPHX) != 0) {
        std::cout << "MIGraphX is available without a plugin path; skipping the not-found case.\n";
        return 0;
    }
    REQUIRE(loadPinned(model, CHD_NN_ORT_MIGRAPHX) == CHD_E_NN_BACKEND_UNAVAILABLE);
    REQUIRE(lastErrorHas("no plugin library found"));
    return 0;
}

int testFail(const std::string &model, const std::string &stub)
{
    const std::string message = "stub plugin refused registration as MIGraphXExecutionProvider";

    REQUIRE(chd_nn_backend_set_library_path(CHD_NN_ORT_MIGRAPHX, stub.c_str()) == CHD_OK);
    REQUIRE(chd_nn_backend_is_available(CHD_NN_ORT_MIGRAPHX) == 1);

    REQUIRE(loadPinned(model, CHD_NN_ORT_MIGRAPHX) == CHD_E_NN_BACKEND_UNAVAILABLE);
    REQUIRE(lastErrorHas(message));
    REQUIRE(lastErrorHas(stub));

    // The cached failure is reported the same way, from both loaders.
    REQUIRE(loadPinned(model, CHD_NN_ORT_MIGRAPHX) == CHD_E_NN_BACKEND_UNAVAILABLE);
    REQUIRE(lastErrorHas(message));
    chd_nn_session_opts_t opts{};
    chd_nn_session_opts_default(&opts);
    opts.backend = CHD_NN_ORT_MIGRAPHX;
    std::vector<char> bytes = chd_test::readFileBytes(model);
    REQUIRE(!bytes.empty());
    chd_nn_model_t *m = nullptr;
    REQUIRE(chd_nn_model_load_from_memory(bytes.data(), bytes.size(), &opts, &m)
            == CHD_E_NN_BACKEND_UNAVAILABLE);
    REQUIRE(m == nullptr);
    REQUIRE(lastErrorHas(message));

    // The auto chain steps past the failed plugin.
    REQUIRE(chd_nn_model_load_from_file(model.c_str(), nullptr, &m) == CHD_OK);
    chd_nn_backend_t backend = CHD_NN_BACKEND_AUTO;
    REQUIRE(chd_nn_model_get_active_backend(m, &backend) == CHD_OK);
    REQUIRE(backend != CHD_NN_ORT_MIGRAPHX);
    chd_nn_model_free(m);

    // Nothing was registered, so the path may still change, and a fresh env
    // attempts the registration again.
    REQUIRE(chd_nn_backend_set_library_path(CHD_NN_ORT_MIGRAPHX, model.c_str()) == CHD_OK);
    REQUIRE(chd_nn_backend_set_library_path(CHD_NN_ORT_MIGRAPHX, stub.c_str()) == CHD_OK);
    chd_shutdown();
    REQUIRE(loadPinned(model, CHD_NN_ORT_MIGRAPHX) == CHD_E_NN_BACKEND_UNAVAILABLE);
    REQUIRE(lastErrorHas(message));
    return 0;
}

int testEmpty(const std::string &model, const std::string &stub)
{
    const std::string message = "lists no AMD GPU";

    REQUIRE(chd_nn_backend_set_library_path(CHD_NN_ORT_MIGRAPHX, stub.c_str()) == CHD_OK);
    REQUIRE(loadPinned(model, CHD_NN_ORT_MIGRAPHX) == CHD_E_NN_BACKEND_UNAVAILABLE);
    REQUIRE(lastErrorHas(message));
    REQUIRE(lastErrorHas(stub));
    REQUIRE(chd_nn_backend_is_available(CHD_NN_ORT_MIGRAPHX) == 1);

    // The registered library cannot be swapped; naming it again is fine.
    REQUIRE(chd_nn_backend_set_library_path(CHD_NN_ORT_MIGRAPHX, model.c_str())
            == CHD_E_UNSUPPORTED);
    REQUIRE(lastErrorHas(stub));
    REQUIRE(chd_nn_backend_set_library_path(CHD_NN_ORT_MIGRAPHX, stub.c_str()) == CHD_OK);

    // A fresh env registers it again from the same file.
    chd_shutdown();
    REQUIRE(chd_nn_backend_is_available(CHD_NN_ORT_MIGRAPHX) == 1);
    REQUIRE(loadPinned(model, CHD_NN_ORT_MIGRAPHX) == CHD_E_NN_BACKEND_UNAVAILABLE);
    REQUIRE(lastErrorHas(message));
    return 0;
}

}  // namespace

int main()
{
    if (chd_has_feature("onnxruntime") != 1) {
        std::cout << "Skipping plugin EP tests (ONNX Runtime backend not built).\n";
        return 0;
    }
    const char *stub = std::getenv("CHD_TEST_PLUGIN_STUB");
    const char *mode = std::getenv("CHD_TEST_PLUGIN_STUB_MODE");
    const std::string model = chd_test::syntheticFixture();
    if (stub == nullptr || mode == nullptr || model.empty()) {
        std::cerr << "CHD_TEST_PLUGIN_STUB, CHD_TEST_PLUGIN_STUB_MODE and the fixture are required\n";
        return 1;
    }

    int rc = testNotFound(model);
    if (std::strcmp(mode, "fail") == 0) {
        rc |= testFail(model, stub);
    } else if (std::strcmp(mode, "empty") == 0) {
        rc |= testEmpty(model, stub);
    } else {
        std::cerr << "unknown CHD_TEST_PLUGIN_STUB_MODE " << mode << "\n";
        rc = 1;
    }
    chd_shutdown();
    if (rc == 0) std::cout << "Plugin EP tests passed (" << mode << ").\n";
    return rc;
}
