/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * A stand-in ONNX Runtime plugin execution-provider library for the plugin
 * registration tests. It exports the two entry points ONNX Runtime looks for
 * and behaves according to CHD_TEST_PLUGIN_STUB_MODE:
 *
 *   fail   CreateEpFactories returns an error, so registration fails.
 *   empty  CreateEpFactories returns no factories, so registration succeeds
 *          but no EpDevice ever names this EP.
 *
 * Either way it refuses a registration name other than the one MIGraphX
 * plugins expect, so the tests also prove which name the library uses.
 */

#include <onnxruntime_c_api.h>

#include <stdlib.h>
#include <string.h>

#define STUB_EXPORT __attribute__((visibility("default")))

static const char kExpectedName[] = "MIGraphXExecutionProvider";

STUB_EXPORT OrtStatus *CreateEpFactories(const char *registration_name,
                                         const OrtApiBase *api_base,
                                         const OrtLogger *logger,
                                         struct OrtEpFactory **factories,
                                         size_t max_factories,
                                         size_t *num_factories)
{
    const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
    const char *mode = getenv("CHD_TEST_PLUGIN_STUB_MODE");
    (void)logger;
    (void)factories;
    (void)max_factories;

    *num_factories = 0;
    if (registration_name == NULL || strcmp(registration_name, kExpectedName) != 0)
        return api->CreateStatus(ORT_FAIL, "stub plugin: unexpected registration name");
    if (mode != NULL && strcmp(mode, "fail") == 0)
        return api->CreateStatus(ORT_FAIL,
                                 "stub plugin refused registration as MIGraphXExecutionProvider");
    return NULL;
}

STUB_EXPORT OrtStatus *ReleaseEpFactory(struct OrtEpFactory *factory)
{
    (void)factory;
    return NULL;
}
