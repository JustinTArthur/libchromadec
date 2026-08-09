// SPDX-License-Identifier: GPL-3.0-or-later
//
// Per-OS ONNX Runtime execution-provider selection.
//
// Attaching an EP to an Ort::SessionOptions is a fallible op
// (the runtime may not have the provider library available, or the provider
// may reject the model at session-create time). attachProviderChain walks a
// per-OS priority list per the ORT-family chd_nn_backend_t the caller asked for,
// attaching the first provider that succeeds and reporting back which one
// actually attached.
//
// AUTO chains:
//   Windows: TensorRT → CUDA → DirectML → CPU
//   Linux:   TensorRT → CUDA → MIGraphX → CPU
//   macOS:   CoreML → CPU
//
// Explicit (non-AUTO) requests try only that provider; if the attach fails,
// they return CHD_E_NN_BACKEND_UNAVAILABLE rather than silently falling
// back. CPU is the always-available leaf.
//
// Provider availability queries use Ort::GetAvailableProviders() rather than
// LoadLibrary probes (per vapoursynth-analog patch #5; LoadLibrary on
// Windows triggers the provider's DllMain before ORT's host pointer is set,
// crashing the provider's overridden operator new).

#ifndef CHD_NN_PROVIDER_SELECT_H
#define CHD_NN_PROVIDER_SELECT_H

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <chromadec/nn.h>

namespace chd::nn {

// ORT execution-provider preference. A subset of the public chd_nn_backend_t
// enum — only the ORT-family values (CHD_NN_ORT_*) ever reach this subsystem;
// the native backends (CHD_NN_COREML, …) are dispatched before ORT is touched.
using ProviderPreference = chd_nn_backend_t;

// Engine-cache configuration passed through to the attach helpers that
// support persistent caching of compiled graphs (TensorRT, MIGraphX).
// `dir` is an absolute, already-mkdir'd path; empty means "no caching."
// `modelPath` is the path to the model being loaded; used by MIGraphX to
// derive a per-model cache filename (the MIGraphX EP wants a specific
// file path, not a directory).
struct EngineCacheConfig {
    std::string dir;
    std::string modelPath;
};

// Returns the per-OS auto fallback chain, with the caller's preference
// (when non-AUTO) prepended at the front of the chain.
std::vector<ProviderPreference> buildAutoChain(ProviderPreference requested);

// True if `provider` is in the ORT runtime's list of available providers.
// CPU is always available; other providers depend on which provider DLLs
// are loaded at startup. Uses Ort::GetAvailableProviders().
bool providerIsAvailable(ProviderPreference provider);

// Attach a provider chain to `options`. Stops at the first successfully
// attached provider; writes that provider into `*outAttached`. Returns true
// on success.
//
// `cache` is consulted by EPs that support persistent caching (TRT,
// MIGraphX). EPs that don't ignore it. `precision` is consulted by EPs
// with a reduced-precision engine mode (TRT); the rest ignore it.
//
// The CPU provider is appended implicitly (ORT's default), so a chain that
// reaches it always succeeds — but `outAttached` is set to CHD_NN_ORT_CPU.
//
// On failure (every provider in the chain failed to attach with an error
// different from "provider not available"), `outError` carries a multi-line
// description of what was tried and why.
bool attachProviderChain(Ort::SessionOptions &options,
                         const std::vector<ProviderPreference> &chain,
                         const EngineCacheConfig &cache,
                         chd_nn_compute_precision_t precision,
                         ProviderPreference *outAttached,
                         std::string *outError);

// Per-provider attach helpers. Each returns true on success or false +
// writes `outError`. These are exposed for unit tests; production callers
// should go through attachProviderChain.
bool attachCpu(Ort::SessionOptions &options, std::string *outError);
bool attachCuda(Ort::SessionOptions &options, std::string *outError);
bool attachCoreML(Ort::SessionOptions &options, std::string *outError);
bool attachDirectML(Ort::SessionOptions &options, std::string *outError);
bool attachTensorRT(Ort::SessionOptions &options,
                    const EngineCacheConfig &cache,
                    chd_nn_compute_precision_t precision,
                    std::string *outError);
bool attachMIGraphX(Ort::SessionOptions &options,
                    const EngineCacheConfig &cache,
                    std::string *outError);

// Pin the dlopen'd libraries of the execution provider a session actually
// attached (CUDA, TensorRT, or MIGraphX — the ones ORT loads dynamically and
// dlcloses at env teardown), plus the ONNX Runtime core they are bound to,
// into the process for the rest of its lifetime. Call after a session has
// been created with that EP attached; the libraries are guaranteed loaded by
// then. No-op off Linux, for providers without a dlopen'd library, and for
// libraries this process never loaded. Idempotent per library.
void pinProviderLibraries(chd_nn_backend_t provider);

}  // namespace chd::nn

#endif  // CHD_NN_PROVIDER_SELECT_H
