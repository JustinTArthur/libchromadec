// SPDX-License-Identifier: GPL-3.0-or-later
//
// Decoder registry: translate a chd_decoder_kind_t to a
// concrete chd::decoders::Decoder subclass, applying caller-supplied options
// from the C ABI option maps. Each `kind` here corresponds 1:1 to a
// CHD_DEC_* enumerator in <chromadec/decoder.h>.
//
// The registry also owns the option-name → option-type → applicable-kinds
// mapping that backs chd_decoder_has_option and the per-kind validation
// inside chd_decoder_set_option_*. Centralising the table here keeps the
// C ABI shims thin and avoids duplicating "is this option valid for this
// decoder" logic across c_api_decoder.cpp.

#ifndef CHD_DECODERS_REGISTRY_H
#define CHD_DECODERS_REGISTRY_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <chromadec/decoder.h>

#include "../metadata/core.h"

#include "decoder_base.h"

#if defined(CHD_WITH_NN)
namespace chd::nn { class InferenceEngine; }
#endif

namespace chd::decoders::registry {

enum class OptionType { F64, I32, Bool, Str };

// Caller-supplied options as stashed by the C ABI option setters. One map
// per type. Option names use the CHD_OPT_* registry strings.
struct OptionMaps {
    std::unordered_map<std::string, double>      f64;
    std::unordered_map<std::string, int32_t>     i32;
    std::unordered_map<std::string, bool>        boolean;
    std::unordered_map<std::string, std::string> str;
};

// Resolve CHD_DEC_AUTO to a concrete kind based on the input video
// system. Non-AUTO kinds pass through unchanged. Used by commit before
// the registry factory dispatch.
chd_decoder_kind_t resolveAuto(chd_decoder_kind_t kind,
                               chd::metadata::VideoSystem system);

// Return true if `name` is a valid option for `kind` with the given type.
// Backs chd_decoder_has_option and the validation arm of every
// chd_decoder_set_option_* call.
bool optionApplies(chd_decoder_kind_t kind, const std::string &name, OptionType type);

// Build the concrete decoder for `kind`, with its Configuration populated
// from `opts`. Returns nullptr if `kind` is unknown (caller should map to
// CHD_E_DECODER_UNKNOWN). Decoder::configure() is NOT called here — the
// caller still drives that, since it needs final VideoParameters.
std::unique_ptr<chd::decoders::Decoder> build(chd_decoder_kind_t kind,
                                              const OptionMaps &opts);

#if defined(CHD_WITH_NN)
// Bind the NN session to the decoder if the kind supports one. Returns
// true if the decoder accepted the session (or no session was needed),
// false if the kind doesn't support NN but a session was supplied.
bool applyNnModel(chd_decoder_kind_t kind,
                  chd::decoders::Decoder &decoder,
                  std::shared_ptr<chd::nn::InferenceEngine> engine);
#endif

// True if `kind` is one of the NN-driven kinds (nnTransform3D, ldzeug2).
// Used by the option setters to gate NN-only option names.
bool kindUsesNn(chd_decoder_kind_t kind);

}  // namespace chd::decoders::registry

#endif  // CHD_DECODERS_REGISTRY_H
