// SPDX-License-Identifier: GPL-3.0-or-later
#include "registry.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "comb/comb.h"
#include "comb/ntsc_decoder.h"
#include "mono/mono_decoder.h"
#include "palcolour/pal_decoder.h"
#include "palcolour/palcolour.h"
#include "secam/secam_decoder.h"

#if defined(CHD_WITH_NN)
#include "ldzeug/ldzeug_color_cnn.h"
#include "ldzeug/ldzeug_luma_sep.h"
#endif

namespace chd::decoders::registry {

namespace {

template <typename M>
auto findOr(const M &m, const std::string &key, typename M::mapped_type fallback) {
    auto it = m.find(key);
    return it == m.end() ? fallback : it->second;
}

// The colour matrix the OutputWriter will invert. Strings were validated at
// commit; unknown values fall back to the defaults here.
chd::color::ColorConversion colorConversionFor(const OptionMaps &o) {
    const auto cdp = chd::color::parseColorDifferencePrecision(
        findOr(o.str, CHD_OPT_COLOR_DIFFERENCE_PRECISION, std::string{}))
        .value_or(chd::color::ColorDifferencePrecision::Modern);
    const auto bsp = chd::color::parseBroadcastScalingPrecision(
        findOr(o.str, CHD_OPT_BROADCAST_SCALING_PRECISION, std::string{}))
        .value_or(chd::color::BroadcastScalingPrecision::Scientific);
    return chd::color::resolveColorConversion(cdp, bsp);
}

// Common option names that apply to most decoders. The active-crop bounds
// are inclusive on both axes.
bool isCropOption(const std::string &name, OptionType type) {
    if (type != OptionType::I32) return false;
    return name == CHD_OPT_FIRST_ACTIVE_FRAME_LINE
        || name == CHD_OPT_LAST_ACTIVE_FRAME_LINE
        || name == CHD_OPT_FIRST_ACTIVE_SAMPLE
        || name == CHD_OPT_LAST_ACTIVE_SAMPLE;
}

bool isOutputOption(const std::string &name, OptionType type) {
    if (type == OptionType::I32 && (name == CHD_OPT_PADDING_MULTIPLE
                                    || name == CHD_OPT_THREAD_COUNT)) return true;
    if (type == OptionType::Str  && (name == CHD_OPT_OUTPUT_FORMAT
                                    || name == CHD_OPT_OUTPUT_CLAMP
                                    || name == CHD_OPT_COLOR_DIFFERENCE_PRECISION
                                    || name == CHD_OPT_BROADCAST_SCALING_PRECISION)) return true;
    if (type == OptionType::Bool && (name == CHD_OPT_OUTPUT_Y4M_HEADERS
                                    || name == CHD_OPT_REVERSE_FIELD_ORDER)) return true;
    return false;
}

bool isCombKind(chd_decoder_kind_t k) {
    return k == CHD_DEC_NTSC_1D || k == CHD_DEC_NTSC_2D
        || k == CHD_DEC_NTSC_3D || k == CHD_DEC_NTSC_3D_NO_ADAPT
        || k == CHD_DEC_NN_TRANSFORM3D;
}

bool isPalKind(chd_decoder_kind_t k) {
    return k == CHD_DEC_PAL_2D || k == CHD_DEC_TRANSFORM_2D || k == CHD_DEC_TRANSFORM_3D;
}

bool isLdzeugKind(chd_decoder_kind_t k) {
    return k == CHD_DEC_LDZEUG_COLOR_CNN
        || k == CHD_DEC_LDZEUG_LUMA_SEP
        || k == CHD_DEC_LDZEUG_LUMA_SEP_FRAME;
}

std::vector<double> readTransformThresholds(const std::string &path) {
    std::ifstream in(path);
    std::vector<double> out;
    if (!in.is_open()) return out;
    double v;
    while (in >> v) out.push_back(v);
    return out;
}

}  // namespace

chd_decoder_kind_t resolveAuto(chd_decoder_kind_t kind, chd::metadata::VideoSystem system) {
    if (kind != CHD_DEC_AUTO) return kind;
    switch (system) {
        case chd::metadata::NTSC:  return CHD_DEC_NTSC_2D;
        case chd::metadata::PAL:
        case chd::metadata::PAL_M: return CHD_DEC_PAL_2D;
        case chd::metadata::SECAM: return CHD_DEC_SECAM;
    }
    // Systems with no decoder family stay AUTO, which commit reports as an
    // unsupported kind.
    return CHD_DEC_AUTO;
}

bool kindUsesNn(chd_decoder_kind_t kind) {
    return kind == CHD_DEC_NN_TRANSFORM3D || isLdzeugKind(kind);
}

bool optionApplies(chd_decoder_kind_t kind, const std::string &name, OptionType type) {
    // Output + pipeline-shape options apply to every kind (including AUTO so
    // setters can land before commit picks a concrete kind).
    if (isOutputOption(name, type)) return true;
    if (isCropOption(name, type)) return true;

    const bool comb   = isCombKind(kind) || kind == CHD_DEC_AUTO;
    const bool pal    = isPalKind(kind)  || kind == CHD_DEC_AUTO;
    const bool ldzeug = isLdzeugKind(kind);
    const bool nn3d   = kind == CHD_DEC_NN_TRANSFORM3D;
    const bool secam  = kind == CHD_DEC_SECAM;

    // Comb / PAL chroma-trim options.
    if (type == OptionType::F64 && name == CHD_OPT_CHROMA_GAIN)     return comb || pal || ldzeug || secam;
    if (type == OptionType::F64 && name == CHD_OPT_CHROMA_PHASE_DEG) return comb || pal || ldzeug;
    if (type == OptionType::F64 && name == CHD_OPT_LUMA_NR_LEVEL) {
        return comb || pal || kind == CHD_DEC_MONO;
    }
    if (type == OptionType::F64 && name == CHD_OPT_CHROMA_NR_LEVEL) return comb;

    // Burst-locked demodulation: the comb decodes on the burst-locked axes,
    // the ldzeug2 kinds rotate their demodulated output onto the measured
    // phase.
    if (type == OptionType::Bool && name == CHD_OPT_PHASE_COMPENSATION) return comb || ldzeug;

    // Adaptive-3D knobs: the 3D candidate penalties they tune only run on the
    // adaptive 3D comb, so they apply there alone instead of silently
    // no-opping on the other comb kinds.
    if (type == OptionType::F64  && name == CHD_OPT_COMB_ADAPT_THRESHOLD) return kind == CHD_DEC_NTSC_3D;
    if (type == OptionType::F64  && name == CHD_OPT_COMB_CHROMA_WEIGHT)   return kind == CHD_DEC_NTSC_3D;
    if (type == OptionType::Bool && name == CHD_OPT_COMB_SHOW_MAP)        return kind == CHD_DEC_NTSC_3D;

    // SECAM line-identification options. Value validation happens at commit
    // (chroma_ident_manual is required iff chroma_ident_mode is "manual").
    if (type == OptionType::Str && name == CHD_OPT_CHROMA_IDENT_MODE)   return secam;
    if (type == OptionType::Str && name == CHD_OPT_CHROMA_IDENT_MANUAL) return secam;

    // SECAM FM click concealment; range and pairing validated at commit.
    if (type == OptionType::F64 && name == CHD_OPT_CHROMA_CLICK_NR_LEVEL)       return secam;
    if (type == OptionType::F64 && name == CHD_OPT_CHROMA_CLICK_ENV_DIP_DB)     return secam;
    if (type == OptionType::F64 && name == CHD_OPT_CHROMA_CLICK_FREQ_OVERSHOOT) return secam;

    // Transform-PAL options.
    if (type == OptionType::F64 && name == CHD_OPT_TRANSFORM_THRESHOLD)
        return kind == CHD_DEC_TRANSFORM_2D || kind == CHD_DEC_TRANSFORM_3D;
    if (type == OptionType::Str && name == CHD_OPT_TRANSFORM_THRESHOLDS_FILE)
        return kind == CHD_DEC_TRANSFORM_2D || kind == CHD_DEC_TRANSFORM_3D;

    // NN-specific options.
    if (type == OptionType::F64  && name == CHD_OPT_NN_INPUT_MAGNITUDE_SCALE) return nn3d;
    if (type == OptionType::Bool && name == CHD_OPT_NN_CHROMA_BANDPASS)
        return kind == CHD_DEC_LDZEUG_LUMA_SEP || kind == CHD_DEC_LDZEUG_LUMA_SEP_FRAME;

    return false;
}

namespace {

void fillCombConfig(chd::decoders::comb::Comb::Configuration &c, const OptionMaps &o,
                    chd_decoder_kind_t kind) {
    c.chromaGain  = findOr(o.f64, CHD_OPT_CHROMA_GAIN, c.chromaGain);
    c.chromaPhase = findOr(o.f64, CHD_OPT_CHROMA_PHASE_DEG, c.chromaPhase);
    c.yNRLevel    = findOr(o.f64, CHD_OPT_LUMA_NR_LEVEL, c.yNRLevel);
    c.cNRLevel    = findOr(o.f64, CHD_OPT_CHROMA_NR_LEVEL, c.cNRLevel);
    c.adaptThreshold = findOr(o.f64, CHD_OPT_COMB_ADAPT_THRESHOLD, c.adaptThreshold);
    c.chromaWeight   = findOr(o.f64, CHD_OPT_COMB_CHROMA_WEIGHT, c.chromaWeight);
    c.showMap    = findOr(o.boolean, CHD_OPT_COMB_SHOW_MAP, c.showMap);
    c.phaseCompensation = findOr(o.boolean, CHD_OPT_PHASE_COMPENSATION, c.phaseCompensation);

    c.colorConversion = colorConversionFor(o);

    // Dimensionality and adaptivity are determined by the decoder kind. There
    // are no separate options.
    switch (kind) {
        case CHD_DEC_NTSC_1D: c.dimensions = 1; c.adaptive = false; break;
        case CHD_DEC_NTSC_2D: c.dimensions = 2; c.adaptive = false; break;
        case CHD_DEC_NTSC_3D: c.dimensions = 3; c.adaptive = true;  break;
        case CHD_DEC_NTSC_3D_NO_ADAPT: c.dimensions = 3; c.adaptive = false; break;
        case CHD_DEC_NN_TRANSFORM3D:
            c.dimensions = 3;
            c.adaptive = false;
            c.nnTransform3D = true;
            c.nnInputMagnitudeScale =
                findOr(o.f64, CHD_OPT_NN_INPUT_MAGNITUDE_SCALE, c.nnInputMagnitudeScale);
            break;
        default: break;
    }
}

void fillPalConfig(chd::decoders::palcolour::PalColour::Configuration &c, const OptionMaps &o,
                   chd_decoder_kind_t kind) {
    c.chromaGain  = findOr(o.f64, CHD_OPT_CHROMA_GAIN, c.chromaGain);
    c.chromaPhase = findOr(o.f64, CHD_OPT_CHROMA_PHASE_DEG, c.chromaPhase);
    c.yNRLevel    = findOr(o.f64, CHD_OPT_LUMA_NR_LEVEL, c.yNRLevel);
    c.transformThreshold = findOr(o.f64, CHD_OPT_TRANSFORM_THRESHOLD, c.transformThreshold);

    const auto thresholdsFile = findOr(o.str, CHD_OPT_TRANSFORM_THRESHOLDS_FILE, std::string{});
    if (!thresholdsFile.empty()) {
        c.transformThresholds = readTransformThresholds(thresholdsFile);
    }

    c.colorConversion = colorConversionFor(o);

    switch (kind) {
        case CHD_DEC_PAL_2D:
            c.separation = chd::decoders::palcolour::PalColour::palColourFilter;
            break;
        case CHD_DEC_TRANSFORM_2D:
            c.separation = chd::decoders::palcolour::PalColour::transform2DFilter;
            break;
        case CHD_DEC_TRANSFORM_3D:
            c.separation = chd::decoders::palcolour::PalColour::transform3DFilter;
            break;
        default: break;
    }
}

}  // namespace

std::unique_ptr<chd::decoders::Decoder> build(chd_decoder_kind_t kind, const OptionMaps &opts) {
    switch (kind) {
        case CHD_DEC_MONO: {
            chd::decoders::mono::MonoDecoder::MonoConfiguration mc;
            mc.yNRLevel = findOr(opts.f64, CHD_OPT_LUMA_NR_LEVEL, mc.yNRLevel);
            return std::make_unique<chd::decoders::mono::MonoDecoder>(mc);
        }
        case CHD_DEC_NTSC_1D:
        case CHD_DEC_NTSC_2D:
        case CHD_DEC_NTSC_3D:
        case CHD_DEC_NTSC_3D_NO_ADAPT:
        case CHD_DEC_NN_TRANSFORM3D: {
            chd::decoders::comb::Comb::Configuration cc;
            fillCombConfig(cc, opts, kind);
            return std::make_unique<chd::decoders::comb::NtscDecoder>(cc);
        }
        case CHD_DEC_PAL_2D:
        case CHD_DEC_TRANSFORM_2D:
        case CHD_DEC_TRANSFORM_3D: {
            chd::decoders::palcolour::PalColour::Configuration pc;
            fillPalConfig(pc, opts, kind);
            return std::make_unique<chd::decoders::palcolour::PalDecoder>(pc);
        }
        case CHD_DEC_SECAM: {
            using IdentMode = chd::decoders::secam::SecamDecoder::IdentMode;
            chd::decoders::secam::SecamDecoder::SecamConfiguration sc;
            sc.chromaGain = findOr(opts.f64, CHD_OPT_CHROMA_GAIN, sc.chromaGain);
            // Strings were validated at commit; unknown values fall back to
            // the defaults here.
            const auto mode = findOr(opts.str, CHD_OPT_CHROMA_IDENT_MODE, std::string{});
            if (mode == "porch")        sc.identMode = IdentMode::Porch;
            else if (mode == "bottles") sc.identMode = IdentMode::Bottles;
            else if (mode == "manual")  sc.identMode = IdentMode::Manual;
            else                        sc.identMode = IdentMode::Auto;
            const auto manual = findOr(opts.str, CHD_OPT_CHROMA_IDENT_MANUAL, std::string{});
            sc.manualFirstComponent = (manual == "dr_first") ? 1 : 0;
            sc.clickNrLevel = findOr(opts.f64, CHD_OPT_CHROMA_CLICK_NR_LEVEL, sc.clickNrLevel);
            sc.clickEnvDipDbOverride =
                findOr(opts.f64, CHD_OPT_CHROMA_CLICK_ENV_DIP_DB, sc.clickEnvDipDbOverride);
            sc.clickFreqOvershootOverride = findOr(
                opts.f64, CHD_OPT_CHROMA_CLICK_FREQ_OVERSHOOT, sc.clickFreqOvershootOverride);
            sc.colorConversion = colorConversionFor(opts);
            return std::make_unique<chd::decoders::secam::SecamDecoder>(sc);
        }
#if defined(CHD_WITH_NN)
        case CHD_DEC_LDZEUG_COLOR_CNN: {
            auto d = std::make_unique<chd::decoders::ldzeug::LdzeugColorCnnDecoder>();
            d->setMode(chd::decoders::ldzeug::LdzeugDecoderBase::Mode::Field);
            d->setChromaGain (findOr(opts.f64, CHD_OPT_CHROMA_GAIN, 1.0));
            d->setChromaPhase(findOr(opts.f64, CHD_OPT_CHROMA_PHASE_DEG, 0.0));
            d->setPhaseCompensation(findOr(opts.boolean, CHD_OPT_PHASE_COMPENSATION, false));
            return d;
        }
        case CHD_DEC_LDZEUG_LUMA_SEP:
        case CHD_DEC_LDZEUG_LUMA_SEP_FRAME: {
            auto d = std::make_unique<chd::decoders::ldzeug::LdzeugLumaSepDecoder>();
            d->setMode(kind == CHD_DEC_LDZEUG_LUMA_SEP_FRAME
                           ? chd::decoders::ldzeug::LdzeugDecoderBase::Mode::Frame
                           : chd::decoders::ldzeug::LdzeugDecoderBase::Mode::Field);
            d->setChromaGain (findOr(opts.f64, CHD_OPT_CHROMA_GAIN, 1.0));
            d->setChromaPhase(findOr(opts.f64, CHD_OPT_CHROMA_PHASE_DEG, 0.0));
            d->setChromaBandpass(findOr(opts.boolean, CHD_OPT_NN_CHROMA_BANDPASS, true));
            d->setPhaseCompensation(findOr(opts.boolean, CHD_OPT_PHASE_COMPENSATION, false));
            return d;
        }
#else
        case CHD_DEC_LDZEUG_COLOR_CNN:
        case CHD_DEC_LDZEUG_LUMA_SEP:
        case CHD_DEC_LDZEUG_LUMA_SEP_FRAME:
            return nullptr;  // NN disabled at build time
#endif
        case CHD_DEC_AUTO:
        default:
            return nullptr;
    }
}

#if defined(CHD_WITH_NN)
bool applyNnModel(chd_decoder_kind_t kind,
                  chd::decoders::Decoder &decoder,
                  std::shared_ptr<chd::nn::InferenceEngine> engine) {
    if (!engine) return true;  // nothing to bind is always OK
    if (kind == CHD_DEC_NN_TRANSFORM3D) {
        auto *nd = dynamic_cast<chd::decoders::comb::NtscDecoder *>(&decoder);
        if (!nd) return false;
        nd->setNnModel(std::move(engine));
        return true;
    }
    if (kind == CHD_DEC_LDZEUG_COLOR_CNN
     || kind == CHD_DEC_LDZEUG_LUMA_SEP
     || kind == CHD_DEC_LDZEUG_LUMA_SEP_FRAME) {
        auto *ld = dynamic_cast<chd::decoders::ldzeug::LdzeugDecoderBase *>(&decoder);
        if (!ld) return false;
        ld->setNnModel(std::move(engine));
        return true;
    }
    // Other kinds don't take a session; supplying one is a caller mistake.
    return false;
}
#endif

}  // namespace chd::decoders::registry
