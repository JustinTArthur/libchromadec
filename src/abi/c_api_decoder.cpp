// SPDX-License-Identifier: GPL-3.0-or-later
//
// chd_decoder lifecycle (set_option/set_nn_model/commit) +
// chd_decode_frame (sync, random-access) + chd_decode_frames_async.
//
// The handle stores caller options uncommitted until chd_decoder_commit,
// at which point the registry builds the concrete IDecoder, the
// OutputWriter is configured, and the decoder is configured against the
// committed active crop. From that point chd_decode_frame is
// callable until chd_decoder_free.

#include <chromadec/decoder.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../common/error_state.h"
#include "../decoders/registry.h"
#include "../decoders/source_field.h"
#include "../dropout/dropout_corrector.h"
#include "../metadata/core.h"
#include "../output/component_frame.h"
#include "../reader/source.h"
#include "handles.h"

namespace {

chd_status_t set_arg_error(const char *what, const char *detail = nullptr) {
    std::string msg = std::string("chromadec: ") + what + ": null argument";
    if (detail != nullptr) msg += " (" + std::string(detail) + ")";
    chd::detail::set_last_error(msg);
    return CHD_E_INVALID_ARG;
}

// Translate the string form of the output_format option to an
// OutputWriter::PixelFormat. Returns -1 on unknown name. The float-output
// formats ("yuv444ps", "grays", "rgbs") bypass OutputWriter::convert entirely
// (we render direct from the ComponentFrame via OutputWriter::convertToFloat
// / convertToFloatRGB); OutputWriter still needs a valid integer format to
// size headers / padding, so each maps to the integer format with the
// matching plane layout.
int parseOutputFormat(const std::string &name) {
    if (name == "yuv444p16" || name == "YUV444P16") return chd::output::OutputWriter::YUV444P16;
    if (name == "rgb48"     || name == "RGB48")     return chd::output::OutputWriter::RGB48;
    if (name == "gray16"    || name == "GRAY16")    return chd::output::OutputWriter::GRAY16;
    if (name == "yuv444ps"  || name == "YUV444PS")  return chd::output::OutputWriter::YUV444P16;
    if (name == "rgbs"      || name == "RGBS")      return chd::output::OutputWriter::RGB48;
    if (name == "grays"     || name == "GRAYS")     return chd::output::OutputWriter::GRAY16;
    if (name == "yuv440p16" || name == "YUV440P16") return chd::output::OutputWriter::YUV440P16;
    if (name == "yuv440ps"  || name == "YUV440PS")  return chd::output::OutputWriter::YUV440P16;
    return -1;
}

chd_pixel_format_t parseChdPixelFormat(const std::string &name) {
    if (name == "yuv444p16" || name == "YUV444P16") return CHD_PIXEL_YUV444P16;
    if (name == "rgb48"     || name == "RGB48")     return CHD_PIXEL_RGB48;
    if (name == "gray16"    || name == "GRAY16")    return CHD_PIXEL_GRAY16;
    if (name == "yuv444ps"  || name == "YUV444PS")  return CHD_PIXEL_YUV444PS;
    if (name == "rgbs"      || name == "RGBS")      return CHD_PIXEL_RGBS;
    if (name == "grays"     || name == "GRAYS")     return CHD_PIXEL_GRAYS;
    if (name == "yuv440p16" || name == "YUV440P16") return CHD_PIXEL_YUV440P16;
    if (name == "yuv440ps"  || name == "YUV440PS")  return CHD_PIXEL_YUV440PS;
    return CHD_PIXEL_YUV444P16;
}

// Translate the string form of the output_clamp option to an
// OutputWriter::ClampMode. Returns -1 on unknown name.
int parseClampMode(const std::string &name) {
    if (name == "none" || name == "NONE")
        return chd::output::OutputWriter::CLAMP_NONE;
    if (name == "legal_rgb_sdr" || name == "LEGAL_RGB_SDR")
        return chd::output::OutputWriter::CLAMP_LEGAL_RGB_SDR;
    if (name == "legal_rgb_hdr" || name == "LEGAL_RGB_HDR")
        return chd::output::OutputWriter::CLAMP_LEGAL_RGB_HDR;
    if (name == "legal_ycbcr_bt601" || name == "LEGAL_YCBCR_BT601")
        return chd::output::OutputWriter::CLAMP_LEGAL_YCBCR_BT601;
    return -1;
}

// Apply caller-supplied active-crop overrides to the VideoParameters before the
// decoder configures itself. Mirrors LineParameters::applyTo with the i32
// option-map as the source. The options are inclusive on both axes; the sample
// crop is half-open inside VideoParameters, so the last sample converts here.
void applyCropOverrides(chd::metadata::LdDecodeMetaData::VideoParameters &vp,
                        const chd::decoders::registry::OptionMaps &o) {
    // Whether the caller set a bound is the option's presence in the map, not a
    // sentinel value, so an out-of-range bound reaches validateCrop and is
    // reported rather than silently ignored.
    auto apply = [&](const char *k, int32_t &dst, int32_t bias = 0) {
        auto it = o.i32.find(k);
        if (it != o.i32.end()) dst = it->second + bias;
    };
    apply(CHD_OPT_FIRST_ACTIVE_FRAME_LINE, vp.firstActiveFrameLine);
    apply(CHD_OPT_LAST_ACTIVE_FRAME_LINE,  vp.lastActiveFrameLine);
    apply(CHD_OPT_FIRST_ACTIVE_SAMPLE,     vp.activeVideoStart);
    // The sample crop is inclusive on the ABI and half-open in VideoParameters.
    apply(CHD_OPT_LAST_ACTIVE_SAMPLE,      vp.activeVideoEnd, 1);
}

// Reject a crop that leaves the stored row, or that selects nothing. The
// widened-crop escape hatch is the only way to reach samples outside the
// default active window, so it is also the only place the bound can go bad.
chd_status_t validateCrop(const chd::metadata::LdDecodeMetaData::VideoParameters &vp) {
    if (vp.activeVideoStart < 0 || vp.activeVideoEnd > vp.fieldWidth
        || vp.activeVideoEnd <= vp.activeVideoStart) {
        chd::detail::set_last_error(
            "chd_decoder_commit: active sample crop must satisfy 0 <= first_active_sample "
            "<= last_active_sample < field_width");
        return CHD_E_INVALID_ARG;
    }
    // ComponentFrame stores (fieldHeight * 2) - 1 rows: the trailing line of the
    // second field is padding and never allocated, so the last valid frame line
    // is (fieldHeight * 2) - 2. Admitting the row above it would let the output
    // and decode paths dereference one row past the buffer.
    if (vp.firstActiveFrameLine < 0
        || vp.lastActiveFrameLine >= (vp.fieldHeight * 2) - 1
        || vp.lastActiveFrameLine < vp.firstActiveFrameLine) {
        chd::detail::set_last_error(
            "chd_decoder_commit: active frame-line crop must satisfy 0 <= "
            "first_active_frame_line <= last_active_frame_line < (2 * field_height) - 1");
        return CHD_E_INVALID_ARG;
    }
    return CHD_OK;
}

// The destination map is passed as a pointer-to-member so it is selected
// from d->optionMaps only after the null check below; naming the map at the
// call site (d->optionMaps.f64) would dereference d before it is validated.
template <typename T>
chd_status_t setOpt(chd_decoder_t *d, const char *name,
                    chd::decoders::registry::OptionType type, T value,
                    std::unordered_map<std::string, T> chd::decoders::registry::OptionMaps::*dst) {
    if (d == nullptr || name == nullptr) return set_arg_error("chd_decoder_set_option");
    if (d->committed) {
        chd::detail::set_last_error("chd_decoder_set_option: decoder already committed");
        return CHD_E_INVALID_ARG;
    }
    if (!chd::decoders::registry::optionApplies(d->kind, name, type)) {
        chd::detail::set_last_error(std::string("chd_decoder_set_option: option \"")
                                    + name + "\" not valid for this decoder kind");
        return CHD_E_INVALID_ARG;
    }
    (d->optionMaps.*dst)[name] = std::move(value);
    return CHD_OK;
}

// Decode one source's frame into `outCF`, applying dropout correction using
// that source's own metadata and extra sources. Used once for a plain
// composite decode, and twice (luma + chroma) for a decode-level Y/C merge.
// The passed `decoder` is the caller's worker instance; the caller holds its
// mutex for the whole call (Decoder subclasses keep non-reentrant scratch).
chd_status_t decodeSourceFrame(
    chd_decoder_t *d,
    chd::reader::ISource &source, chd::metadata::LdDecodeMetaData &meta,
    chd::decoders::Decoder &decoder, int32_t lookBehind, int32_t lookAhead,
    std::vector<chd_video_source> &extraSources,
    std::unique_ptr<chd::dropout::MultiSourceAlignment> &alignment,
    std::once_flag &alignmentOnce,
    int64_t frame_index, chd::output::ComponentFrame &outCF,
    chd::dropout::DropoutCorrectionStats &stats) {
    const int32_t numFrames = meta.getNumberOfFrames();
    if (frame_index < 0 || frame_index >= numFrames) {
        chd::detail::set_last_error("chd_decode_frame: frame_index out of range");
        return CHD_E_OUT_OF_RANGE;
    }

    // Load the field window (lookbehind + this frame + lookahead). frame_index
    // is 0-based at the C ABI; SourceField is 1-based internally.
    std::vector<chd::decoders::SourceField> fields;
    int32_t startIndex = 0;
    int32_t endIndex   = 0;
    const int32_t firstFrameNumber = static_cast<int32_t>(frame_index) + 1;
    chd::decoders::SourceField::loadFields(
        source, meta, firstFrameNumber, /*numFrames=*/1, lookBehind, lookAhead,
        fields, startIndex, endIndex);

    // Apply dropout correction if requested. Single-source if no extras
    // are attached; otherwise build a vector<ExtraSourceFrame> from each
    // chd_video_source and use the multi-source DropoutCorrector overload.
    if (d->dropoutOptsSet && d->dropoutOpts.enabled != 0) {
        chd::dropout::DropoutCorrector corrector(d->videoParameters);

        // Build the multi-source VBI alignment once. Its constructor scans
        // every source's field VBI, so it is cached on the decoder and shared
        // (read-only) across workers.
        if (!extraSources.empty()) {
            std::call_once(alignmentOnce, [&] {
                std::vector<chd::metadata::LdDecodeMetaData *> sources;
                sources.reserve(extraSources.size() + 1);
                sources.push_back(&meta);  // this plane's capture is source 0
                for (auto &ex : extraSources) sources.push_back(ex.metadata.get());
                alignment =
                    std::make_unique<chd::dropout::MultiSourceAlignment>(std::move(sources));
            });
        }

        std::vector<chd::dropout::ExtraSourceFrame> extras;
        extras.reserve(extraSources.size());
        for (size_t i = 0; i < extraSources.size(); ++i) {
            auto &ex = extraSources[i];
            if (ex.metadata == nullptr || ex.source == nullptr) continue;
            const int32_t exFrames = ex.metadata->getNumberOfFrames();

            // Map the primary frame to the matching disc frame in this source
            // (by VBI CAV/CLV number where available, positionally otherwise).
            const int32_t exFrame = alignment->sourceFrameForPrimaryFrame(
                firstFrameNumber, static_cast<int32_t>(i) + 1);
            if (exFrame < 1 || exFrame > exFrames) {
                // This source does not cover the primary's frame — skip it for
                // this frame rather than padding with black.
                continue;
            }
            const int32_t exFirst  = ex.metadata->getFirstFieldNumber(exFrame);
            const int32_t exSecond = ex.metadata->getSecondFieldNumber(exFrame);

            chd::dropout::ExtraSourceFrame esf;
            esf.firstFieldData  = ex.source->getVideoField(exFirst);
            esf.secondFieldData = ex.source->getVideoField(exSecond);
            esf.firstFieldMeta  = ex.metadata->getField(exFirst);
            esf.secondFieldMeta = ex.metadata->getField(exSecond);
            esf.videoParams     = ex.metadata->getVideoParameters();
            // Quality: VITS bPSNR average; synthesized CVBS metadata leaves
            // VitsMetrics inUse=false / bPSNR=0, which is fine — the
            // corrector still treats it as a usable extra, just without a
            // quality-based tiebreaker.
            esf.quality = (esf.firstFieldMeta.vitsMetrics.bPSNR
                         + esf.secondFieldMeta.vitsMetrics.bPSNR) / 2.0;
            extras.push_back(std::move(esf));
        }

        if (extras.empty()) {
            corrector.correctFrame(fields[startIndex], fields[startIndex + 1],
                                   d->dropoutOpts.overcorrect != 0,
                                   d->dropoutOpts.intra_field_only != 0,
                                   &stats);
        } else {
            corrector.correctFrame(fields[startIndex], fields[startIndex + 1],
                                   extras,
                                   d->dropoutOpts.overcorrect != 0,
                                   d->dropoutOpts.intra_field_only != 0,
                                   &stats);
        }
    }

    std::vector<chd::output::ComponentFrame> componentFrames(1);
    try {
        decoder.decodeFrames(fields, startIndex, endIndex, componentFrames);
    } catch (const std::exception &e) {
        chd::detail::set_last_error(std::string("chd_decode_frame: decoder threw: ") + e.what());
        return CHD_E_INTERNAL;
    }
    outCF = std::move(componentFrames[0]);
    return CHD_OK;
}

chd_status_t decodeFrameLocked(chd_decoder_t *d, size_t workerIdx,
                               int64_t frame_index, chd_frame_t **out) {
    chd::metadata::LdDecodeMetaData *meta = d->video->metadata.get();
    if (meta == nullptr) {
        chd::detail::set_last_error("chd_decode_frame: missing metadata");
        return CHD_E_METADATA_MISSING;
    }

    // Decode the primary source (the whole composite, or the luma plane of a
    // Y/C merge). Its dropout stats are the ones published below.
    chd::output::ComponentFrame primaryCF;
    chd::dropout::DropoutCorrectionStats stats;
    if (const chd_status_t rc = decodeSourceFrame(
            d, *d->video->source, *meta, *d->decoders[workerIdx],
            d->lookBehind, d->lookAhead, d->video->extraSources,
            d->multiSourceAlignment, d->multiSourceAlignmentOnce,
            frame_index, primaryCF, stats);
        rc != CHD_OK) {
        return rc;
    }

    // Decode-level Y/C merge: decode the chroma plane with the colour kind and
    // graft its U/V onto the Mono-decoded luma frame.
    if (d->video->chromaSource != nullptr) {
        chd::output::ComponentFrame chromaCF;
        chd::dropout::DropoutCorrectionStats chromaStats;
        if (const chd_status_t rc = decodeSourceFrame(
                d, *d->video->chromaSource, *d->video->chromaMetadata,
                *d->chromaDecoders[workerIdx], d->chromaLookBehind, d->chromaLookAhead,
                d->video->chromaExtraSources,
                d->chromaMultiSourceAlignment, d->chromaMultiSourceAlignmentOnce,
                frame_index, chromaCF, chromaStats);
            rc != CHD_OK) {
            return rc;
        }
        primaryCF.setU(*chromaCF.getU());
        primaryCF.setV(*chromaCF.getV());
        // Line-sequential (SECAM) chroma carries its row lattice, ident
        // report, and concealment results with the chroma plane; empty for
        // QAM decodes.
        primaryCF.chromaRowComponents = std::move(chromaCF.chromaRowComponents);
        primaryCF.chromaIdent = chromaCF.chromaIdent;
        primaryCF.chromaConcealedSpans = std::move(chromaCF.chromaConcealedSpans);
        primaryCF.chromaClick = chromaCF.chromaClick;
    }

    auto frame = std::make_unique<chd_frame>();
    frame->format = d->outputPixelFormat;

    // Emitted frame: the active crop inside whatever black border padding added.
    const int32_t activeWidth  = d->outputWriter.getActiveWidth();
    const int32_t outputWidth  = d->outputWriter.getOutputWidth();
    const int32_t leftPad      = d->outputWriter.getLeftPadSamples();
    const int32_t topPad       = d->outputWriter.getTopPadLines();

    // Publish click-concealment results: spans mapped into the committed
    // active-output framing (cached per frame for the unified dropout-span
    // query) and the effective thresholds actually applied.
    if (primaryCF.chromaClick.valid) {
        std::vector<chd_dropout_span_t> concealed;
        concealed.reserve(primaryCF.chromaConcealedSpans.size());
        const int32_t firstLine = d->videoParameters.firstActiveFrameLine;
        const int32_t lastLine = d->videoParameters.lastActiveFrameLine;
        const int32_t avs = d->videoParameters.activeVideoStart;
        for (const auto &span : primaryCF.chromaConcealedSpans) {
            if (span.frameRow < firstLine || span.frameRow > lastLine) continue;
            const int32_t xs = std::clamp(span.xStart - avs, 0, activeWidth) + leftPad;
            const int32_t xe = std::clamp(span.xEnd - avs, 0, activeWidth) + leftPad;
            if (xe <= xs) continue;
            concealed.push_back({span.frameRow - firstLine + topPad, xs, xe,
                                 CHD_DROPOUT_ORIGIN_DECODER_CONCEALMENT});
        }
        std::lock_guard<std::mutex> sl(d->statsMutex);
        // Bound the cache; evicting the lowest frame index keeps streaming
        // (ascending) decodes holding their recent frames.
        if (d->concealedSpansByFrame.size() > 4096) {
            d->concealedSpansByFrame.erase(d->concealedSpansByFrame.begin());
        }
        d->concealedSpansByFrame[frame_index] = std::move(concealed);
        d->lastClickEnvDipDb = primaryCF.chromaClick.envDipDb;
        d->lastClickFreqOvershoot = primaryCF.chromaClick.freqOvershoot;
        d->haveClickThresholds = true;
    }

    if (frame->format == CHD_PIXEL_YUV440PS || frame->format == CHD_PIXEL_YUV440P16) {
        // 4:4:0: full-height Y, chroma planes holding only the rows the
        // frame's component lattice assigns to each of Db/Dr, woven by row
        // parity like the luma plane. Geometry is per-frame; the frame
        // carries it for chd_frame_get_plane_info.
        const int32_t outputHeight = d->outputWriter.getOutputHeight();
        try {
            frame->chroma440 = (frame->format == CHD_PIXEL_YUV440PS)
                ? d->outputWriter.convertToFloat440(primaryCF, frame->floatPlane)
                : d->outputWriter.convert440(primaryCF, frame->u16Plane);
        } catch (const std::exception &e) {
            chd::detail::set_last_error(
                std::string("chd_decode_frame: 4:4:0 conversion failed: ") + e.what());
            return CHD_E_INTERNAL;
        }
        // Border rows carry no chroma and stay -1; the active rows sit below
        // whatever top border padding added.
        frame->rowComponent.assign(static_cast<size_t>(outputHeight), -1);
        const int32_t firstLine = d->videoParameters.firstActiveFrameLine;
        const int32_t activeHeight = d->outputWriter.getActiveHeight();
        for (int32_t y = 0; y < activeHeight; y++) {
            frame->rowComponent[static_cast<size_t>(topPad + y)] =
                primaryCF.chromaRowComponents[static_cast<size_t>(firstLine + y)];
        }
        if (primaryCF.chromaIdent.valid) {
            frame->identReport.mechanism = static_cast<chd_chroma_ident_mechanism_t>(
                primaryCF.chromaIdent.mechanism);
            frame->identReport.confidence = primaryCF.chromaIdent.confidence;
            frame->identReport.field_confidence[0] = primaryCF.chromaIdent.fieldConfidence[0];
            frame->identReport.field_confidence[1] = primaryCF.chromaIdent.fieldConfidence[1];
            frame->identReport.first_row_component =
                (frame->rowComponent[static_cast<size_t>(topPad)] == 1)
                    ? CHD_CHROMA_ROW_DR : CHD_CHROMA_ROW_DB;
            frame->hasIdentReport = true;
        }
        frame->outputWidth = outputWidth;
        frame->outputHeight = outputHeight;
        frame->info.format = frame->format;
        frame->info.width  = outputWidth;
        frame->info.height = outputHeight;
        frame->info.num_planes = 3;
    } else if (frame->format == CHD_PIXEL_YUV444PS || frame->format == CHD_PIXEL_GRAYS) {
        // Honor any requested padding: the float path mirrors the integer path's
        // committed geometry (top/bottom blank lines), it does not crop tight.
        // GRAYS emits only the E'Y plane; YUV444PS also emits E'Cb/E'Cr.
        const bool includeChroma = (frame->format == CHD_PIXEL_YUV444PS);
        const int32_t outputHeight = d->outputWriter.getOutputHeight();
        try {
            d->outputWriter.convertToFloat(primaryCF, frame->floatPlane, includeChroma);
        } catch (const std::exception &e) {
            chd::detail::set_last_error(
                std::string("chd_decode_frame: float conversion failed: ") + e.what());
            return CHD_E_INTERNAL;
        }
        frame->outputWidth = outputWidth;
        frame->outputHeight = outputHeight;
        frame->info.format = frame->format;
        frame->info.width  = outputWidth;
        frame->info.height = outputHeight;
        frame->info.num_planes = includeChroma ? 3 : 1;
    } else if (frame->format == CHD_PIXEL_RGBS) {
        // Direct ComponentFrame → R'G'B' float planes; no Y'CbCr integer
        // intermediate. Same geometry as the integer convert() path.
        const int32_t outputHeight = d->outputWriter.getOutputHeight();
        try {
            d->outputWriter.convertToFloatRGB(primaryCF, frame->floatPlane);
        } catch (const std::exception &e) {
            chd::detail::set_last_error(
                std::string("chd_decode_frame: RGB float conversion failed: ") + e.what());
            return CHD_E_INTERNAL;
        }
        frame->outputWidth = outputWidth;
        frame->outputHeight = outputHeight;
        frame->info.format = frame->format;
        frame->info.width  = outputWidth;
        frame->info.height = outputHeight;
        frame->info.num_planes = 3;
    } else {
        try {
            d->outputWriter.convert(primaryCF, frame->u16Plane);
        } catch (const std::exception &e) {
            chd::detail::set_last_error(
                std::string("chd_decode_frame: conversion failed: ") + e.what());
            return CHD_E_INTERNAL;
        }
        const int32_t planes =
            (d->outputConfig.pixelFormat == chd::output::OutputWriter::GRAY16) ? 1 : 3;
        const int32_t outputHeight = static_cast<int32_t>(
            frame->u16Plane.size() / static_cast<size_t>(outputWidth) / planes);
        frame->outputWidth  = outputWidth;
        frame->outputHeight = outputHeight;
        frame->info.format  = d->outputPixelFormat;
        frame->info.width   = outputWidth;
        frame->info.height  = outputHeight;
        frame->info.num_planes = planes;
    }
    frame->info.frame_index = frame_index;

    {
        std::lock_guard<std::mutex> sl(d->statsMutex);
        d->lastDropoutStats.corrected      = stats.corrected;
        d->lastDropoutStats.failed         = stats.failed;
        d->lastDropoutStats.total_distance = stats.totalDistance;
    }

    *out = frame.release();
    return CHD_OK;
}

}  // namespace

extern "C" {

// ── Lifecycle ────────────────────────────────────────────────────────────────

chd_status_t chd_decoder_create(chd_video_t *v, chd_decoder_kind_t kind, chd_decoder_t **out) {
    if (v == nullptr || out == nullptr) return set_arg_error("chd_decoder_create");
    auto handle = std::make_unique<chd_decoder>();
    handle->video = v;
    handle->kind  = kind;
    *out = handle.release();
    return CHD_OK;
}

void chd_decoder_free(chd_decoder_t *d) { delete d; }

// ── Options ──────────────────────────────────────────────────────────────────

chd_status_t chd_decoder_set_option_f64(chd_decoder_t *d, const char *name, double v) {
    return setOpt(d, name, chd::decoders::registry::OptionType::F64, v,
                  &chd::decoders::registry::OptionMaps::f64);
}

chd_status_t chd_decoder_set_option_i32(chd_decoder_t *d, const char *name, int32_t v) {
    return setOpt(d, name, chd::decoders::registry::OptionType::I32, v,
                  &chd::decoders::registry::OptionMaps::i32);
}

chd_status_t chd_decoder_set_option_bool(chd_decoder_t *d, const char *name, int v) {
    return setOpt<bool>(d, name, chd::decoders::registry::OptionType::Bool, v != 0,
                        &chd::decoders::registry::OptionMaps::boolean);
}

chd_status_t chd_decoder_set_option_str(chd_decoder_t *d, const char *name, const char *v) {
    if (v == nullptr) return set_arg_error("chd_decoder_set_option_str", "value");
    return setOpt<std::string>(d, name, chd::decoders::registry::OptionType::Str, v,
                               &chd::decoders::registry::OptionMaps::str);
}

chd_status_t chd_decoder_has_option(const chd_decoder_t *d, const char *name) {
    if (d == nullptr || name == nullptr) return set_arg_error("chd_decoder_has_option");
    using OT = chd::decoders::registry::OptionType;
    const bool any = chd::decoders::registry::optionApplies(d->kind, name, OT::F64)
                  || chd::decoders::registry::optionApplies(d->kind, name, OT::I32)
                  || chd::decoders::registry::optionApplies(d->kind, name, OT::Bool)
                  || chd::decoders::registry::optionApplies(d->kind, name, OT::Str);
    return any ? CHD_OK : CHD_E_INVALID_ARG;
}

chd_status_t chd_decoder_set_nn_model(chd_decoder_t *d, chd_nn_model_t *m) {
    if (d == nullptr) return set_arg_error("chd_decoder_set_nn_model");
    if (d->committed) {
        chd::detail::set_last_error("chd_decoder_set_nn_model: decoder already committed");
        return CHD_E_INVALID_ARG;
    }
#if defined(CHD_WITH_NN)
    if (!chd::decoders::registry::kindUsesNn(d->kind) && d->kind != CHD_DEC_AUTO) {
        chd::detail::set_last_error(
            "chd_decoder_set_nn_model: decoder kind does not accept an NN model");
        return CHD_E_INVALID_ARG;
    }
    d->nnModelPending = (m != nullptr) ? m->engine : nullptr;
    return CHD_OK;
#else
    (void)m;
    chd::detail::set_last_error("chd_decoder_set_nn_model: library built without NN support");
    return CHD_E_INTERNAL;
#endif
}

// ── Commit ───────────────────────────────────────────────────────────────────

chd_status_t chd_decoder_commit(chd_decoder_t *d) {
    if (d == nullptr) return set_arg_error("chd_decoder_commit");
    if (d->committed) return CHD_OK;  // idempotent: re-commit is a no-op
    // Caller contract: chd_decoder_commit + chd_decoder_set_option_* +
    // chd_decode_frame must not be called concurrently with each other
    // (the public header documents this on chd_decoder_set_option_*).
    // No lifecycle mutex needed.

    if (d->video == nullptr || d->video->source == nullptr) {
        chd::detail::set_last_error("chd_decoder_commit: video has no source");
        return CHD_E_INVALID_ARG;
    }

    const chd::metadata::VideoSystem system = d->video->source->parameters().system;
    const chd_decoder_kind_t resolved = chd::decoders::registry::resolveAuto(d->kind, system);

    chd::output::OutputWriter::Configuration outCfg;
    // Default to no padding (1 = output the active crop as-is). Consumers that
    // need codec-friendly dimensions can opt in via CHD_OPT_PADDING_MULTIPLE.
    outCfg.paddingAmount = 1;
    {
        auto it = d->optionMaps.i32.find(CHD_OPT_PADDING_MULTIPLE);
        if (it != d->optionMaps.i32.end()) outCfg.paddingAmount = it->second;
        // The border grows the frame by up to paddingAmount on each axis, and
        // the frame is sized in int32, so an unbounded multiple overflows it.
        // The cap sits far above any codec or model alignment.
        constexpr int32_t kMaxPaddingMultiple = 1024;
        if (outCfg.paddingAmount < 1 || outCfg.paddingAmount > kMaxPaddingMultiple) {
            chd::detail::set_last_error(
                "chd_decoder_commit: padding_multiple must be between 1 (no padding) and 1024");
            return CHD_E_INVALID_ARG;
        }
    }
    outCfg.outputY4m = false;
    {
        auto it = d->optionMaps.boolean.find(CHD_OPT_OUTPUT_Y4M_HEADERS);
        if (it != d->optionMaps.boolean.end()) outCfg.outputY4m = it->second;
    }

    chd_pixel_format_t abiFormat = CHD_PIXEL_YUV444P16;
    {
        auto it = d->optionMaps.str.find(CHD_OPT_OUTPUT_FORMAT);
        if (it != d->optionMaps.str.end()) {
            const int pf = parseOutputFormat(it->second);
            if (pf < 0) {
                chd::detail::set_last_error(
                    "chd_decoder_commit: unknown output_format \"" + it->second + "\"");
                return CHD_E_INVALID_ARG;
            }
            outCfg.pixelFormat = static_cast<chd::output::OutputWriter::PixelFormat>(pf);
            abiFormat = parseChdPixelFormat(it->second);
        } else {
            outCfg.pixelFormat = chd::output::OutputWriter::YUV444P16;
            abiFormat = CHD_PIXEL_YUV444P16;
        }
    }
    {
        auto it = d->optionMaps.str.find(CHD_OPT_OUTPUT_CLAMP);
        if (it != d->optionMaps.str.end()) {
            const int cm = parseClampMode(it->second);
            if (cm < 0) {
                chd::detail::set_last_error(
                    "chd_decoder_commit: unknown output_clamp \"" + it->second + "\"");
                return CHD_E_INVALID_ARG;
            }
            outCfg.clampMode = static_cast<chd::output::OutputWriter::ClampMode>(cm);
        }
    }
    {
        auto it = d->optionMaps.str.find(CHD_OPT_COLOR_DIFFERENCE_PRECISION);
        if (it != d->optionMaps.str.end()) {
            const auto p = chd::color::parseColorDifferencePrecision(it->second);
            if (!p) {
                chd::detail::set_last_error(
                    "chd_decoder_commit: unknown color_difference_precision \""
                    + it->second + "\"");
                return CHD_E_INVALID_ARG;
            }
            outCfg.colorDifferencePrecision = *p;
        }
    }
    {
        auto it = d->optionMaps.str.find(CHD_OPT_BROADCAST_SCALING_PRECISION);
        if (it != d->optionMaps.str.end()) {
            const auto p = chd::color::parseBroadcastScalingPrecision(it->second);
            if (!p) {
                chd::detail::set_last_error(
                    "chd_decoder_commit: unknown broadcast_scaling_precision \""
                    + it->second + "\"");
                return CHD_E_INVALID_ARG;
            }
            outCfg.broadcastScalingPrecision = *p;
        }
    }
    {
        // SECAM's line-sequential chroma decodes to 4:4:0 (or luma-only).
        // Emitting full-height chroma or matrixed RGB would force the library
        // to pick a vertical chroma reconstruction kernel; that decision
        // belongs to the consuming application, so those formats are
        // rejected. Conversely, 4:4:0 exists only as the SECAM lattice.
        const bool is440 = (abiFormat == CHD_PIXEL_YUV440P16
                            || abiFormat == CHD_PIXEL_YUV440PS);
        const bool lumaOnly = (abiFormat == CHD_PIXEL_GRAY16
                               || abiFormat == CHD_PIXEL_GRAYS);
        if (system == chd::metadata::SECAM && !is440 && !lumaOnly) {
            chd::detail::set_last_error(
                "chd_decoder_commit: SECAM sources decode to 4:4:0 chroma "
                "(\"yuv440ps\"/\"yuv440p16\") or luma-only (\"grays\"/\"gray16\"); "
                "vertical chroma reconstruction is the consumer's choice");
            return CHD_E_INVALID_ARG;
        }
        if (is440 && system != chd::metadata::SECAM) {
            chd::detail::set_last_error(
                "chd_decoder_commit: 4:4:0 output is produced only by "
                "line-sequential (SECAM) decodes");
            return CHD_E_INVALID_ARG;
        }
        if (is440 && outCfg.outputY4m) {
            chd::detail::set_last_error(
                "chd_decoder_commit: output_y4m_headers does not support 4:4:0 output");
            return CHD_E_INVALID_ARG;
        }
    }

    {
        // SECAM line-identification options: validate the enum strings and
        // the manual-mode cross-option requirement (optionApplies is
        // per-kind only, so the pairing is checked here).
        auto modeIt = d->optionMaps.str.find(CHD_OPT_CHROMA_IDENT_MODE);
        auto manualIt = d->optionMaps.str.find(CHD_OPT_CHROMA_IDENT_MANUAL);
        std::string mode;
        if (modeIt != d->optionMaps.str.end()) {
            mode = modeIt->second;
            if (mode != "auto" && mode != "porch" && mode != "bottles" && mode != "manual") {
                chd::detail::set_last_error(
                    "chd_decoder_commit: unknown chroma_ident_mode \"" + mode + "\"");
                return CHD_E_INVALID_ARG;
            }
        }
        if (manualIt != d->optionMaps.str.end()) {
            if (manualIt->second != "db_first" && manualIt->second != "dr_first") {
                chd::detail::set_last_error(
                    "chd_decoder_commit: unknown chroma_ident_manual \""
                    + manualIt->second + "\"");
                return CHD_E_INVALID_ARG;
            }
            if (mode != "manual") {
                chd::detail::set_last_error(
                    "chd_decoder_commit: chroma_ident_manual is only meaningful with "
                    "chroma_ident_mode=\"manual\"");
                return CHD_E_INVALID_ARG;
            }
        } else if (mode == "manual") {
            chd::detail::set_last_error(
                "chd_decoder_commit: chroma_ident_mode=\"manual\" requires "
                "chroma_ident_manual (\"db_first\" or \"dr_first\")");
            return CHD_E_INVALID_ARG;
        }
    }
    {
        // SECAM click-concealment options: level range plus the rule that
        // the expert absolute overrides only mean something when the stage
        // is enabled at all.
        auto lvlIt = d->optionMaps.f64.find(CHD_OPT_CHROMA_CLICK_NR_LEVEL);
        const double level = (lvlIt != d->optionMaps.f64.end()) ? lvlIt->second : 1.0;
        if (lvlIt != d->optionMaps.f64.end()
            && (!std::isfinite(level) || level < 0.0 || level > 1.0)) {
            chd::detail::set_last_error(
                "chd_decoder_commit: chroma_click_nr_level must be in [0.0, 1.0]");
            return CHD_E_INVALID_ARG;
        }
        for (const char *name :
             {CHD_OPT_CHROMA_CLICK_ENV_DIP_DB, CHD_OPT_CHROMA_CLICK_FREQ_OVERSHOOT}) {
            auto it = d->optionMaps.f64.find(name);
            if (it == d->optionMaps.f64.end()) continue;
            if (!std::isfinite(it->second) || it->second <= 0.0) {
                chd::detail::set_last_error(std::string("chd_decoder_commit: ") + name
                                            + " must be a positive threshold");
                return CHD_E_INVALID_ARG;
            }
            if (level <= 0.0) {
                chd::detail::set_last_error(
                    std::string("chd_decoder_commit: ") + name
                    + " is only meaningful when chroma_click_nr_level > 0");
                return CHD_E_INVALID_ARG;
            }
        }
    }

    chd::metadata::LdDecodeMetaData *meta = d->video->metadata.get();
    if (meta == nullptr) {
        chd::detail::set_last_error("chd_decoder_commit: failed to resolve video metadata");
        return CHD_E_METADATA_MISSING;
    }
    chd::metadata::LdDecodeMetaData::VideoParameters vp = meta->getVideoParameters();
    applyCropOverrides(vp, d->optionMaps);
    if (const chd_status_t s = validateCrop(vp); s != CHD_OK) return s;

    if (abiFormat == CHD_PIXEL_YUV440P16 || abiFormat == CHD_PIXEL_YUV440PS) {
        // The 4:4:0 planes weave both fields' half-rate chroma lattices by
        // row parity, which tiles only over whole two-line pairs of each
        // field: any other crop leaves one plane with unequal row counts on
        // the two parities and no slot that keeps parity meaning field.
        const int32_t activeLines = vp.lastActiveFrameLine - vp.firstActiveFrameLine + 1;
        if ((activeLines % 4) != 0) {
            chd::detail::set_last_error(
                "chd_decoder_commit: 4:4:0 output weaves both fields' chroma rows, "
                "so the active frame-line crop must span a multiple of 4 lines (got "
                + std::to_string(activeLines) + ")");
            return CHD_E_INVALID_ARG;
        }
    }

    // Apply REVERSE_FIELD_ORDER. Matches upstream ld-chroma-decoder `-r`
    // flag: flips the metadata-wide isFirstFieldFirst flag, which
    // SourceField::loadFields consumes via getFirstFieldNumber /
    // getSecondFieldNumber to swap which physical field is treated as
    // "first" within each frame. Default (no option, or option=false)
    // restores still-frame default of first-field-first.
    {
        auto it = d->optionMaps.boolean.find(CHD_OPT_REVERSE_FIELD_ORDER);
        const bool reverse = (it != d->optionMaps.boolean.end()) && it->second;
        meta->setIsFirstFieldFirst(!reverse);
    }

    // Padding surrounds the crop rather than widening it, so vp is final here:
    // every per-worker decoder configures against the same crop the output
    // writer frames.
    d->outputWriter.updateConfiguration(vp, outCfg);

    if (d->kind == CHD_DEC_NONE) {
        // Geometry/metadata-only decoder: snapshot the committed framing but
        // build no chroma decode engines and require no NN model. The dropout
        // span/mask queries and chd_decoder_get_output_info read this state;
        // chd_decode_frame is rejected.
        d->resolvedKind      = CHD_DEC_NONE;
        d->videoParameters   = vp;
        d->outputConfig      = outCfg;
        d->outputPixelFormat = abiFormat;
        d->lookBehind        = 0;
        d->lookAhead         = 0;
        d->threadCount       = 0;
        d->committed         = true;
        return CHD_OK;
    }

    // Resolve worker count before building decoders.
    int32_t threadCount = 0;
    {
        auto it = d->optionMaps.i32.find(CHD_OPT_THREAD_COUNT);
        threadCount = (it != d->optionMaps.i32.end()) ? it->second : 0;
        if (threadCount <= 0) {
            const unsigned hc = std::thread::hardware_concurrency();
            threadCount = hc == 0 ? 2 : static_cast<int32_t>(hc);
        }
    }

    // Build threadCount decoder instances so chd_decode_frames_async
    // workers can each own one without locking. Configure each; bind the
    // shared NN session to each if applicable. NN kinds without a model
    // bound fail fast.
#if defined(CHD_WITH_NN)
    const bool needsNn = chd::decoders::registry::kindUsesNn(resolved);
    if (needsNn && !d->nnModelPending) {
        chd::detail::set_last_error(
            "chd_decoder_commit: NN decoder kind requires chd_decoder_set_nn_model first");
        return CHD_E_NN_MODEL_LOAD;
    }
#endif

    // Build `threadCount` instances of one decoder kind, configured against
    // the post-padding vp. `applyNn` binds the pending NN session (only the
    // colour kind ever needs it; the luma Mono pass never does).
    auto buildDecoders =
        [&](chd_decoder_kind_t kind, bool applyNn,
            std::vector<std::unique_ptr<chd::decoders::Decoder>> &outDecs) -> chd_status_t {
        outDecs.clear();
        outDecs.reserve(threadCount);
        for (int32_t w = 0; w < threadCount; w++) {
            auto inst = chd::decoders::registry::build(kind, d->optionMaps);
            if (!inst) {
                chd::detail::set_last_error("chd_decoder_commit: unknown or unsupported decoder kind");
                return CHD_E_DECODER_UNKNOWN;
            }
#if defined(CHD_WITH_NN)
            if (applyNn && d->nnModelPending) {
                if (!chd::decoders::registry::applyNnModel(kind, *inst, d->nnModelPending)) {
                    chd::detail::set_last_error(
                        "chd_decoder_commit: NN model bound to a non-NN decoder kind");
                    return CHD_E_DECODER_INCOMPATIBLE;
                }
            }
#else
            (void)applyNn;
#endif
            chd::detail::clear_last_error();
            if (!inst->configure(vp)) {
                chd::detail::set_last_error(
                    "chd_decoder_commit: " +
                    chd::detail::detail_or("decoder rejected the input video parameters"));
                return CHD_E_DECODER_INCOMPATIBLE;
            }
            outDecs.push_back(std::move(inst));
        }
        return CHD_OK;
    };

    // For a decode-level Y/C merge the configured (resolved) kind decodes the
    // chroma plane; the luma plane is decoded with Mono. Otherwise the resolved
    // kind decodes the single composite source.
    const bool ycMerge = (d->video->chromaSource != nullptr);

    std::vector<std::unique_ptr<chd::decoders::Decoder>> built;
    std::vector<std::unique_ptr<chd::decoders::Decoder>> builtChroma;
    if (ycMerge) {
        if (const chd_status_t rc = buildDecoders(resolved, /*applyNn=*/true, builtChroma);
            rc != CHD_OK) return rc;
        if (const chd_status_t rc = buildDecoders(CHD_DEC_MONO, /*applyNn=*/false, built);
            rc != CHD_OK) return rc;
    } else {
        if (const chd_status_t rc = buildDecoders(resolved, /*applyNn=*/true, built);
            rc != CHD_OK) return rc;
    }

    std::vector<std::unique_ptr<std::mutex>> builtMutexes;
    builtMutexes.reserve(threadCount);
    for (int32_t w = 0; w < threadCount; w++) {
        builtMutexes.push_back(std::make_unique<std::mutex>());
    }

    // The metadata's VideoParameters stay at their pre-padding values:
    // SourceField::loadFields only reads `isSubcarrierLocked` from it
    // (which padding doesn't touch), and DecoderPool follows the same
    // convention of not writing back the post-padding vp. Keeping the
    // metadata pristine lets multiple decoders share one chd_video with
    // independent padding choices.

    d->resolvedKind = resolved;
    d->decoders = std::move(built);
    d->decoderMutexes = std::move(builtMutexes);
    d->chromaDecoders = std::move(builtChroma);
    d->videoParameters = vp;
    d->outputConfig = outCfg;
    d->outputPixelFormat = abiFormat;
    d->lookBehind = d->decoders.front()->getLookBehind();
    d->lookAhead  = d->decoders.front()->getLookAhead();
    if (ycMerge) {
        d->chromaLookBehind = d->chromaDecoders.front()->getLookBehind();
        d->chromaLookAhead  = d->chromaDecoders.front()->getLookAhead();
    }
    d->threadCount = threadCount;

    d->committed = true;
    return CHD_OK;
}

chd_status_t chd_decoder_get_chroma_click_thresholds(const chd_decoder_t *d,
                                                     double *env_dip_db,
                                                     double *freq_overshoot) {
    if (d == nullptr || env_dip_db == nullptr || freq_overshoot == nullptr) {
        return set_arg_error("chd_decoder_get_chroma_click_thresholds");
    }
    auto *mutableD = const_cast<chd_decoder_t *>(d);
    std::lock_guard<std::mutex> sl(mutableD->statsMutex);
    if (!d->haveClickThresholds) {
        chd::detail::set_last_error(
            "chd_decoder_get_chroma_click_thresholds: no decode has run with "
            "chroma_click_nr_level > 0");
        return CHD_E_UNSUPPORTED;
    }
    *env_dip_db = d->lastClickEnvDipDb;
    *freq_overshoot = d->lastClickFreqOvershoot;
    return CHD_OK;
}

// ── Sync decode ─────────────────────────────────────────────────────────────

chd_status_t chd_decode_frame(chd_decoder_t *d, int64_t frame_index, chd_frame_t **out) {
    if (d == nullptr || out == nullptr) return set_arg_error("chd_decode_frame");
    *out = nullptr;
    if (!d->committed) {
        chd::detail::set_last_error("chd_decode_frame: chd_decoder_commit not called");
        return CHD_E_INVALID_ARG;
    }
    if (d->resolvedKind == CHD_DEC_NONE) {
        chd::detail::set_last_error(
            "chd_decode_frame: decoder kind is CHD_DEC_NONE (geometry/dropout only)");
        return CHD_E_DECODER_INCOMPATIBLE;
    }
    // Sync decode uses worker 0's decoder + its own mutex. Concurrent
    // chd_decode_frames_async calls can run on workers 1..N-1 against
    // different decoder instances; they don't contend with sync.
    std::lock_guard<std::mutex> lock(*d->decoderMutexes[0]);
    return decodeFrameLocked(d, /*workerIdx=*/0, frame_index, out);
}

// ── Async decode ─────────────────────────────────────────────────────────────

chd_status_t chd_decode_frames_async(chd_decoder_t *d,
                                      const int64_t *indices, size_t n,
                                      chd_frame_done_cb cb, void *user,
                                      chd_cancel_t *cancel) {
    if (d == nullptr || cb == nullptr) return set_arg_error("chd_decode_frames_async");
    if (n > 0 && indices == nullptr) return set_arg_error("chd_decode_frames_async", "indices");
    if (!d->committed) {
        chd::detail::set_last_error("chd_decode_frames_async: chd_decoder_commit not called");
        return CHD_E_INVALID_ARG;
    }
    if (d->resolvedKind == CHD_DEC_NONE) {
        chd::detail::set_last_error(
            "chd_decode_frames_async: decoder kind is CHD_DEC_NONE (geometry/dropout only)");
        return CHD_E_DECODER_INCOMPATIBLE;
    }
    if (n == 0) return CHD_OK;

    // W workers; each owns decoder[w] for the duration of the call. Workers
    // race-pull next-index from `next` so unequal frame costs balance
    // automatically. decoderMutexes[w] is held per decode, not per worker:
    // it must cover decodeFrameLocked (which mutates decoder[w] state and
    // contends with sync `chd_decode_frame` on worker 0), but must be
    // released before `cb` runs — a callback that re-enters the library
    // (e.g. chd_decode_frame, which takes decoderMutexes[0]) would
    // self-deadlock on the non-recursive mutex otherwise.
    const size_t W = std::min<size_t>(n, d->decoders.size());
    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(W);
    for (size_t w = 0; w < W; w++) {
        workers.emplace_back([d, indices, n, cb, user, cancel, w, &next]() {
            while (true) {
                const size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= n) return;
                const int64_t idx = indices[i];
                if (cancel != nullptr && cancel->requested.load(std::memory_order_acquire)) {
                    cb(user, CHD_E_CANCELLED, idx, nullptr);
                    continue;
                }
                chd_frame_t *frame = nullptr;
                chd_status_t rc;
                {
                    std::lock_guard<std::mutex> lock(*d->decoderMutexes[w]);
                    rc = decodeFrameLocked(d, w, idx, &frame);
                }
                cb(user, rc, idx, frame);
            }
        });
    }
    for (auto &t : workers) t.join();
    return CHD_OK;
}

}  // extern "C"
