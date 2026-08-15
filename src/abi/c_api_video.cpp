// SPDX-License-Identifier: GPL-3.0-or-later
#include <chromadec/video.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../common/error_state.h"
#include "../common/log.h"
#include "../format/sample_encoding.h"
#include "../format/signal_state.h"
#include "../format/video_standards.h"
#include "../metadata/core.h"
#include "../metadata/cvbs_metadata_sqlite.h"
#include "../metadata/ld_metadata_sqlite.h"
#include "../reader/cvbs_composite_source.h"
#include "../reader/cvbs_yc_source.h"
#include "../reader/source.h"
#include "../reader/tbc_source.h"
#include "handles.h"

namespace fs = std::filesystem;

namespace {

// Records the detail and returns the status. CHD_E_INVALID_ARG is the default
// because most of the guards here really are bad caller arguments; anything
// about the state of the files on disk names its own status.
chd_status_t set_error(const std::string &msg, chd_status_t status = CHD_E_INVALID_ARG) {
    chd::detail::set_last_error(msg);
    return status;
}

// Synthesize an LdDecodeMetaData from an ISource that doesn't carry one
// (CVBS sources, whose .meta sidecar covers VideoParameters but not the
// per-field Field rows that SourceField::loadFields + the multi-source
// DropoutCorrector both consume). Generates alternating is-first-field
// metadata matching the ld-decode convention so 1-based frame
// number → field number translation works.
// NTSC only: the comb decoder derives each line's chroma sign from the
// four-field RS-170 sequence position, so every field needs a measured
// fieldPhaseID (an ld-decode `.tbc` carries one per field; CVBS files don't).
// PAL decoders detect burst phase per line and need none.
std::vector<int32_t> measureNtscFieldPhaseIds(chd::reader::ISource &src, int32_t nf) {
    std::vector<int32_t> ids(static_cast<size_t>(nf), -1);
    const auto &vp = src.parameters();
    if (vp.system != chd::metadata::NTSC) return ids;
    if (!chd::format::getSampleEncoding(src.sampleEncoding()).hasStandardAmplitudeMapping) {
        return ids;
    }

    // Post-VBI field lines with a stable burst, inside every field of the
    // shortest servable capture.
    constexpr int32_t kFirstRow = 30;
    constexpr int32_t kNumRows = 12;
    std::vector<std::optional<bool>> polarity(static_cast<size_t>(nf));
    for (int32_t i = 0; i < nf; i++) {
        const auto rows = src.getVideoField(i + 1, kFirstRow + 1, kFirstRow + kNumRows);
        polarity[i] = chd::format::measureNtscFieldBurstPolarity(
            rows.data(), kFirstRow, kNumRows, vp.fieldWidth, vp.colourBurstStart,
            vp.colourBurstEnd, vp.blanking16bIre, vp.white16bIre);
    }

    int32_t assigned = 0;
    for (int32_t i = 0; i + 1 < nf; i += 2) {
        if (!polarity[i].has_value() || !polarity[i + 1].has_value()) continue;
        // RS-170: fields 1 and 4 carry positive burst phase on even field
        // lines, so a frame's (first, second) polarity pair fixes its
        // position in the four-field sequence.
        const bool p1 = *polarity[i];
        const bool p2 = *polarity[i + 1];
        ids[i]     = p1 ? (p2 ? 4 : 1) : (p2 ? 3 : 2);
        ids[i + 1] = p1 ? (p2 ? 1 : 2) : (p2 ? 4 : 3);
        assigned += 2;
    }
    if (assigned > 0 && assigned < nf) {
        chd::log::warn() << "synthesizeMetadata: NTSC burst phase measurable on only"
                         << assigned << "of" << nf << "fields; unmeasured frames keep"
                         << "an unknown field phase and may decode with inverted chroma";
    }
    return ids;
}

std::unique_ptr<chd::metadata::LdDecodeMetaData>
synthesizeMetadata(chd::reader::ISource &src, bool firstFieldFirst) {
    auto meta = std::make_unique<chd::metadata::LdDecodeMetaData>();
    meta->setVideoParameters(src.parameters());
    meta->setIsFirstFieldFirst(firstFieldFirst);

    const int32_t nf = src.getNumberOfAvailableFields();
    if (nf <= 0) return meta;

    const std::vector<int32_t> phaseIds = measureNtscFieldPhaseIds(src, nf);
    for (int32_t i = 0; i < nf; i++) {
        chd::metadata::LdDecodeMetaData::Field f;
        f.seqNo = i + 1;
        f.isFirstField = (i % 2 == 0) == firstFieldFirst;
        f.fieldPhaseID = phaseIds[i];
        meta->appendField(f);
    }
    return meta;
}

// Declared-vs-measured chroma check for 625/50 captures (warn-only, never
// switches decode semantics): the back-porch reference either alternates
// like the SECAM FM carrier pair or sits constant at the PAL burst
// frequency. A contradiction usually means the open-time declaration (or
// its absence) is wrong for this capture. For a Y/C pair this runs on the
// chroma plane; a luma plane carries no porch reference and measures
// inconclusive.
void warn625ChromaSignatureMismatch(chd::reader::ISource &src, const char *what) {
    const auto &vp = src.parameters();
    if (vp.system != chd::metadata::PAL && vp.system != chd::metadata::SECAM) return;
    if (vp.colourBurstStart <= 0 || vp.colourBurstEnd <= vp.colourBurstStart
        || vp.colourBurstEnd >= vp.fieldWidth || vp.sampleRate <= 0.0) {
        return;
    }
    if (!chd::format::getSampleEncoding(src.sampleEncoding()).hasStandardAmplitudeMapping) {
        return;
    }

    // Post-VBI rows with a settled reference, over the first few fields.
    constexpr int32_t kFirstRow = 40;
    constexpr int32_t kNumRows = 32;
    const int32_t numFields = std::min<int32_t>(src.getNumberOfAvailableFields(), 4);
    std::vector<uint16_t> rowBuffer;
    std::vector<double> means, alternations;
    for (int32_t i = 0; i < numFields; i++) {
        const auto rows = src.getVideoField(i + 1, kFirstRow + 1, kFirstRow + kNumRows);
        const auto sig = chd::format::measure625ChromaPorchSignature(
            rows.data(), kNumRows, vp.fieldWidth, vp.colourBurstStart,
            vp.colourBurstEnd, vp.sampleRate, vp.blanking16bIre, vp.white16bIre);
        if (!sig) continue;
        means.push_back(sig->meanHz);
        alternations.push_back(sig->alternationHz);
    }
    if (means.empty()) return;
    double mean = 0.0, alternation = 0.0;
    for (size_t i = 0; i < means.size(); i++) {
        mean += means[i];
        alternation += alternations[i];
    }
    mean /= means.size();
    alternation /= alternations.size();

    const bool looksSecam = alternation > 100000.0 && alternation < 220000.0;
    const bool looksPal = alternation < 50000.0 && std::abs(mean - 4433618.75) < 40000.0;
    if (vp.system == chd::metadata::PAL && looksSecam) {
        chd::log::warn().nospace()
            << what << ": back-porch chroma reference alternates by "
            << alternation << " Hz line to line, the SECAM FM carrier-pair "
            << "signature; if this capture is SECAM, re-declare it with "
            << "chd_video_params_t.standard = CHD_STD_SECAM";
    } else if (vp.system == chd::metadata::SECAM && looksPal) {
        chd::log::warn().nospace()
            << what << ": back-porch chroma reference is a constant "
            << mean << " Hz burst, the PAL signature; the SECAM declaration "
            << "looks wrong for this capture";
    }
}

chd_video_standard_t toAbiStandard(chd::metadata::VideoSystem system) {
    switch (system) {
        case chd::metadata::PAL:   return CHD_STD_PAL;
        case chd::metadata::NTSC:  return CHD_STD_NTSC;
        case chd::metadata::PAL_M: return CHD_STD_PAL_M;
        case chd::metadata::SECAM: return CHD_STD_SECAM;
    }
    return CHD_STD_UNKNOWN;
}

std::optional<chd::metadata::VideoSystem> fromAbiStandard(chd_video_standard_t standard) {
    switch (standard) {
        case CHD_STD_NTSC:  return chd::metadata::NTSC;
        case CHD_STD_PAL:   return chd::metadata::PAL;
        case CHD_STD_PAL_M: return chd::metadata::PAL_M;
        case CHD_STD_SECAM: return chd::metadata::SECAM;
        default:            return std::nullopt;
    }
}

// 625-line systems share raster geometry with each other, as do the 525-line
// systems; an open-time colour-standard re-declaration is only meaningful
// within the same line standard.
bool is625Line(chd::metadata::VideoSystem system) {
    return system == chd::metadata::PAL || system == chd::metadata::SECAM;
}

chd_sample_encoding_t toAbiEncoding(chd::format::SampleEncoding encoding) {
    switch (encoding) {
        case chd::format::SampleEncoding::CVBS_U10_4FSC:   return CHD_ENC_CVBS_U10_4FSC;
        case chd::format::SampleEncoding::CVBS_U16_4FSC:   return CHD_ENC_CVBS_U16_4FSC;
        case chd::format::SampleEncoding::CVBS_TPG21_4FSC: return CHD_ENC_CVBS_TPG21_4FSC;
        case chd::format::SampleEncoding::CVBS_S16_4FSC:   return CHD_ENC_CVBS_S16_4FSC;
        case chd::format::SampleEncoding::RAW_S16_28M:     return CHD_ENC_RAW_S16_28M;
        case chd::format::SampleEncoding::RAW_S16_40M:     return CHD_ENC_RAW_S16_40M;
    }
    return CHD_ENC_UNKNOWN;
}

chd_signal_state_t toAbiSignalState(chd::format::SignalState state) {
    switch (state) {
        case chd::format::SignalState::STANDARD_TBC_LOCKED:      return CHD_SIG_STANDARD_TBC_LOCKED;
        case chd::format::SignalState::STANDARD_TBC_UNLOCKED:    return CHD_SIG_STANDARD_TBC_UNLOCKED;
        case chd::format::SignalState::STANDARD_RAW:             return CHD_SIG_STANDARD_RAW;
        case chd::format::SignalState::NONSTANDARD_TBC_LOCKED:   return CHD_SIG_NONSTANDARD_TBC_LOCKED;
        case chd::format::SignalState::NONSTANDARD_TBC_UNLOCKED: return CHD_SIG_NONSTANDARD_TBC_UNLOCKED;
        case chd::format::SignalState::NONSTANDARD_RAW:          return CHD_SIG_NONSTANDARD_RAW;
    }
    return CHD_SIG_UNKNOWN;
}

chd_frame_layout_t toAbiFrameLayout(chd::format::FrameLayout layout) {
    switch (layout) {
        case chd::format::FrameLayout::FIELD_RASTER: return CHD_FRAME_LAYOUT_FIELD_RASTER;
        case chd::format::FrameLayout::FRAME_NATIVE: return CHD_FRAME_LAYOUT_FRAME_NATIVE;
        case chd::format::FrameLayout::UNKNOWN:      break;
    }
    return CHD_FRAME_LAYOUT_UNKNOWN;
}

chd::format::FrameLayout fromAbiFrameLayout(chd_frame_layout_t layout) {
    switch (layout) {
        case CHD_FRAME_LAYOUT_FIELD_RASTER: return chd::format::FrameLayout::FIELD_RASTER;
        case CHD_FRAME_LAYOUT_FRAME_NATIVE: return chd::format::FrameLayout::FRAME_NATIVE;
        default:                            return chd::format::FrameLayout::UNKNOWN;
    }
}

// Native frame total for the standard (the spec's exact frame sizes),
// reported through chd_video_info_t regardless of the container layout.
int32_t nativeSamplesPerFrame(chd::metadata::VideoSystem system) {
    switch (system) {
        case chd::metadata::PAL:
            return chd::format::getVideoStandard(chd::format::VideoStandard::PAL).samplesPerFrame;
        case chd::metadata::NTSC:
            return chd::format::getVideoStandard(chd::format::VideoStandard::NTSC).samplesPerFrame;
        case chd::metadata::PAL_M:
            return chd::format::getVideoStandard(chd::format::VideoStandard::PAL_M).samplesPerFrame;
        case chd::metadata::SECAM:
            // SECAM shares the 625/50 sampling lattice with PAL.
            return chd::format::getVideoStandard(chd::format::VideoStandard::PAL).samplesPerFrame;
    }
    return 0;
}

// A CVBS-spec sidecar uses the `.meta` extension (cvbs_file table); an
// ld-decode sidecar is `.db` (sqlite) or `.json`. The flavour decides which
// reader + metadata path an input takes.
bool isCvbsSidecar(const std::string &path) {
    const std::string suffix = ".meta";
    return path.size() >= suffix.size()
        && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Strip extension and append ".meta" for sidecar auto-location. The CVBS
// spec's file naming convention pairs `<basename>.cvbs` (or
// `<basename>.cvbsy` + `<basename>.cvbsc`) with `<basename>.meta`. Stripping
// whatever extension is present also covers the pre-v1.5.0 `.composite`,
// `.y` and `.c` names.
std::string defaultMetaPath(const std::string &dataPath) {
    const fs::path p(dataPath);
    fs::path stem = p;
    stem.replace_extension("");
    stem += ".meta";
    return stem.string();
}

// Resolve the sidecar for a data file and report its flavour. With an
// explicit sidecar, the flavour follows its extension. Auto-location order:
// the ld-decode `<path>.db` then `<path>.json`, then the CVBS
// `<basename>.meta`. `found` is false when nothing was located (the caller
// then needs a chd_video_params_t override).
struct SidecarResolution {
    std::string path;
    bool        isCvbs = false;
    bool        found  = false;
};

SidecarResolution resolveSidecarFlavour(const std::string &dataPath,
                                        const char *sidecarOrNull) {
    SidecarResolution r;
    if (sidecarOrNull != nullptr) {
        r.path  = sidecarOrNull;
        r.found = fs::exists(r.path);
        r.isCvbs = isCvbsSidecar(r.path);
        return r;
    }
    const std::string db   = dataPath + ".db";
    const std::string json = dataPath + ".json";
    const std::string meta = defaultMetaPath(dataPath);
    if (fs::exists(db))        { r.path = db;   r.found = true; r.isCvbs = false; }
    else if (fs::exists(json)) { r.path = json; r.found = true; r.isCvbs = false; }
    else if (fs::exists(meta)) { r.path = meta; r.found = true; r.isCvbs = true; }
    return r;
}

// Resolution chain for CVBS open functions:
//   metaPathOrNull → auto-located <basename>.meta → override_or_null →
//   CHD_E_METADATA_MISSING.
//
// The first available source wins. The returned preset triple is what the
// CvbsCompositeSource / CvbsYcSource open path needs; `meta` (when a sidecar
// was found) also carries the black_level override, applied at open time.
struct ResolvedCvbsParams {
    const chd::format::VideoStandardPreset *videoStandard;
    chd::format::SampleEncoding             sampleEncoding;
    chd::format::SignalState                signalState;
    std::optional<chd::metadata::CvbsMetadata> meta;  // present when sidecar found
    // Merged field-wise from the caller override even when a sidecar is
    // present: the `cvbs_file` schema carries neither the container layout,
    // the sampling lattice, nor the field order, so these stay overridable.
    chd::format::FrameLayout                layoutOverride = chd::format::FrameLayout::UNKNOWN;
    std::optional<bool>                     subcarrierLockedOverride;
    bool                                    secondFieldFirst = false;
    std::optional<int64_t>                  declaredFrames;
    // The capture is SECAM stored under the byte-compatible 625/50 PAL
    // preset (the CVBS spec has no SECAM preset yet); re-declare the opened
    // source and its synthesized metadata after open.
    bool                                    declareSecam = false;
};

chd_status_t resolveCvbsParams(const std::string &dataPath,
                               const char *metaPathOrNull,
                               const chd_video_params_t *overrideOrNull,
                               ResolvedCvbsParams *out) {
    if (overrideOrNull != nullptr) {
        out->layoutOverride = fromAbiFrameLayout(overrideOrNull->layout);
        out->subcarrierLockedOverride = overrideOrNull->is_subcarrier_locked != 0;
        out->secondFieldFirst = overrideOrNull->is_second_field_first != 0;
    }

    // Try explicit or auto-located sidecar first.
    std::string sidecar;
    if (metaPathOrNull != nullptr) {
        sidecar = metaPathOrNull;
    } else {
        sidecar = defaultMetaPath(dataPath);
        if (!fs::exists(sidecar)) sidecar.clear();
    }
    if (!sidecar.empty()) {
        if (!fs::exists(sidecar)) {
            return set_error("CVBS metadata sidecar not found: " + sidecar,
                         CHD_E_FILE_NOT_FOUND);
        }
        auto parsed = chd::metadata::readCvbsMetadata(sidecar);
        if (!parsed) {
            // detail already populated by reader
            return CHD_E_METADATA_CORRUPT;
        }
        out->videoStandard  = parsed->videoStandard;
        out->sampleEncoding = parsed->sampleEncoding;
        out->signalState    = parsed->signalState;
        out->declaredFrames = parsed->numberOfSequentialFrames;
        if (overrideOrNull != nullptr && overrideOrNull->standard == CHD_STD_SECAM) {
            if (parsed->videoStandard->videoSystem != chd::metadata::PAL) {
                return set_error(
                    "CVBS open: a SECAM re-declaration requires the sidecar's "
                    "preset to be the 625-line PAL lattice");
            }
            out->declareSecam = true;
        }
        out->meta           = std::move(parsed);
        return CHD_OK;
    }

    // No sidecar — fall back to caller-supplied overrides.
    if (overrideOrNull == nullptr) {
        chd::detail::set_last_error(
            "CVBS open: no metadata sidecar and no chd_video_params_t override");
        return CHD_E_METADATA_MISSING;
    }

    // Translate ABI standard enum back into a format preset. The CVBS spec
    // has no SECAM preset yet, so a SECAM declaration selects the
    // byte-compatible 625/50 PAL lattice and re-declares after open.
    const chd::format::VideoStandardPreset *standard = nullptr;
    switch (overrideOrNull->standard) {
        case CHD_STD_PAL:   standard = &chd::format::getVideoStandard(chd::format::VideoStandard::PAL);   break;
        case CHD_STD_NTSC:  standard = &chd::format::getVideoStandard(chd::format::VideoStandard::NTSC);  break;
        case CHD_STD_PAL_M: standard = &chd::format::getVideoStandard(chd::format::VideoStandard::PAL_M); break;
        case CHD_STD_SECAM:
            standard = &chd::format::getVideoStandard(chd::format::VideoStandard::PAL);
            out->declareSecam = true;
            break;
        default:
            return set_error("CVBS open: chd_video_params_t.standard unset or unknown");
    }

    chd::format::SampleEncoding encoding;
    switch (overrideOrNull->encoding) {
        case CHD_ENC_CVBS_U10_4FSC:   encoding = chd::format::SampleEncoding::CVBS_U10_4FSC;   break;
        case CHD_ENC_CVBS_U16_4FSC:   encoding = chd::format::SampleEncoding::CVBS_U16_4FSC;   break;
        case CHD_ENC_CVBS_TPG21_4FSC: encoding = chd::format::SampleEncoding::CVBS_TPG21_4FSC; break;
        case CHD_ENC_CVBS_S16_4FSC:   encoding = chd::format::SampleEncoding::CVBS_S16_4FSC;   break;
        case CHD_ENC_RAW_S16_28M:     encoding = chd::format::SampleEncoding::RAW_S16_28M;     break;
        case CHD_ENC_RAW_S16_40M:     encoding = chd::format::SampleEncoding::RAW_S16_40M;     break;
        default:
            return set_error("CVBS open: chd_video_params_t.encoding unset or unknown");
    }

    chd::format::SignalState state;
    switch (overrideOrNull->signal_state) {
        case CHD_SIG_STANDARD_TBC_LOCKED:      state = chd::format::SignalState::STANDARD_TBC_LOCKED;      break;
        case CHD_SIG_STANDARD_TBC_UNLOCKED:    state = chd::format::SignalState::STANDARD_TBC_UNLOCKED;    break;
        case CHD_SIG_STANDARD_RAW:             state = chd::format::SignalState::STANDARD_RAW;             break;
        case CHD_SIG_NONSTANDARD_TBC_LOCKED:   state = chd::format::SignalState::NONSTANDARD_TBC_LOCKED;   break;
        case CHD_SIG_NONSTANDARD_TBC_UNLOCKED: state = chd::format::SignalState::NONSTANDARD_TBC_UNLOCKED; break;
        case CHD_SIG_NONSTANDARD_RAW:          state = chd::format::SignalState::NONSTANDARD_RAW;          break;
        default:
            return set_error("CVBS open: chd_video_params_t.signal_state unset or unknown");
    }

    out->videoStandard  = standard;
    out->sampleEncoding = encoding;
    out->signalState    = state;
    out->meta           = std::nullopt;
    return CHD_OK;
}

// One opened composite-shaped source plus the metadata the decode path needs.
// For ld-decode inputs `metadata` carries the real per-field rows; for CVBS
// inputs it is synthesized from the source parameters + field count.
struct OpenedSource {
    std::unique_ptr<chd::reader::ISource> source;
    std::unique_ptr<chd::metadata::LdDecodeMetaData> metadata;
    bool metadataSynthesized = false;
};

// Open a single composite source (ld-decode `.tbc` or CVBS `.cvbs`),
// auto-detecting the sidecar flavour. `fn` is the caller name for error
// prefixes; `path` must already exist (callers check).
chd_status_t openCompositeSource(const std::string &fn, const std::string &path,
                                 const char *sidecarOrNull,
                                 const chd_video_params_t *overrideOrNull,
                                 OpenedSource *out) {
    const SidecarResolution sc = resolveSidecarFlavour(path, sidecarOrNull);
    if (sidecarOrNull != nullptr && !sc.found) {
        return set_error(fn + ": sidecar file does not exist: " + sc.path,
                         CHD_E_FILE_NOT_FOUND);
    }

    if (sc.found && !sc.isCvbs) {
        // ld-decode path: real per-field metadata + TbcSource.
        auto metadata = std::make_unique<chd::metadata::LdDecodeMetaData>();
        chd::detail::clear_last_error();
        try {
            if (!metadata->read(sc.path)) {
                return set_error(fn + ": " + chd::detail::detail_or(
                                                 "failed to read sidecar metadata"),
                                 CHD_E_METADATA_CORRUPT);
            }
        } catch (const std::exception &e) {
            return set_error(fn + ": " + e.what(), CHD_E_METADATA_CORRUPT);
        }
        // A non-zero override standard re-declares the colour standard over
        // the sidecar's, for captures whose sidecar cannot express it (a
        // vhs-decode ME-SECAM sidecar says "PAL"). The line standard must
        // match; a 525-line capture cannot be re-declared as a 625-line one.
        if (overrideOrNull != nullptr && overrideOrNull->standard != CHD_STD_UNKNOWN) {
            const auto declared = fromAbiStandard(overrideOrNull->standard);
            if (!declared) {
                return set_error(fn + ": chd_video_params_t.standard unknown");
            }
            const auto current = metadata->getVideoParameters().system;
            if (*declared != current) {
                if (is625Line(*declared) != is625Line(current)) {
                    return set_error(fn + ": declared standard's line standard does not"
                                          " match the capture's");
                }
                metadata->overrideVideoSystem(*declared);
            }
        }
        const auto &vp = metadata->getVideoParameters();
        auto src = std::make_unique<chd::reader::TbcSource>();
        chd::detail::clear_last_error();
        if (!src->open(path, vp.fieldWidth * vp.fieldHeight, vp.fieldWidth)) {
            return set_error(fn + ": " + chd::detail::detail_or("failed to open sample file"),
                             CHD_E_IO);
        }
        src->bindVideoParameters(vp);
        out->source = std::move(src);
        out->metadata = std::move(metadata);
        out->metadataSynthesized = false;
        return CHD_OK;
    }

    // CVBS path: preset triple from the `.meta` sidecar or the override.
    ResolvedCvbsParams resolved{};
    const chd_status_t rc = resolveCvbsParams(
        path, sc.found ? sc.path.c_str() : nullptr, overrideOrNull, &resolved);
    if (rc != CHD_OK) return rc;
    auto src = std::make_unique<chd::reader::CvbsCompositeSource>();
    const std::optional<int32_t> blackOverride =
        resolved.meta ? resolved.meta->blackLevelOverride : std::nullopt;
    chd::detail::clear_last_error();
    if (!src->open(path, *resolved.videoStandard, resolved.sampleEncoding,
                   resolved.signalState, blackOverride, resolved.layoutOverride,
                   resolved.declaredFrames, resolved.subcarrierLockedOverride)) {
        return set_error(fn + ": " + chd::detail::detail_or("failed to open sample file"),
                         CHD_E_IO);
    }
    out->metadata = synthesizeMetadata(*src, !resolved.secondFieldFirst);
    if (resolved.declareSecam) {
        out->metadata->overrideVideoSystem(chd::metadata::SECAM);
        const auto &mvp = out->metadata->getVideoParameters();
        src->redeclareVideoSystem(mvp.system, mvp.fSC);
    }
    out->source = std::move(src);
    out->metadataSynthesized = true;
    return CHD_OK;
}

// Open one composite extra source and append it to `dst`.
chd_status_t addCompositeExtra(const std::string &fn,
                               std::vector<chd_video_extra> &dst,
                               const std::string &path, const char *sidecarOrNull) {
    if (!fs::exists(path)) {
        return set_error(fn + ": file does not exist: " + path,
                         CHD_E_FILE_NOT_FOUND);
    }
    OpenedSource opened;
    const chd_status_t rc = openCompositeSource(fn, path, sidecarOrNull, nullptr, &opened);
    if (rc != CHD_OK) return rc;
    chd_video_extra extra;
    extra.source = std::move(opened.source);
    extra.metadata = std::move(opened.metadata);
    extra.metadataSynthesized = opened.metadataSynthesized;
    dst.push_back(std::move(extra));
    return CHD_OK;
}

}  // namespace

extern "C" {

chd_status_t chd_video_open_composite(const char *path,
                                      const char *metadata_path_or_null,
                                      const chd_video_params_t *override_or_null,
                                      chd_video_t **out) {
    if (path == nullptr || out == nullptr) {
        return set_error("chd_video_open_composite: null argument");
    }
    *out = nullptr;
    if (!fs::exists(path)) {
        return set_error(std::string("chd_video_open_composite: file does not exist: ") + path,
                         CHD_E_FILE_NOT_FOUND);
    }

    OpenedSource opened;
    const chd_status_t rc = openCompositeSource(
        "chd_video_open_composite", path, metadata_path_or_null, override_or_null, &opened);
    if (rc != CHD_OK) return rc;

    warn625ChromaSignatureMismatch(*opened.source, "chd_video_open_composite");

    auto handle = std::make_unique<chd_video>();
    handle->primaryPath        = path;
    handle->source             = std::move(opened.source);
    handle->metadata           = std::move(opened.metadata);
    handle->metadataSynthesized = opened.metadataSynthesized;
    *out = handle.release();
    return CHD_OK;
}

chd_status_t chd_video_open_yc(const char *luma_path, const char *chroma_path,
                               const char *metadata_path_or_null,
                               const chd_video_params_t *override_or_null,
                               chd_video_t **out) {
    if (luma_path == nullptr || chroma_path == nullptr || out == nullptr) {
        return set_error("chd_video_open_yc: null argument");
    }
    *out = nullptr;
    if (!fs::exists(luma_path) || !fs::exists(chroma_path)) {
        return set_error("chd_video_open_yc: luma/chroma file does not exist",
                         CHD_E_FILE_NOT_FOUND);
    }

    const SidecarResolution sc = resolveSidecarFlavour(luma_path, metadata_path_or_null);
    // A CVBS `.cvbsy`/`.cvbsc` pair (a `.meta` sidecar, or no sidecar + override) reads
    // through a single CvbsYcSource that reconstructs a composite from the
    // centred-chroma `.cvbsc`. A vhs-decode luma.tbc + chroma.tbc pair instead
    // decodes each plane separately and merges (set up below).
    const bool decodeMerge = sc.found && !sc.isCvbs;

    auto handle = std::make_unique<chd_video>();
    handle->primaryPath = luma_path;

    if (!decodeMerge) {
        ResolvedCvbsParams resolved{};
        const chd_status_t rc = resolveCvbsParams(
            luma_path, sc.found ? sc.path.c_str() : nullptr, override_or_null, &resolved);
        if (rc != CHD_OK) return rc;
        auto src = std::make_unique<chd::reader::CvbsYcSource>();
        const std::optional<int32_t> blackOverride =
            resolved.meta ? resolved.meta->blackLevelOverride : std::nullopt;
        chd::detail::clear_last_error();
        if (!src->open(luma_path, chroma_path, *resolved.videoStandard,
                       resolved.sampleEncoding, resolved.signalState, blackOverride,
                       resolved.layoutOverride, resolved.declaredFrames,
                       resolved.subcarrierLockedOverride)) {
            return set_error("chd_video_open_yc: " +
                                 chd::detail::detail_or("failed to open y/c files"),
                             CHD_E_IO);
        }
        handle->metadata = synthesizeMetadata(*src, !resolved.secondFieldFirst);
        if (resolved.declareSecam) {
            handle->metadata->overrideVideoSystem(chd::metadata::SECAM);
            const auto &mvp = handle->metadata->getVideoParameters();
            src->redeclareVideoSystem(mvp.system, mvp.fSC);
        }
        warn625ChromaSignatureMismatch(*src, "chd_video_open_yc");
        handle->metadataSynthesized = true;
        handle->source  = std::move(src);
        *out = handle.release();
        return CHD_OK;
    }

    OpenedSource luma;
    chd_status_t rc = openCompositeSource(
        "chd_video_open_yc (luma)", luma_path, metadata_path_or_null, override_or_null, &luma);
    if (rc != CHD_OK) return rc;

    // Prefer the chroma plane's own sidecar, but fall back to the luma sidecar:
    // vhs-decode writes one shared `<base>.tbc.json` for the pair, not a
    // separate `<base>_chroma.tbc.json`. The shared sidecar describes the
    // (identical) geometry of both planes.
    const SidecarResolution chromaSc = resolveSidecarFlavour(chroma_path, nullptr);
    const char *chromaSidecar = chromaSc.found ? nullptr : sc.path.c_str();
    OpenedSource chroma;
    rc = openCompositeSource(
        "chd_video_open_yc (chroma)", chroma_path, chromaSidecar, override_or_null, &chroma);
    if (rc != CHD_OK) return rc;

    // The two planes come from one capture; require matching geometry and
    // frame count so the decoded Y and U/V line up.
    const auto &lvp = luma.source->parameters();
    const auto &cvp = chroma.source->parameters();
    if (lvp.fieldWidth != cvp.fieldWidth || lvp.fieldHeight != cvp.fieldHeight) {
        return set_error("chd_video_open_yc: luma and chroma have mismatched dimensions",
                         CHD_E_FORMAT_UNSUPPORTED);
    }
    if (luma.metadata->getNumberOfFrames() != chroma.metadata->getNumberOfFrames()) {
        return set_error("chd_video_open_yc: luma and chroma have different frame counts",
                         CHD_E_FORMAT_UNSUPPORTED);
    }

    warn625ChromaSignatureMismatch(*chroma.source, "chd_video_open_yc");

    handle->source             = std::move(luma.source);
    handle->metadata           = std::move(luma.metadata);
    handle->metadataSynthesized = luma.metadataSynthesized;
    handle->chromaSource       = std::move(chroma.source);
    handle->chromaMetadata     = std::move(chroma.metadata);
    *out = handle.release();
    return CHD_OK;
}

void chd_video_free(chd_video_t *v) {
    delete v;
}

chd_status_t chd_video_get_info(const chd_video_t *v, chd_video_info_t *out) {
    if (v == nullptr || out == nullptr || v->source == nullptr) {
        return set_error("chd_video_get_info: null argument");
    }
    const auto &vp = v->source->parameters();

    out->standard               = toAbiStandard(vp.system);
    // Report the source's actual sample encoding. An ld-decode `.tbc` reads as
    // CVBS_U16_4FSC (its on-disk layout); CVBS sources report the preset they
    // were opened with.
    out->encoding               = toAbiEncoding(v->source->sampleEncoding());
    out->signal_state           = toAbiSignalState(v->source->signalState());
    out->layout                 = toAbiFrameLayout(v->source->frameLayout());
    out->field_width            = vp.fieldWidth;
    out->field_height           = vp.fieldHeight;
    out->samples_per_frame      = nativeSamplesPerFrame(vp.system);
    out->sample_rate_hz         = vp.sampleRate;
    out->fsc_hz                 = vp.fSC;
    // VideoParameters carries the sample crop half-open; the ABI reports all
    // four crop bounds inclusive, matching the CHD_OPT_*_ACTIVE_* options.
    out->first_active_sample    = vp.activeVideoStart;
    out->last_active_sample     = vp.activeVideoEnd - 1;
    out->first_active_frame_line = vp.firstActiveFrameLine;
    out->last_active_frame_line  = vp.lastActiveFrameLine;
    out->black_16b_ire          = vp.black16bIre;
    out->white_16b_ire          = vp.white16bIre;
    out->blanking_16b_ire       = vp.blanking16bIre;
    // Both TBC and (synthesized) CVBS metadata expose frame count + field
    // order through the same accessors.
    out->num_frames         = const_cast<chd_video_t *>(v)->metadata->getNumberOfFrames();
    out->is_first_field_first =
        const_cast<chd_video_t *>(v)->metadata->getIsFirstFieldFirst() ? 1 : 0;
    out->is_widescreen          = vp.isWidescreen ? 1 : 0;
    out->is_subcarrier_locked   = vp.isSubcarrierLocked ? 1 : 0;
    return CHD_OK;
}

// Field-sequential signal line number <-> 0-indexed woven-raster frame line.
// Field 1 (top field) is even frame lines / signal lines 1..H; field 2 is odd
// frame lines / signal lines H+1..(2H)-1, where H = fieldHeight. See video.h
// for the full contract (source raster only; not monotonic; first may exceed
// last).
chd_status_t chd_video_frame_line_to_signal_line(const chd_video_t *v,
                                                  int32_t frame_line, int32_t *out) {
    if (v == nullptr || out == nullptr || v->source == nullptr) {
        return set_error("chd_video_frame_line_to_signal_line: null argument");
    }
    const int32_t H = v->source->parameters().fieldHeight;
    if (frame_line < 0 || frame_line > (2 * H) - 2) {
        return set_error("chd_video_frame_line_to_signal_line: frame_line out of range "
                         "[0, (2 * field_height) - 2]", CHD_E_OUT_OF_RANGE);
    }
    *out = (frame_line % 2 == 0) ? (frame_line / 2 + 1)
                                 : (H + (frame_line + 1) / 2);
    return CHD_OK;
}

chd_status_t chd_video_signal_line_to_frame_line(const chd_video_t *v,
                                                  int32_t signal_line, int32_t *out) {
    if (v == nullptr || out == nullptr || v->source == nullptr) {
        return set_error("chd_video_signal_line_to_frame_line: null argument");
    }
    const int32_t H = v->source->parameters().fieldHeight;
    if (signal_line < 1 || signal_line > (2 * H) - 1) {
        return set_error("chd_video_signal_line_to_frame_line: signal_line out of range "
                         "[1, (2 * field_height) - 1]", CHD_E_OUT_OF_RANGE);
    }
    *out = (signal_line <= H) ? (2 * (signal_line - 1))
                              : (2 * (signal_line - H - 1) + 1);
    return CHD_OK;
}

// Interface-standard sample numbering (SMPTE ST 244 / EBU Tech 3280-E,
// sample 0 = first digital active sample) <-> 0-indexed stored-row sample.
// The rotation between the two follows the source's horizontal alignment:
// blanking-start rows put standard sample 0 a full digital active line
// before the row end; sync-start rows start at the first sample at or after
// the standard's own 0H. See video.h for the full contract.
static chd_status_t standardSampleRotation(const chd_video_t *v, const char *caller,
                                           int32_t &rowWidth, int32_t &rowOfStandardZero) {
    const auto &vp = v->source->parameters();
    chd::format::VideoStandard standard;
    switch (vp.system) {
    case chd::metadata::NTSC:  standard = chd::format::VideoStandard::NTSC;  break;
    case chd::metadata::PAL_M: standard = chd::format::VideoStandard::PAL_M; break;
    default:                   standard = chd::format::VideoStandard::PAL;   break;
    }
    const auto &preset = chd::format::getVideoStandard(standard);
    const int32_t samplesPerLine =
        (preset.standard == chd::format::VideoStandard::PAL)
            ? 1135
            : static_cast<int32_t>(preset.samplesPerLineAvg);
    if (vp.fieldWidth != samplesPerLine) {
        return set_error(std::string(caller) + ": source row width " +
                             std::to_string(vp.fieldWidth) +
                             " is not the standard 4fsc line of " +
                             std::to_string(samplesPerLine) + " samples",
                         CHD_E_UNSUPPORTED);
    }
    if (v->source->horizontalAlignment() == chd::format::HorizontalAlignment::BLANKING_START) {
        rowOfStandardZero = samplesPerLine - preset.digitalActiveSamples;
    } else {
        rowOfStandardZero = samplesPerLine - static_cast<int32_t>(std::ceil(
            preset.digitalActiveSamples + preset.zeroHBlankingStartRow));
    }
    rowWidth = samplesPerLine;
    return CHD_OK;
}

chd_status_t chd_video_standard_sample_to_row_sample(const chd_video_t *v,
                                                     int32_t standard_sample, int32_t *out) {
    if (v == nullptr || out == nullptr || v->source == nullptr) {
        return set_error("chd_video_standard_sample_to_row_sample: null argument");
    }
    int32_t rowWidth = 0, rowOfStandardZero = 0;
    const chd_status_t status = standardSampleRotation(
        v, "chd_video_standard_sample_to_row_sample", rowWidth, rowOfStandardZero);
    if (status != CHD_OK) return status;
    if (standard_sample < 0 || standard_sample >= rowWidth) {
        return set_error("chd_video_standard_sample_to_row_sample: standard_sample out of "
                         "range [0, field_width - 1]", CHD_E_OUT_OF_RANGE);
    }
    *out = (standard_sample + rowOfStandardZero) % rowWidth;
    return CHD_OK;
}

chd_status_t chd_video_row_sample_to_standard_sample(const chd_video_t *v,
                                                     int32_t row_sample, int32_t *out) {
    if (v == nullptr || out == nullptr || v->source == nullptr) {
        return set_error("chd_video_row_sample_to_standard_sample: null argument");
    }
    int32_t rowWidth = 0, rowOfStandardZero = 0;
    const chd_status_t status = standardSampleRotation(
        v, "chd_video_row_sample_to_standard_sample", rowWidth, rowOfStandardZero);
    if (status != CHD_OK) return status;
    if (row_sample < 0 || row_sample >= rowWidth) {
        return set_error("chd_video_row_sample_to_standard_sample: row_sample out of "
                         "range [0, field_width - 1]", CHD_E_OUT_OF_RANGE);
    }
    *out = (row_sample - rowOfStandardZero + rowWidth) % rowWidth;
    return CHD_OK;
}

chd_status_t chd_video_add_extra_source_composite(chd_video_t *v, const char *path,
                                                  const char *metadata_path_or_null) {
    if (v == nullptr || path == nullptr) {
        return set_error("chd_video_add_extra_source_composite: null argument");
    }
    return addCompositeExtra("chd_video_add_extra_source_composite",
                             v->extraSources, path, metadata_path_or_null);
}

chd_status_t chd_video_add_extra_source_yc(chd_video_t *v, const char *luma_path,
                                           const char *chroma_path,
                                           const char *metadata_path_or_null) {
    if (v == nullptr || luma_path == nullptr || chroma_path == nullptr) {
        return set_error("chd_video_add_extra_source_yc: null argument");
    }
    if (!fs::exists(luma_path) || !fs::exists(chroma_path)) {
        return set_error("chd_video_add_extra_source_yc: luma/chroma file does not exist",
                         CHD_E_FILE_NOT_FOUND);
    }

    const SidecarResolution sc = resolveSidecarFlavour(luma_path, metadata_path_or_null);
    if (sc.found && !sc.isCvbs) {
        // vhs-decode pair: a luma extra corrects the luma plane and a chroma
        // extra corrects the separately-decoded chroma plane.
        chd_status_t rc = addCompositeExtra(
            "chd_video_add_extra_source_yc (luma)", v->extraSources,
            luma_path, metadata_path_or_null);
        if (rc != CHD_OK) return rc;
        // Chroma plane falls back to the shared luma sidecar (vhs-decode writes
        // one `.tbc.json` per pair).
        const SidecarResolution chromaSc = resolveSidecarFlavour(chroma_path, nullptr);
        const char *chromaSidecar = chromaSc.found ? nullptr : sc.path.c_str();
        return addCompositeExtra(
            "chd_video_add_extra_source_yc (chroma)", v->chromaExtraSources,
            chroma_path, chromaSidecar);
    }

    // CVBS `.cvbsy`/`.cvbsc` pair: one CvbsYcSource extra, matching the primary.
    ResolvedCvbsParams resolved{};
    const chd_status_t rc = resolveCvbsParams(
        luma_path, sc.found ? sc.path.c_str() : nullptr, nullptr, &resolved);
    if (rc != CHD_OK) return rc;
    auto src = std::make_unique<chd::reader::CvbsYcSource>();
    const std::optional<int32_t> blackOverride =
        resolved.meta ? resolved.meta->blackLevelOverride : std::nullopt;
    chd::detail::clear_last_error();
    if (!src->open(luma_path, chroma_path, *resolved.videoStandard,
                   resolved.sampleEncoding, resolved.signalState, blackOverride,
                   resolved.layoutOverride, resolved.declaredFrames,
                   resolved.subcarrierLockedOverride)) {
        return set_error("chd_video_add_extra_source_yc: " +
                             chd::detail::detail_or("open failed"),
                         CHD_E_IO);
    }
    chd_video_extra extra;
    extra.metadata = synthesizeMetadata(*src, !resolved.secondFieldFirst);
    extra.metadataSynthesized = true;
    extra.source = std::move(src);
    v->extraSources.push_back(std::move(extra));
    return CHD_OK;
}

}  // extern "C"
