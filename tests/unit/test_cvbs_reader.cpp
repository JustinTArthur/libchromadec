// SPDX-License-Identifier: GPL-3.0-or-later
//
// End-to-end smoke tests for the CVBS readers.
//
//   1. Synthesise a `.cvbs` file with two NTSC fields' worth of
//      uniform CVBS_U10_4FSC samples + a `.meta` sqlite sidecar with
//      preset = NTSC, encoding = CVBS_U10_4FSC, state = STANDARD_TBC_LOCKED.
//      Exercise chd_video_open_composite + chd_video_get_info; verify
//      the field count and sample-encoding fold (× 64).
//
//   2. Synthesise a `.cvbsc` chroma plane; verify the chroma-plane read
//      re-centres the 512-centred excursion on blanking.
//
//   3. Synthesise a meta sidecar with an unknown preset; verify the open
//      fails with CHD_E_METADATA_CORRUPT and a descriptive error.
//
//   4. Frame-layout coverage: resolveFrameLayout decision table (including
//      the 526-native = 525-packed size collision), the NTSC/PAL_M
//      frame-native conform (row mapping + dummy padding line), PAL
//      frame-native rejection, the subcarrier-lock derivation, and the
//      field-wise layout / is_subcarrier_locked override merge.

#include <chromadec/decoder.h>
#include <chromadec/frame.h>
#include <chromadec/log.h>
#include <chromadec/video.h>

#include <sqlite3.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Internal-only headers — pulled in via the static lib link in meson.build
// (matches the test_encode_orc_color_bars pattern).
#include "../../src/common/log.h"
#include "../../src/reader/cvbs_composite_source.h"
#include "../../src/format/video_standards.h"

namespace fs = std::filesystem;

namespace {

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond \
                  << " (last_error: " << (chd_last_error() ? chd_last_error() : "(none)") << ")\n"; \
        return 1; \
    } \
} while (0)

// The library emits nothing until a consumer installs a sink, so the
// diagnostic checks below turn one on around the call they want to see.
void enableDebugLogging() {
    chd_log_to_stderr();
    chd_set_log_level(CHD_LOG_DEBUG);
}

void disableDebugLogging() {
    chd_set_log_level(CHD_LOG_INFO);
    chd_set_log_callback(nullptr, nullptr);
}

const char *kCvbsSchema = R"(
PRAGMA user_version = 7;

CREATE TABLE cvbs_file (
    cvbs_file_id                INTEGER PRIMARY KEY,
    preset                      TEXT    NOT NULL,
    sample_encoding_preset      TEXT    NOT NULL,
    signal_state_preset         TEXT    NOT NULL,
    signal_type                 TEXT    NOT NULL,
    decoder                     TEXT    NOT NULL,
    git_branch                  TEXT,
    git_commit                  TEXT,
    number_of_sequential_frames INTEGER,
    black_level                 INTEGER,
    has_nonstandard_values      BOOLEAN,
    capture_notes               TEXT
);
)";

// user_version 8 adds the audio_locked column. Superseded, but files written
// against it are still in the wild and must open.
const char *kCvbsSchemaV8 = R"(
PRAGMA user_version = 8;

CREATE TABLE cvbs_file (
    cvbs_file_id                INTEGER PRIMARY KEY,
    preset                      TEXT    NOT NULL,
    sample_encoding_preset      TEXT    NOT NULL,
    signal_state_preset         TEXT    NOT NULL,
    signal_type                 TEXT    NOT NULL,
    decoder                     TEXT    NOT NULL,
    git_branch                  TEXT,
    git_commit                  TEXT,
    number_of_sequential_frames INTEGER,
    black_level                 INTEGER,
    has_nonstandard_values      BOOLEAN,
    audio_locked                BOOLEAN,
    capture_notes               TEXT
);
)";

// The spec's current revision: user_version 10 drops audio_locked and carries
// audio metadata in a sibling table this reader never touches.
const char *kCvbsSchemaV10 = R"(
PRAGMA user_version = 10;

CREATE TABLE cvbs_file (
    cvbs_file_id                INTEGER PRIMARY KEY,
    preset                      TEXT    NOT NULL,
    sample_encoding_preset      TEXT    NOT NULL,
    signal_state_preset         TEXT    NOT NULL,
    signal_type                 TEXT    NOT NULL,
    decoder                     TEXT    NOT NULL,
    git_branch                  TEXT,
    git_commit                  TEXT,
    number_of_sequential_frames INTEGER,
    black_level                 INTEGER,
    has_nonstandard_values      BOOLEAN,
    capture_notes               TEXT
);

CREATE TABLE audio_channel_pair (
    channel_pair                INTEGER PRIMARY KEY,
    description                 TEXT
);
)";

bool writeMetaSidecar(const std::string &metaPath, const std::string &preset,
                      const std::string &encoding, const std::string &state,
                      const std::string &signalType,
                      const char *schema = kCvbsSchema) {
    if (fs::exists(metaPath)) fs::remove(metaPath);
    sqlite3 *db = nullptr;
    if (sqlite3_open(metaPath.c_str(), &db) != SQLITE_OK) return false;
    char *err = nullptr;
    if (sqlite3_exec(db, schema, nullptr, nullptr, &err) != SQLITE_OK) {
        std::cerr << "schema error: " << err << "\n";
        sqlite3_free(err);
        sqlite3_close(db);
        return false;
    }
    std::string sql =
        "INSERT INTO cvbs_file (cvbs_file_id, preset, sample_encoding_preset, "
        "signal_state_preset, signal_type, decoder) VALUES (1, '" + preset + "', '" +
        encoding + "', '" + state + "', '" + signalType + "', 'cvbs-encode')";
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
        std::cerr << "insert error: " << err << "\n";
        sqlite3_free(err);
        sqlite3_close(db);
        return false;
    }
    sqlite3_close(db);
    return true;
}

// Write a binary file of `numSamples` int16_t values, all equal to `value`.
bool writeUniformSamples(const std::string &path, size_t numSamples, int16_t value) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    std::vector<int16_t> buf(numSamples, value);
    out.write(reinterpret_cast<const char *>(buf.data()),
              static_cast<std::streamsize>(buf.size() * sizeof(int16_t)));
    return out.good();
}

int testCompositeOpen() {
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "test.cvbs").string();
    const std::string meta      = (tmpDir / "test.meta").string();

    // Two NTSC fields = 2 × 910 × 263 = 478,660 samples of value 256 (blanking
    // in 10-bit CVBS_U10_4FSC). After conversion: 256 × 64 = 16384.
    const size_t samplesPerField = 910 * 263;
    REQUIRE(writeUniformSamples(composite, samplesPerField * 2, 256));
    REQUIRE(writeMetaSidecar(meta, "NTSC", "CVBS_U10_4FSC", "STANDARD_TBC_LOCKED", "composite"));

    chd_video_t *v = nullptr;
    REQUIRE(chd_video_open_composite(composite.c_str(), meta.c_str(), nullptr, &v) == CHD_OK);
    REQUIRE(v != nullptr);

    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    REQUIRE(info.standard     == CHD_STD_NTSC);
    REQUIRE(info.encoding     == CHD_ENC_CVBS_U10_4FSC);
    REQUIRE(info.signal_state == CHD_SIG_STANDARD_TBC_LOCKED);
    REQUIRE(info.field_width  == 910);
    REQUIRE(info.field_height == 263);
    REQUIRE(info.num_frames   == 1);  // 2 fields = 1 frame

    chd_video_free(v);
    return 0;
}

int testCompositeMissingMeta() {
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "nometa.cvbs").string();
    const std::string meta      = (tmpDir / "nometa.meta").string();
    if (fs::exists(meta)) fs::remove(meta);
    REQUIRE(writeUniformSamples(composite, 910 * 263 * 2, 256));

    // No metadata sidecar, no override → CHD_E_METADATA_MISSING.
    chd_video_t *v = nullptr;
    const chd_status_t rc = chd_video_open_composite(composite.c_str(), nullptr, nullptr, &v);
    REQUIRE(rc == CHD_E_METADATA_MISSING);
    REQUIRE(v == nullptr);
    return 0;
}

int testCompositeUnknownPreset() {
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "bad.cvbs").string();
    const std::string meta      = (tmpDir / "bad.meta").string();
    REQUIRE(writeUniformSamples(composite, 910 * 263 * 2, 256));
    REQUIRE(writeMetaSidecar(meta, "DOES_NOT_EXIST", "CVBS_U10_4FSC", "STANDARD_TBC_LOCKED", "composite"));

    chd_video_t *v = nullptr;
    const chd_status_t rc = chd_video_open_composite(composite.c_str(), meta.c_str(), nullptr, &v);
    REQUIRE(rc == CHD_E_METADATA_CORRUPT);
    REQUIRE(v == nullptr);
    return 0;
}

int testCompositeOverride() {
    // No sidecar; supply chd_video_params_t override.
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "override.cvbs").string();
    const std::string meta      = (tmpDir / "override.meta").string();
    if (fs::exists(meta)) fs::remove(meta);
    REQUIRE(writeUniformSamples(composite, 910 * 263 * 2, 256));

    chd_video_params_t params{};
    params.standard     = CHD_STD_NTSC;
    params.encoding     = CHD_ENC_CVBS_U10_4FSC;
    params.signal_state = CHD_SIG_STANDARD_TBC_UNLOCKED;

    chd_video_t *v = nullptr;
    REQUIRE(chd_video_open_composite(composite.c_str(), nullptr, &params, &v) == CHD_OK);

    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    REQUIRE(info.signal_state == CHD_SIG_STANDARD_TBC_UNLOCKED);
    chd_video_free(v);
    return 0;
}

int testCompositeSampleConversion() {
    // Read a field via the internal source (the C ABI doesn't yet expose
    // field access). Confirms the per-encoding ×64 fold.
    using namespace chd::format;
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "samples.cvbs").string();
    const size_t samplesPerField = 910 * 263;
    // Write samples with int16 value 282 (10-bit "black" for PAL — testing
    // the conversion path itself).
    REQUIRE(writeUniformSamples(composite, samplesPerField, 282));

    chd::reader::CvbsCompositeSource src;
    REQUIRE(src.open(composite, getVideoStandard(VideoStandard::NTSC),
                     SampleEncoding::CVBS_U10_4FSC, SignalState::STANDARD_TBC_LOCKED));
    REQUIRE(src.isSourceValid());
    REQUIRE(src.getNumberOfAvailableFields() == 1);

    auto data = src.getVideoField(1);
    REQUIRE(data.size() == samplesPerField);
    REQUIRE(data[0] == 282 * 64);
    REQUIRE(data[samplesPerField - 1] == 282 * 64);
    return 0;
}

int testChromaPlaneRead() {
    using namespace chd::format;
    using Plane = chd::reader::CvbsCompositeSource::Plane;
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string cPath = (tmpDir / "yc.cvbsc").string();

    // One NTSC field of chroma at value 600 (excursion of +88 from centre
    // 512). A chroma-plane read puts the excursion on blanking:
    //   240 × 64 + (600 − 512) × 64 = 15360 + 5632 = 20992.
    const size_t samplesPerField = 910 * 263;
    REQUIRE(writeUniformSamples(cPath, samplesPerField, 600));

    chd::reader::CvbsCompositeSource src;
    REQUIRE(src.open(cPath, getVideoStandard(VideoStandard::NTSC),
                     SampleEncoding::CVBS_U10_4FSC, SignalState::STANDARD_TBC_LOCKED,
                     std::nullopt, FrameLayout::UNKNOWN, std::nullopt, std::nullopt,
                     Plane::CHROMA));
    auto data = src.getVideoField(1);
    REQUIRE(data.size() == samplesPerField);
    REQUIRE(src.parameters().blanking16bIre == 15360);
    REQUIRE(data[0] == 15360 + 5632);
    REQUIRE(data[samplesPerField - 1] == 15360 + 5632);
    return 0;
}

// Write frameCount native frames where every sample of frame line L (0-based)
// holds the value base + L + frameIndex, so the conform's row mapping is
// observable. extraSamplesPerFrame appends PAL's 4 leftover samples after the
// uniform lines, valued base + linesPerFrame + frameIndex (continuing the
// numbering).
bool writeNativeFrames(const std::string &path, int32_t samplesPerLine,
                       int32_t linesPerFrame, int32_t frameCount,
                       int32_t extraSamplesPerFrame = 0, int32_t base = 0) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    std::vector<int16_t> line(static_cast<size_t>(samplesPerLine));
    for (int32_t frame = 0; frame < frameCount; ++frame) {
        for (int32_t l = 0; l < linesPerFrame; ++l) {
            std::fill(line.begin(), line.end(), static_cast<int16_t>(base + l + frame));
            out.write(reinterpret_cast<const char *>(line.data()),
                      static_cast<std::streamsize>(line.size() * sizeof(int16_t)));
        }
        std::vector<int16_t> extras(static_cast<size_t>(extraSamplesPerFrame),
                                    static_cast<int16_t>(base + linesPerFrame + frame));
        out.write(reinterpret_cast<const char *>(extras.data()),
                  static_cast<std::streamsize>(extras.size() * sizeof(int16_t)));
    }
    return out.good();
}

int testResolveFrameLayout() {
    using namespace chd::format;
    const auto &ntsc = getVideoStandard(VideoStandard::NTSC);
    const auto &palm = getVideoStandard(VideoStandard::PAL_M);
    const int64_t nativeBytes = ntsc.samplesPerFrame * 2;                  // 955,500
    const int64_t packedBytes = fieldPackedSamplesPerFrame(ntsc) * 2;      // 957,320

    // Unambiguous sizes resolve by modulo alone.
    REQUIRE(resolveFrameLayout(FrameLayout::UNKNOWN, ntsc, SignalState::STANDARD_TBC_LOCKED,
                               2, nativeBytes * 3) == FrameLayout::FRAME_NATIVE);
    REQUIRE(resolveFrameLayout(FrameLayout::UNKNOWN, ntsc, SignalState::STANDARD_TBC_LOCKED,
                               2, packedBytes * 3) == FrameLayout::FIELD_RASTER);
    REQUIRE(resolveFrameLayout(FrameLayout::UNKNOWN, palm, SignalState::STANDARD_TBC_LOCKED,
                               2, palm.samplesPerFrame * 2LL * 3) == FrameLayout::FRAME_NATIVE);

    // 526 native frames and 525 field-packed frames are the same byte count;
    // the sidecar frame count disambiguates, a tie falls back to FIELD_RASTER.
    const int64_t collision = nativeBytes * 526;
    REQUIRE(collision == packedBytes * 525);
    REQUIRE(resolveFrameLayout(FrameLayout::UNKNOWN, ntsc, SignalState::STANDARD_TBC_LOCKED,
                               2, collision) == FrameLayout::FIELD_RASTER);
    REQUIRE(resolveFrameLayout(FrameLayout::UNKNOWN, ntsc, SignalState::STANDARD_TBC_LOCKED,
                               2, collision, 526) == FrameLayout::FRAME_NATIVE);
    REQUIRE(resolveFrameLayout(FrameLayout::UNKNOWN, ntsc, SignalState::STANDARD_TBC_LOCKED,
                               2, collision, 525) == FrameLayout::FIELD_RASTER);

    // Truncated files match neither total: fall back to FIELD_RASTER.
    REQUIRE(resolveFrameLayout(FrameLayout::UNKNOWN, ntsc, SignalState::STANDARD_TBC_LOCKED,
                               2, nativeBytes + 128) == FrameLayout::FIELD_RASTER);

    // Frame totals are not normative without TBC at the standard rate.
    REQUIRE(resolveFrameLayout(FrameLayout::UNKNOWN, ntsc, SignalState::STANDARD_RAW,
                               2, nativeBytes) == FrameLayout::FIELD_RASTER);
    REQUIRE(resolveFrameLayout(FrameLayout::UNKNOWN, ntsc, SignalState::NONSTANDARD_TBC_LOCKED,
                               2, nativeBytes) == FrameLayout::FIELD_RASTER);

    // An explicit override always wins.
    REQUIRE(resolveFrameLayout(FrameLayout::FRAME_NATIVE, ntsc, SignalState::STANDARD_RAW,
                               2, packedBytes) == FrameLayout::FRAME_NATIVE);
    return 0;
}

int testFrameNativeNtscConform() {
    using namespace chd::format;
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "native_ntsc.cvbs").string();
    REQUIRE(writeNativeFrames(composite, 910, 525, 2));

    chd::reader::CvbsCompositeSource src;
    REQUIRE(src.open(composite, getVideoStandard(VideoStandard::NTSC),
                     SampleEncoding::CVBS_U10_4FSC, SignalState::STANDARD_TBC_LOCKED));
    REQUIRE(src.frameLayout() == FrameLayout::FRAME_NATIVE);
    REQUIRE(src.getNumberOfAvailableFields() == 4);
    REQUIRE(src.getFieldLength() == 910 * 263);

    // Flat cut: the first field buffer is temporal lines 0..262, the second
    // is lines 263..524 with its final row entirely padding (NTSC blanking =
    // 240 x 64 canonical), matching ld-chroma-encoder's field layout.
    const auto f1 = src.getVideoField(1);
    REQUIRE(f1.size() == static_cast<size_t>(910 * 263));
    REQUIRE(f1[0] == 0);
    REQUIRE(f1[261 * 910] == 261 * 64);
    REQUIRE(f1[262 * 910] == 262 * 64);
    REQUIRE(f1[263 * 910 - 1] == 262 * 64);

    const auto f2 = src.getVideoField(2);
    REQUIRE(f2[0] == 263 * 64);
    REQUIRE(f2[261 * 910] == 524 * 64);
    REQUIRE(f2[262 * 910] == 240 * 64);
    REQUIRE(f2[263 * 910 - 1] == 240 * 64);

    // Synthetic data has no sync to measure, so the alignment falls back to
    // sync-start. Burst windows keep the ld-decode convention; the active-video
    // crop is SMPTE ST 244's digital active line for a 0H row (125..892 incl).
    REQUIRE(src.parameters().colourBurstStart == 75);
    REQUIRE(src.parameters().colourBurstEnd   == 95);
    REQUIRE(src.parameters().activeVideoStart == 125);
    REQUIRE(src.parameters().activeVideoEnd   == 893);

    // Second frame's fields carry the +1 frame offset.
    REQUIRE(src.getVideoField(3)[0] == 1 * 64);
    REQUIRE(src.getVideoField(4)[0] == 264 * 64);

    // Line-range requests (1-based): the pad row alone, and a range
    // straddling real data + pad.
    const auto padOnly = src.getVideoField(2, 263, 263);
    REQUIRE(padOnly.size() == 910);
    REQUIRE(padOnly[0] == 240 * 64);
    const auto straddle = src.getVideoField(2, 262, 263);
    REQUIRE(straddle.size() == 2 * 910);
    REQUIRE(straddle[0] == 524 * 64);
    REQUIRE(straddle[910] == 240 * 64);
    const auto f1last = src.getVideoField(1, 263, 263);
    REQUIRE(f1last.size() == 910);
    REQUIRE(f1last[0] == 262 * 64);
    return 0;
}

int testFrameNativePalConform() {
    using namespace chd::format;
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "native_pal.cvbs").string();
    // 625 uniform 1135-sample lines plus the 4 leftover samples per frame
    // (valued 625 + frameIndex): exactly 709,379 samples/frame.
    REQUIRE(writeNativeFrames(composite, 1135, 625, 2, 4));

    chd::reader::CvbsCompositeSource src;
    REQUIRE(src.open(composite, getVideoStandard(VideoStandard::PAL),
                     SampleEncoding::CVBS_U10_4FSC, SignalState::STANDARD_TBC_LOCKED));
    REQUIRE(src.frameLayout() == FrameLayout::FRAME_NATIVE);
    REQUIRE(src.parameters().isSubcarrierLocked == true);
    REQUIRE(src.getNumberOfAvailableFields() == 4);
    REQUIRE(src.getFieldLength() == 1135 * 313);

    // Synthetic data has no sync to measure: sync-start fallback. Active-video
    // crop is EBU Tech 3280-E's digital active line for a 0H row (177..1124).
    REQUIRE(src.parameters().colourBurstStart == 98);
    REQUIRE(src.parameters().colourBurstEnd   == 138);
    REQUIRE(src.parameters().activeVideoStart == 177);
    REQUIRE(src.parameters().activeVideoEnd   == 1125);

    // Flat cut: first field = temporal lines 0..312; second field = lines
    // 313..624, then the 4 leftover samples at the start of its final row
    // with 1131 blanking samples (PAL blanking = 256 x 64) behind them.
    const auto f1 = src.getVideoField(1);
    REQUIRE(f1.size() == static_cast<size_t>(1135 * 313));
    REQUIRE(f1[0] == 0);
    REQUIRE(f1[312 * 1135] == 312 * 64);
    REQUIRE(f1[313 * 1135 - 1] == 312 * 64);

    const auto f2 = src.getVideoField(2);
    REQUIRE(f2[0] == 313 * 64);
    REQUIRE(f2[311 * 1135] == 624 * 64);
    REQUIRE(f2[312 * 1135 + 0] == 625 * 64);
    REQUIRE(f2[312 * 1135 + 3] == 625 * 64);
    REQUIRE(f2[312 * 1135 + 4] == 256 * 64);
    REQUIRE(f2[313 * 1135 - 1] == 256 * 64);

    // Second frame indexing across the odd 709,379-sample stride.
    REQUIRE(src.getVideoField(3)[0] == 1 * 64);
    REQUIRE(src.getVideoField(4)[312 * 1135] == 626 * 64);

    // The leftover row alone, via a line-range request.
    const auto tail = src.getVideoField(2, 313, 313);
    REQUIRE(tail.size() == 1135);
    REQUIRE(tail[3] == 625 * 64);
    REQUIRE(tail[4] == 256 * 64);
    return 0;
}

// Bit-exact validation of the PAL frame-native conform against a real
// ld-chroma-encoder subcarrier-locked TBC (tbc-tools). Point
// CHD_TEST_PAL_SCLOCKED_TBC at the .tbc; skipped when unset. The TBC pair
// concatenation IS the native stream plus 1131 dummy samples, so flattening
// each 710,510-sample pair to its first 709,379 samples and conforming back
// must reproduce the encoder's buffers exactly over the native extent. The
// dummy tail is excluded: the encoder synthesises a blanking line there,
// while the conform pads with the blanking level; neither is signal.
int testEncoderScLockedValidation() {
    using namespace chd::format;
    const char *tbcPath = std::getenv("CHD_TEST_PAL_SCLOCKED_TBC");
    if (tbcPath == nullptr) {
        std::cout << "  (encoder scLocked validation skipped: "
                     "CHD_TEST_PAL_SCLOCKED_TBC not set)\n";
        return 0;
    }

    constexpr int64_t kPairSamples   = 710510;
    constexpr int64_t kNativeSamples = 709379;
    constexpr int64_t kFieldSamples  = 1135 * 313;   // 355,255
    constexpr int64_t kOddFieldReal  = kNativeSamples - kFieldSamples;  // 354,124

    std::ifstream tbc(tbcPath, std::ios::binary);
    REQUIRE(tbc.is_open());
    tbc.seekg(0, std::ios::end);
    const int64_t tbcBytes = tbc.tellg();
    tbc.seekg(0, std::ios::beg);
    REQUIRE(tbcBytes % (kPairSamples * 2) == 0);
    const int64_t numFrames = tbcBytes / (kPairSamples * 2);
    std::vector<uint16_t> tbcData(static_cast<size_t>(tbcBytes / 2));
    tbc.read(reinterpret_cast<char *>(tbcData.data()), tbcBytes);
    REQUIRE(tbc.gcount() == tbcBytes);

    // Flatten: native stream = the first 709,379 samples of each field pair.
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string native = (tmpDir / "flattened_sclocked.cvbs").string();
    {
        std::ofstream out(native, std::ios::binary);
        REQUIRE(out.is_open());
        for (int64_t frame = 0; frame < numFrames; ++frame) {
            out.write(reinterpret_cast<const char *>(tbcData.data() + frame * kPairSamples),
                      kNativeSamples * 2);
        }
        REQUIRE(out.good());
    }

    chd::reader::CvbsCompositeSource src;
    // Surface the horizontal-alignment measurement in the validation output.
    enableDebugLogging();
    const bool opened = src.open(native, getVideoStandard(VideoStandard::PAL),
                                 SampleEncoding::CVBS_U16_4FSC,
                                 SignalState::STANDARD_TBC_LOCKED);
    disableDebugLogging();
    REQUIRE(opened);
    REQUIRE(src.frameLayout() == FrameLayout::FRAME_NATIVE);
    REQUIRE(src.getNumberOfAvailableFields() == numFrames * 2);
    REQUIRE(src.parameters().isSubcarrierLocked == true);
    // The derived burst window equals what the encoder wrote into its own
    // sidecar for this capture (capture table has 109/149). The active-video
    // crop is now EBU Tech 3280-E's digital active line (187..1134, 948 wide),
    // deliberately wider than the encoder's own 200..1121 picture crop: the
    // synthesized default follows the standard's digital active line, not the
    // encoder's tighter crop.
    REQUIRE(src.parameters().colourBurstStart == 109);
    REQUIRE(src.parameters().colourBurstEnd   == 149);
    REQUIRE(src.parameters().activeVideoStart == 187);
    REQUIRE(src.parameters().activeVideoEnd   == 1135);

    int64_t compared = 0;
    for (int64_t f = 0; f < numFrames * 2; ++f) {
        const auto data = src.getVideoField(static_cast<int32_t>(f + 1));
        REQUIRE(data.size() == static_cast<size_t>(kFieldSamples));
        const uint16_t *ref = tbcData.data() + (f / 2) * kPairSamples + (f % 2) * kFieldSamples;
        const int64_t realExtent = (f % 2 == 0) ? kFieldSamples : kOddFieldReal;
        REQUIRE(std::memcmp(data.data(), ref, static_cast<size_t>(realExtent) * 2) == 0);
        for (int64_t i = realExtent; i < kFieldSamples; ++i) {
            REQUIRE(data[static_cast<size_t>(i)] == 0x4000);  // PAL blanking pad
        }
        compared += realExtent;
    }

    // Decode smoke through the ABI: the conformed fields must survive the
    // FIR and Transform PAL pipelines (including the gated 2-sample
    // inter-field shift that runs before the transform's filterFields).
    chd_video_params_t params{};
    params.standard     = CHD_STD_PAL;
    params.encoding     = CHD_ENC_CVBS_U16_4FSC;
    params.signal_state = CHD_SIG_STANDARD_TBC_LOCKED;
    chd_video_t *v = nullptr;
    REQUIRE(chd_video_open_composite(native.c_str(), nullptr, &params, &v) == CHD_OK);
    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    REQUIRE(info.layout == CHD_FRAME_LAYOUT_FRAME_NATIVE);
    REQUIRE(info.is_subcarrier_locked == 1);
    const chd_decoder_kind_t kinds[] = {CHD_DEC_PAL_2D, CHD_DEC_TRANSFORM_2D,
                                        CHD_DEC_TRANSFORM_3D};
    for (const auto kind : kinds) {
        chd_decoder_t *dec = nullptr;
        REQUIRE(chd_decoder_create(v, kind, &dec) == CHD_OK);
        REQUIRE(chd_decoder_commit(dec) == CHD_OK);
        chd_frame_t *frame = nullptr;
        REQUIRE(chd_decode_frame(dec, 1, &frame) == CHD_OK);
        chd_frame_free(frame);
        chd_decoder_free(dec);
    }
    chd_video_free(v);

    std::cout << "  encoder scLocked validation: " << numFrames << " frames, "
              << compared << " samples bit-exact + decode smoke (FIR + Transform)\n";
    return 0;
}

int testFrameNativePalMAbi() {
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "native_palm.cvbs").string();
    const std::string meta      = (tmpDir / "native_palm.meta").string();
    REQUIRE(writeNativeFrames(composite, 909, 525, 1));
    REQUIRE(writeMetaSidecar(meta, "PAL_M", "CVBS_U10_4FSC", "STANDARD_TBC_LOCKED", "composite"));

    chd_video_t *v = nullptr;
    REQUIRE(chd_video_open_composite(composite.c_str(), meta.c_str(), nullptr, &v) == CHD_OK);
    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    REQUIRE(info.standard          == CHD_STD_PAL_M);
    REQUIRE(info.layout            == CHD_FRAME_LAYOUT_FRAME_NATIVE);
    REQUIRE(info.samples_per_frame == 477225);
    REQUIRE(info.field_width       == 909);
    REQUIRE(info.field_height      == 263);
    REQUIRE(info.num_frames        == 1);
    REQUIRE(info.is_subcarrier_locked == 0);
    chd_video_free(v);
    return 0;
}

int testPalFrameNativeAbi() {
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "native_pal_abi.cvbs").string();
    const std::string meta      = (tmpDir / "native_pal_abi.meta").string();
    REQUIRE(writeUniformSamples(composite, 709379, 256));
    REQUIRE(writeMetaSidecar(meta, "PAL", "CVBS_U10_4FSC", "STANDARD_TBC_LOCKED", "composite"));

    chd_video_t *v = nullptr;
    REQUIRE(chd_video_open_composite(composite.c_str(), meta.c_str(), nullptr, &v) == CHD_OK);
    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    REQUIRE(info.standard             == CHD_STD_PAL);
    REQUIRE(info.layout               == CHD_FRAME_LAYOUT_FRAME_NATIVE);
    REQUIRE(info.samples_per_frame    == 709379);
    REQUIRE(info.field_width          == 1135);
    REQUIRE(info.field_height         == 313);
    REQUIRE(info.num_frames           == 1);
    REQUIRE(info.is_subcarrier_locked == 1);
    chd_video_free(v);
    return 0;
}

int testSubcarrierLockDerivationAndMerge() {
    // A field-raster PAL .cvbs in STANDARD_TBC_LOCKED state is
    // line-locked by default (burst lock is not lattice lock); the caller
    // override marks the encoder-style subcarrier-locked raster, and merges
    // even though a sidecar is present.
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "raster_pal.cvbs").string();
    const std::string meta      = (tmpDir / "raster_pal.meta").string();
    REQUIRE(writeUniformSamples(composite, 1135 * 313 * 2, 256));
    REQUIRE(writeMetaSidecar(meta, "PAL", "CVBS_U10_4FSC", "STANDARD_TBC_LOCKED", "composite"));

    chd_video_t *v = nullptr;
    REQUIRE(chd_video_open_composite(composite.c_str(), meta.c_str(), nullptr, &v) == CHD_OK);
    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    REQUIRE(info.layout               == CHD_FRAME_LAYOUT_FIELD_RASTER);
    REQUIRE(info.is_subcarrier_locked == 0);
    REQUIRE(info.samples_per_frame    == 709379);
    // Default crop is EBU Tech 3280-E's digital active line, positioned for a
    // sync-start (0H) row: 177..1124 inclusive, 948 samples.
    REQUIRE(info.first_active_sample  == 177);
    REQUIRE(info.last_active_sample   == 1124);
    REQUIRE(info.is_first_field_first == 1);
    chd_video_free(v);

    chd_video_params_t params{};
    params.is_subcarrier_locked = 1;
    v = nullptr;
    REQUIRE(chd_video_open_composite(composite.c_str(), meta.c_str(), &params, &v) == CHD_OK);
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    REQUIRE(info.layout               == CHD_FRAME_LAYOUT_FIELD_RASTER);
    REQUIRE(info.is_subcarrier_locked == 1);
    // An encoder-style scLocked raster switches to blanking-start windows;
    // the digital active line then runs from the first blanking sample to the
    // row end: 187..1134 inclusive, 948 samples.
    REQUIRE(info.first_active_sample  == 187);
    REQUIRE(info.last_active_sample   == 1134);
    chd_video_free(v);

    // Field order merges from the override too (no `.meta` column exists).
    chd_video_params_t swapped{};
    swapped.is_second_field_first = 1;
    v = nullptr;
    REQUIRE(chd_video_open_composite(composite.c_str(), meta.c_str(), &swapped, &v) == CHD_OK);
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    REQUIRE(info.is_first_field_first == 0);
    chd_video_free(v);
    return 0;
}

int testS16_4FscAndSchemaVersions() {
    using namespace chd::format;
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "s16_4fsc.cvbs").string();
    const std::string meta      = (tmpDir / "s16_4fsc.meta").string();

    // NTSC white in CVBS_S16_4FSC: (800 - 240) x 32 = 17920 on disk.
    REQUIRE(writeUniformSamples(composite, 910 * 263 * 2, 17920));
    REQUIRE(writeMetaSidecar(meta, "NTSC", "CVBS_S16_4FSC", "STANDARD_TBC_LOCKED",
                             "composite", kCvbsSchemaV10));

    chd_video_t *v = nullptr;
    REQUIRE(chd_video_open_composite(composite.c_str(), meta.c_str(), nullptr, &v) == CHD_OK);
    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    REQUIRE(info.encoding == CHD_ENC_CVBS_S16_4FSC);
    REQUIRE(info.num_frames == 1);
    chd_video_free(v);

    // The pre-v1.4.0 spelling names the same encoding. It cannot be told apart
    // by user_version (the rename did not bump it), so it resolves whatever
    // schema revision declares it.
    REQUIRE(writeMetaSidecar(meta, "NTSC", "CVBS_S16_FSC", "STANDARD_TBC_LOCKED",
                             "composite", kCvbsSchemaV8));
    REQUIRE(chd_video_open_composite(composite.c_str(), meta.c_str(), nullptr, &v) == CHD_OK);
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    REQUIRE(info.encoding == CHD_ENC_CVBS_S16_4FSC);
    chd_video_free(v);

    // The blanking-offset conversion runs with the standard's own blanking.
    chd::reader::CvbsCompositeSource src;
    REQUIRE(src.open(composite, getVideoStandard(VideoStandard::NTSC),
                     SampleEncoding::CVBS_S16_4FSC, SignalState::STANDARD_TBC_LOCKED));
    REQUIRE(src.getVideoField(1)[0] == 800 * 64);
    return 0;
}

int testResolveFrameNativeAlignment() {
    using namespace chd::format;
    const auto &pal  = getVideoStandard(VideoStandard::PAL);
    const auto &ntsc = getVideoStandard(VideoStandard::NTSC);
    // Encoder-flattened data: 0H near the blanking-start position.
    REQUIRE(resolveFrameNativeAlignment(pal, 9.9, "t") == HorizontalAlignment::BLANKING_START);
    REQUIRE(resolveFrameNativeAlignment(ntsc, 16.4, "t") == HorizontalAlignment::BLANKING_START);
    // Real TPG21 captures: 0H at the row start, wrapping just below rowWidth.
    REQUIRE(resolveFrameNativeAlignment(pal, 1134.85, "t") == HorizontalAlignment::SYNC_START);
    REQUIRE(resolveFrameNativeAlignment(ntsc, 909.4, "t") == HorizontalAlignment::SYNC_START);
    REQUIRE(resolveFrameNativeAlignment(ntsc, 0.6, "t") == HorizontalAlignment::SYNC_START);
    // Active-start and unclassifiable cuts warn and serve sync-start.
    REQUIRE(resolveFrameNativeAlignment(pal, 957.5, "t") == HorizontalAlignment::SYNC_START);
    REQUIRE(resolveFrameNativeAlignment(ntsc, 500.0, "t") == HorizontalAlignment::SYNC_START);
    REQUIRE(resolveFrameNativeAlignment(pal, std::nullopt, "t") == HorizontalAlignment::SYNC_START);
    return 0;
}

// Frame-native file whose rows carry a real sync pulse at the row start,
// like the TPG21 reference captures: the measurement must select sync-start
// windows end to end.
int testFrameNativeSyncStartWindows() {
    using namespace chd::format;
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "syncstart_ntsc.cvbs").string();
    {
        std::ofstream out(composite, std::ios::binary);
        REQUIRE(out.is_open());
        std::vector<int16_t> row(910, 240);          // blanking
        std::fill(row.begin(), row.begin() + 84, 16);  // sync pulse at row start
        for (int32_t l = 0; l < 525 * 2; ++l) {
            out.write(reinterpret_cast<const char *>(row.data()),
                      static_cast<std::streamsize>(row.size() * sizeof(int16_t)));
        }
        REQUIRE(out.good());
    }

    chd::reader::CvbsCompositeSource src;
    REQUIRE(src.open(composite, getVideoStandard(VideoStandard::NTSC),
                     SampleEncoding::CVBS_U10_4FSC, SignalState::STANDARD_TBC_LOCKED));
    REQUIRE(src.frameLayout() == FrameLayout::FRAME_NATIVE);
    REQUIRE(src.parameters().colourBurstStart == 75);
    REQUIRE(src.parameters().activeVideoStart == 125);
    REQUIRE(src.parameters().activeVideoEnd   == 893);
    return 0;
}

int testMeasureRowZeroH() {
    using namespace chd::format;
    const auto &pal = getVideoStandard(VideoStandard::PAL);
    // Synthesize 1135-sample rows: blanking everywhere, a 84-sample sync
    // pulse whose falling edge sits at the requested position.
    auto makeRows = [](int32_t width, int32_t nRows, int32_t syncAt) {
        std::vector<uint16_t> rows(static_cast<size_t>(width) * nRows, 256 * 64);
        for (int32_t r = 0; r < nRows; ++r) {
            for (int32_t i = 0; i < 84 && syncAt + i < width; ++i) {
                rows[static_cast<size_t>(r) * width + syncAt + i] = 4 * 64;
            }
        }
        return rows;
    };

    const auto atTen = makeRows(1135, 50, 10);
    auto measured = measureRowZeroH(atTen.data(), 50, 1135, pal.levels);
    REQUIRE(measured.has_value());
    REQUIRE(*measured > 9.0 && *measured < 10.5);

    const auto atSpecOrigin = makeRows(1135, 50, 957);
    measured = measureRowZeroH(atSpecOrigin.data(), 50, 1135, pal.levels);
    REQUIRE(measured.has_value());
    REQUIRE(*measured > 956.0 && *measured < 957.5);

    // No sync structure at all (uniform data): unmeasurable, not a guess.
    const std::vector<uint16_t> flat(1135 * 50, 256 * 64);
    REQUIRE(!measureRowZeroH(flat.data(), 50, 1135, pal.levels).has_value());
    return 0;
}

int testMeasureNtscFieldBurstPolarity() {
    using namespace chd::format;
    constexpr int32_t width = 910, firstRow = 30, nRows = 12;
    constexpr int32_t blanking = 240 * 64, white = 800 * 64;
    constexpr int32_t burstStart = 75, burstEnd = 95;

    // Burst quad per h%4 as measured from the TPG21 hardware reference
    // capture (frame 0, field 1: the polarity the comb decodes correctly
    // with positive-on-even-lines == true). Sign alternates per line.
    auto makeField = [&](bool negate) {
        std::vector<uint16_t> rows(static_cast<size_t>(width) * nRows,
                                   static_cast<uint16_t>(blanking));
        constexpr int32_t quad[4] = {94, 60, -94, -60};
        for (int32_t r = 0; r < nRows; ++r) {
            const int32_t lineSign = (((firstRow + r) % 2) == 0) ? 1 : -1;
            const int32_t sign = negate ? -lineSign : lineSign;
            for (int32_t h = burstStart; h < burstEnd; ++h) {
                rows[static_cast<size_t>(r) * width + h] =
                    static_cast<uint16_t>(blanking + sign * quad[h % 4] * 64);
            }
        }
        return rows;
    };

    const auto fieldA = makeField(false);
    auto p = measureNtscFieldBurstPolarity(fieldA.data(), firstRow, nRows, width,
                                           burstStart, burstEnd, blanking, white);
    REQUIRE(p.has_value());
    REQUIRE(*p == true);

    const auto fieldB = makeField(true);
    p = measureNtscFieldBurstPolarity(fieldB.data(), firstRow, nRows, width,
                                      burstStart, burstEnd, blanking, white);
    REQUIRE(p.has_value());
    REQUIRE(*p == false);

    // Burst-free rows: unmeasurable, not a guess.
    const std::vector<uint16_t> flatField(static_cast<size_t>(width) * nRows,
                                          static_cast<uint16_t>(blanking));
    REQUIRE(!measureNtscFieldBurstPolarity(flatField.data(), firstRow, nRows, width,
                                           burstStart, burstEnd, blanking, white)
                 .has_value());
    return 0;
}

int testLayoutOverrideMerge() {
    // Force FIELD_RASTER onto a native-sized NTSC file through the override,
    // with the sidecar still supplying the preset triple.
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string composite = (tmpDir / "forced_raster.cvbs").string();
    const std::string meta      = (tmpDir / "forced_raster.meta").string();
    REQUIRE(writeNativeFrames(composite, 910, 525, 2));
    REQUIRE(writeMetaSidecar(meta, "NTSC", "CVBS_U10_4FSC", "STANDARD_TBC_LOCKED", "composite"));

    chd_video_params_t params{};
    params.layout = CHD_FRAME_LAYOUT_FIELD_RASTER;
    chd_video_t *v = nullptr;
    REQUIRE(chd_video_open_composite(composite.c_str(), meta.c_str(), &params, &v) == CHD_OK);
    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    REQUIRE(info.layout == CHD_FRAME_LAYOUT_FIELD_RASTER);
    // 955,500 samples/frame x 2 frames sliced as fixed 910 x 263 fields.
    REQUIRE(info.num_frames == (477750LL * 2) / (910 * 263) / 2);
    chd_video_free(v);
    return 0;
}

int testFrameNativeChromaPlane() {
    using namespace chd::format;
    using Plane = chd::reader::CvbsCompositeSource::Plane;
    fs::path tmpDir = fs::temp_directory_path() / "chd_phase_d_test";
    fs::create_directories(tmpDir);
    const std::string cPath = (tmpDir / "native.cvbsc").string();
    // A frame-native chroma plane: the line index rides on the centre (512 =
    // no excursion) so the conform mapping is observable on blanking, and
    // the row alignment comes from the caller (no sync to measure here).
    REQUIRE(writeNativeFrames(cPath, 910, 525, 1, 0, 512));

    chd::reader::CvbsCompositeSource src;
    REQUIRE(src.open(cPath, getVideoStandard(VideoStandard::NTSC),
                     SampleEncoding::CVBS_U10_4FSC, SignalState::STANDARD_TBC_LOCKED,
                     std::nullopt, FrameLayout::UNKNOWN, std::nullopt, std::nullopt,
                     Plane::CHROMA, HorizontalAlignment::BLANKING_START));
    REQUIRE(src.frameLayout() == FrameLayout::FRAME_NATIVE);
    REQUIRE(src.horizontalAlignment() == HorizontalAlignment::BLANKING_START);
    REQUIRE(src.getNumberOfAvailableFields() == 2);
    const int32_t blanking = src.parameters().blanking16bIre;
    const auto f1 = src.getVideoField(1);
    REQUIRE(f1[261 * 910] == blanking + 261 * 64);
    REQUIRE(f1[262 * 910] == blanking + 262 * 64);
    const auto f2 = src.getVideoField(2);
    REQUIRE(f2[0] == blanking + 263 * 64);
    // The dummy padding line is blanking (an excursion of zero).
    REQUIRE(f2[262 * 910] == blanking);
    return 0;
}

// Diagnostic hook: point CHD_TEST_CVBS_COMPOSITE at a real `.cvbs`
// (with its `.meta` alongside) to surface the layout resolution and the
// horizontal-alignment measurement for that capture.
int testRealCompositeDiagnostics() {
    const char *path = std::getenv("CHD_TEST_CVBS_COMPOSITE");
    if (path == nullptr) return 0;

    chd_video_t *v = nullptr;
    enableDebugLogging();
    const chd_status_t rc = chd_video_open_composite(path, nullptr, nullptr, &v);
    disableDebugLogging();
    REQUIRE(rc == CHD_OK);
    chd_video_info_t info{};
    REQUIRE(chd_video_get_info(v, &info) == CHD_OK);
    std::cout << "  real composite: layout=" << info.layout
              << " encoding=" << info.encoding
              << " frames=" << info.num_frames
              << " subcarrier_locked=" << info.is_subcarrier_locked
              << " active=" << info.first_active_sample << ".." << info.last_active_sample
              << "\n";

    // Decode smoke: the capture must survive its system's default pipeline.
    chd_decoder_t *dec = nullptr;
    REQUIRE(chd_decoder_create(v, CHD_DEC_AUTO, &dec) == CHD_OK);
    REQUIRE(chd_decoder_commit(dec) == CHD_OK);
    chd_frame_t *frame = nullptr;
    REQUIRE(chd_decode_frame(dec, 0, &frame) == CHD_OK);
    chd_frame_free(frame);
    chd_decoder_free(dec);
    chd_video_free(v);
    return 0;
}

}  // namespace

int main() {
    int rc = 0;
    rc |= testCompositeOpen();
    rc |= testCompositeMissingMeta();
    rc |= testCompositeUnknownPreset();
    rc |= testCompositeOverride();
    rc |= testCompositeSampleConversion();
    rc |= testChromaPlaneRead();
    rc |= testResolveFrameLayout();
    rc |= testFrameNativeNtscConform();
    rc |= testFrameNativePalConform();
    rc |= testFrameNativePalMAbi();
    rc |= testPalFrameNativeAbi();
    rc |= testSubcarrierLockDerivationAndMerge();
    rc |= testS16_4FscAndSchemaVersions();
    rc |= testMeasureRowZeroH();
    rc |= testMeasureNtscFieldBurstPolarity();
    rc |= testResolveFrameNativeAlignment();
    rc |= testFrameNativeSyncStartWindows();
    rc |= testLayoutOverrideMerge();
    rc |= testFrameNativeChromaPlane();
    rc |= testEncoderScLockedValidation();
    rc |= testRealCompositeDiagnostics();
    if (rc == 0) std::cout << "All CVBS reader tests passed.\n";
    return rc;
}
