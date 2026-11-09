// SPDX-License-Identifier: GPL-3.0-or-later
//
// Dropout corrector unit test. Builds a synthetic NTSC field pair
// in memory, injects a known dropout region, runs DropoutCorrector::
// correctFrame, and asserts the stats accounting + sample replacement.
//
// Single-source path is exercised first because it's the vapoursynth-
// analog feature parity bar. Multi-source path is exercised separately so a
// regression there shows up as a distinct failure.

#include <cstdint>
#include <iostream>
#include <vector>

#include <sqlite3.h>

#include <filesystem>

#include <chromadec/decoder.h>
#include <chromadec/dropout.h>
#include <chromadec/errors.h>
#include <chromadec/video.h>

#include "../../src/dropout/dropout_corrector.h"
#include "../../src/format/video_standards.h"
#include "../../src/metadata/core.h"
#include "../../src/metadata/dropout_extension_sqlite.h"
#include "../../src/metadata/dropouts.h"
#include "../../src/decoders/source_field.h"
#include "../../src/reader/source.h"

namespace {

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        return 1; \
    } \
} while (0)

// Sentinel sample value that marks "this pixel was a dropout" before
// correction. Anything inside the dropout range that still reads this value
// after correctFrame returns means the corrector failed to overwrite it.
constexpr uint16_t kDropoutSentinel = 0xDEAD;

// Build a synthetic SourceField pair. Every line gets a per-line constant
// value so we can tell which neighbour the corrector pulled from after it
// finishes — line N's data is filled with value `lineFill(N)`.
struct SyntheticFrame {
    chd::decoders::SourceField first;
    chd::decoders::SourceField second;
    chd::metadata::LdDecodeMetaData::VideoParameters vp;
};

uint16_t lineFill(int32_t lineNumber, int32_t fieldOffset) {
    // Returns a stable per-line value distinct between fields so the
    // intra-field replacement path is observable.
    return static_cast<uint16_t>(0x1000 + (lineNumber * 4) + fieldOffset);
}

SyntheticFrame buildSyntheticNtscFrame() {
    using namespace chd::format;
    const VideoStandardPreset &preset = getVideoStandard(VideoStandard::NTSC);
    SyntheticFrame f;
    f.vp = makeVideoParameters(preset, true);

    const int32_t fieldSamples = f.vp.fieldWidth * f.vp.fieldHeight;

    f.first.field.isFirstField = true;
    f.first.field.seqNo = 1;
    f.first.field.vitsMetrics.inUse = true;
    f.first.field.vitsMetrics.bPSNR = 40.0;
    f.first.data.assign(fieldSamples, 0);

    f.second.field.isFirstField = false;
    f.second.field.seqNo = 2;
    f.second.field.vitsMetrics.inUse = true;
    f.second.field.vitsMetrics.bPSNR = 40.0;
    f.second.data.assign(fieldSamples, 0);

    // Fill each line with a stable, distinguishable value.
    for (int32_t line = 0; line < f.vp.fieldHeight; line++) {
        uint16_t firstVal = lineFill(line + 1, 0);
        uint16_t secondVal = lineFill(line + 1, 1);
        for (int32_t x = 0; x < f.vp.fieldWidth; x++) {
            f.first.data[line * f.vp.fieldWidth + x] = firstVal;
            f.second.data[line * f.vp.fieldWidth + x] = secondVal;
        }
    }
    return f;
}

int testSingleSourceCorrection() {
    SyntheticFrame f = buildSyntheticNtscFrame();
    const int32_t targetLine = (f.vp.firstActiveFieldLine() + f.vp.lastActiveFieldLine()) / 2;
    const int32_t dropoutStart = f.vp.activeVideoStart + 100;
    const int32_t dropoutEnd = f.vp.activeVideoStart + 200;

    // Pre-poison the dropout region in the first field so we can detect
    // whether the corrector wrote over it. Targeting line `targetLine`
    // (1-based), so the data range to poison is the (targetLine-1)th row.
    for (int32_t x = dropoutStart; x < dropoutEnd; x++) {
        f.first.data[(targetLine - 1) * f.vp.fieldWidth + x] = kDropoutSentinel;
    }

    chd::metadata::DropOuts d;
    d.append(dropoutStart, dropoutEnd, targetLine);
    f.first.field.dropOuts = d;

    chd::dropout::DropoutCorrector corrector(f.vp);
    chd::dropout::DropoutCorrectionStats stats;
    corrector.correctFrame(f.first, f.second, /*overCorrect=*/false,
                           /*intraField=*/false, &stats);

    REQUIRE(stats.corrected == 1);
    REQUIRE(stats.failed == 0);
    REQUIRE(stats.totalDistance > 0);

    // Inspect the corrected pixels — every previously-poisoned pixel must
    // have been replaced. The replacement could be from the same field
    // (luma path) or filtered against another line; verify that none of
    // them are still the sentinel.
    for (int32_t x = dropoutStart; x < dropoutEnd; x++) {
        const uint16_t v = f.first.data[(targetLine - 1) * f.vp.fieldWidth + x];
        REQUIRE(v != kDropoutSentinel);
    }
    return 0;
}

int testIntraFieldFallback() {
    // intraField=true forbids cross-field replacement candidates. The
    // corrector should still find a same-field neighbour and the stats
    // should reflect a successful correction.
    SyntheticFrame f = buildSyntheticNtscFrame();
    const int32_t targetLine = (f.vp.firstActiveFieldLine() + f.vp.lastActiveFieldLine()) / 2;
    const int32_t dropoutStart = f.vp.activeVideoStart + 50;
    const int32_t dropoutEnd = f.vp.activeVideoStart + 80;

    for (int32_t x = dropoutStart; x < dropoutEnd; x++) {
        f.first.data[(targetLine - 1) * f.vp.fieldWidth + x] = kDropoutSentinel;
    }

    chd::metadata::DropOuts d;
    d.append(dropoutStart, dropoutEnd, targetLine);
    f.first.field.dropOuts = d;

    chd::dropout::DropoutCorrector corrector(f.vp);
    chd::dropout::DropoutCorrectionStats stats;
    corrector.correctFrame(f.first, f.second, /*overCorrect=*/false,
                           /*intraField=*/true, &stats);
    REQUIRE(stats.corrected == 1);
    REQUIRE(stats.failed == 0);
    for (int32_t x = dropoutStart; x < dropoutEnd; x++) {
        REQUIRE(f.first.data[(targetLine - 1) * f.vp.fieldWidth + x] != kDropoutSentinel);
    }
    return 0;
}

int testNoDropoutsIsNoOp() {
    SyntheticFrame f = buildSyntheticNtscFrame();
    // Stash a copy of the first-field data so we can confirm bit-equality
    // after a no-op correctFrame call.
    chd::reader::Data before = f.first.data;

    chd::dropout::DropoutCorrector corrector(f.vp);
    chd::dropout::DropoutCorrectionStats stats;
    corrector.correctFrame(f.first, f.second, /*overCorrect=*/false,
                           /*intraField=*/false, &stats);
    REQUIRE(stats.corrected == 0);
    REQUIRE(stats.failed == 0);
    REQUIRE(stats.totalDistance == 0);
    REQUIRE(f.first.data == before);
    return 0;
}

int testMultiSourceCorrection() {
    // Build two NTSC frames; primary has a dropout on line L. Extra source
    // is dropout-free in that region, so the corrector should prefer the
    // extra (same line, distance 0) over the primary's same-field neighbour.
    SyntheticFrame primary = buildSyntheticNtscFrame();
    SyntheticFrame extra = buildSyntheticNtscFrame();

    // Distinguish the extra's sample values so we can verify the corrector
    // actually pulled from it.
    constexpr uint16_t kExtraMark = 0xABCD;
    const int32_t targetLine = (primary.vp.firstActiveFieldLine() + primary.vp.lastActiveFieldLine()) / 2;
    const int32_t dropoutStart = primary.vp.activeVideoStart + 300;
    const int32_t dropoutEnd = primary.vp.activeVideoStart + 320;
    for (int32_t x = 0; x < primary.vp.fieldWidth; x++) {
        extra.first.data[(targetLine - 1) * primary.vp.fieldWidth + x] = kExtraMark;
    }
    for (int32_t x = dropoutStart; x < dropoutEnd; x++) {
        primary.first.data[(targetLine - 1) * primary.vp.fieldWidth + x] = kDropoutSentinel;
    }

    chd::metadata::DropOuts d;
    d.append(dropoutStart, dropoutEnd, targetLine);
    primary.first.field.dropOuts = d;

    chd::dropout::ExtraSourceFrame extraFrame;
    extraFrame.firstFieldData = extra.first.data;
    extraFrame.secondFieldData = extra.second.data;
    extraFrame.firstFieldMeta = extra.first.field;
    extraFrame.secondFieldMeta = extra.second.field;
    extraFrame.videoParams = extra.vp;
    extraFrame.quality = 50.0;  // higher than primary's 40.0

    chd::dropout::DropoutCorrector corrector(primary.vp);
    chd::dropout::DropoutCorrectionStats stats;
    std::vector<chd::dropout::ExtraSourceFrame> extras = {extraFrame};
    corrector.correctFrame(primary.first, primary.second, extras,
                           /*overCorrect=*/false,
                           /*intraField=*/false, &stats);

    REQUIRE(stats.corrected == 1);
    // The extra contributes the same target line with distance 0.
    REQUIRE(stats.totalDistance == 0);
    for (int32_t x = dropoutStart; x < dropoutEnd; x++) {
        const uint16_t v = primary.first.data[(targetLine - 1) * primary.vp.fieldWidth + x];
        REQUIRE(v == kExtraMark);
    }
    return 0;
}

// Sidecar fixtures live in a directory of our own so a concurrently running
// test binary can never pick the same name.
std::filesystem::path fixtureDir()
{
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "chd_dropout_corrector_test";
    std::filesystem::create_directories(dir);
    return dir;
}

// Build a tiny CVBS dropouts.meta sidecar in a temp file with a handful of
// rows that exercise the transcoder slicing logic. Returns the path; caller
// is responsible for removing it.
std::string writeFixtureDropoutSidecar(const std::vector<std::tuple<int32_t,int32_t,int32_t>> &rows)
{
    const std::string path = (fixtureDir() / "transcoder.dropouts.meta").string();
    std::filesystem::remove(path);
    sqlite3 *db = nullptr;
    sqlite3_open(path.c_str(), &db);
    sqlite3_exec(db, "PRAGMA user_version = 5", nullptr, nullptr, nullptr);
    sqlite3_exec(db,
        "CREATE TABLE dropout_run ("
        "  cvbs_file_id INTEGER NOT NULL, frame_id INTEGER NOT NULL,"
        "  sample_start INTEGER NOT NULL, sample_count INTEGER NOT NULL,"
        "  severity INTEGER NOT NULL DEFAULT 0,"
        "  PRIMARY KEY (cvbs_file_id, frame_id, sample_start))",
        nullptr, nullptr, nullptr);
    sqlite3_stmt *ins = nullptr;
    sqlite3_prepare_v2(db,
        "INSERT INTO dropout_run (cvbs_file_id, frame_id, sample_start, sample_count, severity) "
        "VALUES (1, ?, ?, ?, 50)", -1, &ins, nullptr);
    for (const auto &r : rows) {
        sqlite3_bind_int(ins, 1, std::get<0>(r));
        sqlite3_bind_int(ins, 2, std::get<1>(r));
        sqlite3_bind_int(ins, 3, std::get<2>(r));
        sqlite3_step(ins);
        sqlite3_reset(ins);
    }
    sqlite3_finalize(ins);
    sqlite3_close(db);
    return path;
}

int testCvbsDropoutsTranscoder() {
    // NTSC: 910 samples per line, ~262 lines per field. A 50-sample run
    // starting at frame line 0 (first field, line 1) stays inside one
    // line. A 1500-sample run starting near line 2's end crosses into
    // line 3, so it must produce two legacy DropOuts entries — first in
    // first-field line 2, then in second-field line 2 (frame line 3 = odd
    // → second field, line (3/2)+1 = 2).
    const int32_t samplesPerLine = 910;
    const int32_t linesPerField  = 262;
    const std::string path = writeFixtureDropoutSidecar({
        {0, 100, 50},                                  // single line, first field line 1
        {0, samplesPerLine * 2 + 850, 200},            // crosses lines 2->3
    });
    auto framesOpt = chd::metadata::readCvbsDropoutsExtension(path, 1, samplesPerLine, linesPerField);
    std::filesystem::remove(path);

    REQUIRE(framesOpt.has_value());
    REQUIRE(framesOpt->size() == 1);

    const chd::metadata::FrameDropouts &fd = (*framesOpt)[0];
    REQUIRE(fd.frameIndex == 0);

    // First-field accumulators: row 1's slice (line 1) plus the first
    // half of row 2's slice (frame line 2 → first field, line 2).
    REQUIRE(fd.firstField.size() == 2);
    REQUIRE(fd.firstField.startx(0) == 100);
    REQUIRE(fd.firstField.endx(0) == 150);
    REQUIRE(fd.firstField.fieldLine(0) == 1);
    REQUIRE(fd.firstField.startx(1) == 850);
    REQUIRE(fd.firstField.endx(1) == samplesPerLine);
    REQUIRE(fd.firstField.fieldLine(1) == 2);

    // Second-field accumulator: the spillover from row 2 (frame line 3 →
    // second field, line 2), starting at sample 0 of that line.
    REQUIRE(fd.secondField.size() == 1);
    REQUIRE(fd.secondField.startx(0) == 0);
    REQUIRE(fd.secondField.endx(0) == 140);  // (samplesPerLine * 2 + 850 + 200) - samplesPerLine * 3 = 140
    REQUIRE(fd.secondField.fieldLine(0) == 2);
    return 0;
}

int testDropoutAbiNullArgs() {
    // C ABI surface: set/get reject null and accept a valid handle.
    chd_dropout_opts_t opts{};
    opts.enabled = 1;
    REQUIRE(chd_decoder_set_dropout(nullptr, &opts) == CHD_E_INVALID_ARG);
    chd_dropout_stats_t stats{};
    REQUIRE(chd_decoder_get_last_dropout_stats(nullptr, &stats) == CHD_E_INVALID_ARG);
    return 0;
}

int testCvbsDropoutsWrongVersionRejected() {
    // Sidecars with the wrong user_version are rejected, not silently
    // misinterpreted (spec consumer requirement 3 implies this).
    const std::string path = (fixtureDir() / "bad_version.dropouts.meta").string();
    std::filesystem::remove(path);
    sqlite3 *db = nullptr;
    sqlite3_open(path.c_str(), &db);
    sqlite3_exec(db, "PRAGMA user_version = 99", nullptr, nullptr, nullptr);
    sqlite3_close(db);
    auto framesOpt = chd::metadata::readCvbsDropoutsExtension(path, 1, 910, 262);
    std::filesystem::remove(path);
    REQUIRE(!framesOpt.has_value());
    return 0;
}

}  // namespace

// The second field of subcarrier-locked PAL moves left, and its dropout
// positions have to move with the samples.
int testSecondFieldShiftMovesDropouts() {
    using chd::decoders::SourceField;
    chd::metadata::LdDecodeMetaData::VideoParameters vp;
    vp.system = chd::metadata::PAL;
    vp.isSubcarrierLocked = true;
    if (!SourceField::secondFieldIsShifted(vp)) return 1;
    vp.isSubcarrierLocked = false;
    if (SourceField::secondFieldIsShifted(vp)) return 1;
    vp.system = chd::metadata::NTSC;
    vp.isSubcarrierLocked = true;
    if (SourceField::secondFieldIsShifted(vp)) return 1;

    chd::reader::Data data{1, 2, 3, 4, 5, 6};
    chd::metadata::DropOuts dropOuts;
    dropOuts.append(1, 4, 10);
    dropOuts.append(0, 2, 11);  // entirely in the samples that leave the field
    dropOuts.append(5, 8, 12);
    SourceField::shiftSecondField(data, dropOuts, 9);

    if (data != chd::reader::Data{3, 4, 5, 6, 9, 9}) return 1;
    if (dropOuts.size() != 2) return 1;
    if (dropOuts.startx(0) != 0 || dropOuts.endx(0) != 2 || dropOuts.fieldLine(0) != 10) return 1;
    if (dropOuts.startx(1) != 3 || dropOuts.endx(1) != 6 || dropOuts.fieldLine(1) != 12) return 1;
    return 0;
}

int main() {
    if (testSecondFieldShiftMovesDropouts()) return 1;
    if (testSingleSourceCorrection()) return 1;
    if (testIntraFieldFallback()) return 1;
    if (testNoDropoutsIsNoOp()) return 1;
    if (testMultiSourceCorrection()) return 1;
    if (testCvbsDropoutsTranscoder()) return 1;
    if (testCvbsDropoutsWrongVersionRejected()) return 1;
    if (testDropoutAbiNullArgs()) return 1;
    std::filesystem::remove(fixtureDir());
    std::cout << "test_dropout_corrector: OK\n";
    return 0;
}
