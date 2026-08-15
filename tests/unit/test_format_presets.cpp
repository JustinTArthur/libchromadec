// SPDX-License-Identifier: GPL-3.0-or-later
//
// format/ DATA-table sanity tests. Verifies that preset lookup by
// name (the path the CVBS sidecar reader takes) returns the same constexpr
// rows as lookup by enum, and that the spec-normative sample counts /
// sample-level mappings made it into the table.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include "../../src/format/sample_encoding.h"
#include "../../src/format/signal_state.h"
#include "../../src/format/video_standards.h"

namespace {

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        return 1; \
    } \
} while (0)

int testVideoStandardLookup() {
    using namespace chd::format;
    REQUIRE(findVideoStandardByName("PAL") == &getVideoStandard(VideoStandard::PAL));
    REQUIRE(findVideoStandardByName("NTSC") == &getVideoStandard(VideoStandard::NTSC));
    REQUIRE(findVideoStandardByName("PAL_M") == &getVideoStandard(VideoStandard::PAL_M));
    REQUIRE(findVideoStandardByName("UNKNOWN") == nullptr);
    REQUIRE(findVideoStandardByName("ntsc") == nullptr);  // case-sensitive

    // Spec-normative sample counts (video-standard-presets.md "Exact frame size").
    REQUIRE(getVideoStandard(VideoStandard::PAL).samplesPerFrame == 709379);
    REQUIRE(getVideoStandard(VideoStandard::NTSC).samplesPerFrame == 477750);
    REQUIRE(getVideoStandard(VideoStandard::PAL_M).samplesPerFrame == 477225);

    // Colour-frame sequence lengths.
    REQUIRE(getVideoStandard(VideoStandard::PAL).colourFrameSequenceLength == 4);
    REQUIRE(getVideoStandard(VideoStandard::NTSC).colourFrameSequenceLength == 2);
    REQUIRE(getVideoStandard(VideoStandard::PAL_M).colourFrameSequenceLength == 4);

    // Sample level tables (in the 10-bit domain).
    REQUIRE(getVideoStandard(VideoStandard::PAL).levels.blanking == 256);
    REQUIRE(getVideoStandard(VideoStandard::PAL).levels.black == 256);   // PAL: no setup, black = blanking
    REQUIRE(getVideoStandard(VideoStandard::NTSC).levels.blanking == 240);
    REQUIRE(getVideoStandard(VideoStandard::NTSC).levels.black == 282);  // NTSC: +7.5 IRE setup
    REQUIRE(getVideoStandard(VideoStandard::PAL).levels.peak == 1019);
    REQUIRE(getVideoStandard(VideoStandard::NTSC).levels.peak == 1019);
    return 0;
}

int testSampleEncodingLookup() {
    using namespace chd::format;
    REQUIRE(findSampleEncodingByName("CVBS_U10_4FSC")->encoding == SampleEncoding::CVBS_U10_4FSC);
    REQUIRE(findSampleEncodingByName("CVBS_U16_4FSC")->encoding == SampleEncoding::CVBS_U16_4FSC);
    REQUIRE(findSampleEncodingByName("CVBS_TPG21_4FSC")->encoding == SampleEncoding::CVBS_TPG21_4FSC);
    REQUIRE(findSampleEncodingByName("RAW_S16_28M")->encoding == SampleEncoding::RAW_S16_28M);
    REQUIRE(findSampleEncodingByName("RAW_S16_40M")->encoding == SampleEncoding::RAW_S16_40M);
    REQUIRE(findSampleEncodingByName("CVBS_S16_4FSC")->encoding == SampleEncoding::CVBS_S16_4FSC);
    // Pre-spec-v1.4.0 spelling of the same encoding.
    REQUIRE(findSampleEncodingByName("CVBS_S16_FSC")->encoding == SampleEncoding::CVBS_S16_4FSC);
    REQUIRE(findSampleEncodingByName("does-not-exist") == nullptr);

    REQUIRE(getSampleEncoding(SampleEncoding::CVBS_U10_4FSC).hasStandardAmplitudeMapping);
    REQUIRE(!getSampleEncoding(SampleEncoding::RAW_S16_28M).hasStandardAmplitudeMapping);
    return 0;
}

int testSampleConversion() {
    using namespace chd::format;
    // Only CVBS_S16_4FSC consumes the blanking argument; pass the PAL value
    // (256) for the others to prove it is ignored.
    // CVBS_U10_4FSC: raw is 10-bit value; × 64 to get canonical TBC domain.
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_U10_4FSC, 256, 256) == 16384);
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_U10_4FSC, 0, 256) == 0);
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_U10_4FSC, 1019, 256) == 65216);

    // CVBS_U16_4FSC: raw is already the 10-bit × 64 value in unsigned 16-bit.
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_U16_4FSC, static_cast<int16_t>(16384), 256) == 16384);
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_U16_4FSC, static_cast<int16_t>(-32768), 256) == 32768);

    // CVBS_TPG21_4FSC: int16 = (val10 - 508) × 64; blanking 256 → int16 = -16128.
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_TPG21_4FSC, -16128, 256) == 16384);
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_TPG21_4FSC, 0, 256) == 32512);  // val10 = 508

    // CVBS_S16_4FSC: int16 = (val10 - blanking10) × 32; the offset follows the
    // standard's blanking (spec encoded-level examples).
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_S16_4FSC, 0, 256) == 16384);
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_S16_4FSC, -8064, 256) == 4 * 64);
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_S16_4FSC, 18816, 256) == 844 * 64);
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_S16_4FSC, 0, 240) == 15360);
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_S16_4FSC, 1344, 240) == 282 * 64);
    REQUIRE(convertCompositeSampleToCanonical(SampleEncoding::CVBS_S16_4FSC, 17920, 240) == 800 * 64);

    // Chroma centred at 512 → centred excursion × 64.
    REQUIRE(convertChromaSampleToCenteredCanonical(SampleEncoding::CVBS_U10_4FSC, 512, 256) == 0);
    REQUIRE(convertChromaSampleToCenteredCanonical(SampleEncoding::CVBS_U10_4FSC, 600, 256) == (600 - 512) * 64);
    REQUIRE(convertChromaSampleToCenteredCanonical(SampleEncoding::CVBS_U10_4FSC, 400, 256) == (400 - 512) * 64);

    // CVBS_U16_4FSC chroma: 32768 in u16 is chroma zero.
    REQUIRE(convertChromaSampleToCenteredCanonical(SampleEncoding::CVBS_U16_4FSC,
                                                   static_cast<int16_t>(32768), 256) == 0);

    // CVBS_S16_4FSC chroma: zero maps to (512 - blanking10) × 32 in int16
    // (8192 for PAL, 8704 for NTSC/PAL_M, per the spec examples).
    REQUIRE(convertChromaSampleToCenteredCanonical(SampleEncoding::CVBS_S16_4FSC, 8192, 256) == 0);
    REQUIRE(convertChromaSampleToCenteredCanonical(SampleEncoding::CVBS_S16_4FSC, 8704, 240) == 0);
    REQUIRE(convertChromaSampleToCenteredCanonical(SampleEncoding::CVBS_S16_4FSC,
                                                   8192 + 88 * 32, 256) == 88 * 64);
    return 0;
}

int testSignalStateLookup() {
    using namespace chd::format;
    REQUIRE(findSignalStateByName("STANDARD_TBC_LOCKED")->state == SignalState::STANDARD_TBC_LOCKED);
    REQUIRE(findSignalStateByName("STANDARD_RAW")->burstLocked == false);
    REQUIRE(findSignalStateByName("STANDARD_RAW")->tbcApplied == false);
    REQUIRE(findSignalStateByName("NONSTANDARD_TBC_LOCKED")->standardRate == false);
    REQUIRE(findSignalStateByName("invalid") == nullptr);

    REQUIRE(isDecoderCompatible(SignalState::STANDARD_TBC_LOCKED));
    REQUIRE(isDecoderCompatible(SignalState::STANDARD_TBC_UNLOCKED));
    REQUIRE(!isDecoderCompatible(SignalState::STANDARD_RAW));
    REQUIRE(!isDecoderCompatible(SignalState::NONSTANDARD_RAW));
    return 0;
}

int testMakeVideoParameters() {
    using namespace chd::format;
    // NTSC at STANDARD_TBC_LOCKED → orthogonal 910x263 field, blanking at
    // 240 × 64 = 15360, isSubcarrierLocked = true.
    auto vp = makeVideoParameters(getVideoStandard(VideoStandard::NTSC), true);
    REQUIRE(vp.fieldWidth == 910);
    REQUIRE(vp.fieldHeight == 263);
    REQUIRE(vp.blanking16bIre == 15360);
    REQUIRE(vp.black16bIre == 18048);   // 282 (blanking + 7.5 IRE setup) × 64
    REQUIRE(vp.white16bIre == 51200);
    REQUIRE(vp.isSubcarrierLocked == true);
    REQUIRE(vp.system == chd::metadata::NTSC);

    // PAL orthogonal-line simplification: 1135 samples/line, 313 lines/field.
    auto pal = makeVideoParameters(getVideoStandard(VideoStandard::PAL), false);
    REQUIRE(pal.fieldWidth == 1135);
    REQUIRE(pal.fieldHeight == 313);
    REQUIRE(pal.blanking16bIre == 16384);
    REQUIRE(pal.isSubcarrierLocked == false);
    REQUIRE(pal.system == chd::metadata::PAL);

    // PAL_M: 909 samples/line, 263 lines/field.
    auto palm = makeVideoParameters(getVideoStandard(VideoStandard::PAL_M), true);
    REQUIRE(palm.fieldWidth == 909);
    REQUIRE(palm.fieldHeight == 263);
    REQUIRE(palm.system == chd::metadata::PAL_M);
    return 0;
}

// The digital active line of the 4fsc interface standards (EBU Tech 3280-E for
// PAL, SMPTE ST 244 for NTSC/PAL_M), expressed in the stored row coordinates of
// a given horizontal alignment. Those standards number a row from the start of
// the digital active line, with 0H falling late in the row; our rows start
// either at 0H (line-locked TBC) or at the first digital blanking sample, so
// the digital active line has to be walked round to where our rows put it.
struct DigitalActiveLine {
    int32_t first;  // inclusive
    int32_t last;   // inclusive
};

DigitalActiveLine digitalActiveLine(const chd::format::VideoStandardPreset &preset,
                                    chd::format::HorizontalAlignment alignment) {
    const int32_t samplesPerLine = static_cast<int32_t>(std::lround(preset.samplesPerLineAvg));
    int32_t first;
    if (alignment == chd::format::HorizontalAlignment::SYNC_START) {
        // Stored sample 0 is the first sampling instant at or after 0H, so the
        // standard's sample 0 lands that many samples before the row's end.
        // 0H is quoted relative to the start of digital blanking, which itself
        // starts digitalActiveSamples into the standard's row.
        const double zeroH = preset.digitalActiveSamples + preset.zeroHBlankingStartRow;
        first = samplesPerLine - static_cast<int32_t>(std::ceil(zeroH));
    } else {
        // Rows start with digital blanking, so the digital active line follows it.
        first = samplesPerLine - preset.digitalActiveSamples;
    }
    return {first, first + preset.digitalActiveSamples - 1};
}

int testActiveSamplesAgainstStandards() {
    using namespace chd::format;

    // Anchor the walk against the one mapping the CVBS spec states outright:
    // SMPTE ST 244 puts NTSC's digital active line at its samples 0-767 with 0H
    // between samples 784 and 785, which in our 0H-aligned rows is stored
    // samples 125-892. PAL's 948-sample line lands at stored 177-1124 by the
    // same walk from EBU Tech 3280-E's 0H at sample 957.5.
    const auto ntscSync = digitalActiveLine(getVideoStandard(VideoStandard::NTSC),
                                            HorizontalAlignment::SYNC_START);
    REQUIRE(ntscSync.first == 125);
    REQUIRE(ntscSync.last  == 892);
    const auto palSync = digitalActiveLine(getVideoStandard(VideoStandard::PAL),
                                           HorizontalAlignment::SYNC_START);
    REQUIRE(palSync.first == 177);
    REQUIRE(palSync.last  == 1124);

    // The default synthesized crop IS the digital active line, on both
    // alignments, for every preset. These bounds are the standards' own sample
    // counts (768 / 948 wide) walked into our stored row coordinates; they are
    // hardcoded here rather than recomputed so a drift in either the helper or
    // makeVideoParameters is caught against fixed, standards-derived numbers.
    struct Expected {
        VideoStandard standard;
        HorizontalAlignment alignment;
        int32_t first;  // inclusive
        int32_t last;   // inclusive
    };
    static constexpr Expected EXPECTED[] = {
        {VideoStandard::PAL,   HorizontalAlignment::SYNC_START,     177, 1124},
        {VideoStandard::PAL,   HorizontalAlignment::BLANKING_START, 187, 1134},
        {VideoStandard::NTSC,  HorizontalAlignment::SYNC_START,     125,  892},
        {VideoStandard::NTSC,  HorizontalAlignment::BLANKING_START, 142,  909},
        {VideoStandard::PAL_M, HorizontalAlignment::SYNC_START,     124,  891},
        {VideoStandard::PAL_M, HorizontalAlignment::BLANKING_START, 141,  908},
    };

    for (const auto &e : EXPECTED) {
        const auto &preset = getVideoStandard(e.standard);
        const auto vp = makeVideoParameters(preset, true, e.alignment);
        // VideoParameters carries the crop half-open; the ABI reports it
        // inclusive, and inclusive is what compares against the standards.
        const int32_t firstActiveSample = vp.activeVideoStart;
        const int32_t lastActiveSample  = vp.activeVideoEnd - 1;
        REQUIRE(firstActiveSample == e.first);
        REQUIRE(lastActiveSample  == e.last);
        REQUIRE(lastActiveSample - firstActiveSample + 1 == preset.digitalActiveSamples);

        // And it matches the independently-walked digital active line helper.
        const auto dal = digitalActiveLine(preset, e.alignment);
        REQUIRE(firstActiveSample == dal.first);
        REQUIRE(lastActiveSample  == dal.last);
    }
    return 0;
}

}  // namespace

int main() {
    if (testVideoStandardLookup() != 0) return 1;
    if (testSampleEncodingLookup() != 0) return 1;
    if (testSampleConversion() != 0) return 1;
    if (testSignalStateLookup() != 0) return 1;
    if (testMakeVideoParameters() != 0) return 1;
    if (testActiveSamplesAgainstStandards() != 0) return 1;
    std::cout << "All format-preset tests passed.\n";
    return 0;
}
