// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_standards.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "../common/log.h"

namespace chd::format {

// EBU Tech 3280-E (PAL). PAL carries no setup pedestal, so black sits at
// blanking. Values are 10-bit (sync 01.0h, blanking 40.0h, white D3.0h, max
// legal FE.Ch).
static constexpr SampleLevels10b PAL_LEVELS = {
    /*syncTip*/        4,
    /*blanking*/       256,
    /*black*/          256,
    /*white*/          844,
    /*peak*/           1019,
    /*protectedMinMax*/ 3,
    /*protectedMaxMin*/ 1020,
};

// SMPTE ST 244 coding levels (NTSC, PAL_M) with the +7.5 IRE setup pedestal
// from SMPTE ST 170: black = blanking + 7.5 IRE = 240 + 7.5 * 5.6 = 282. ST 244
// itself defines only sync/blanking/white; max legal value is 3FBh.
static constexpr SampleLevels10b ST244_LEVELS = {
    /*syncTip*/        16,
    /*blanking*/       240,
    /*black*/          282,
    /*white*/          800,
    /*peak*/           1019,
    /*protectedMinMax*/ 3,
    /*protectedMaxMin*/ 1020,
};

// Sampling-rate formulas evaluated at compile time. Constants from the spec.
static constexpr double PAL_SAMPLE_RATE   = 4.0 * 625.0 * 25.0 * (1135.0 / 4.0 + 1.0 / 625.0);
static constexpr double NTSC_SAMPLE_RATE  = 4.0 * 525.0 * (30000.0 / 1001.0) * (455.0 / 2.0);
static constexpr double PALM_SAMPLE_RATE  = 4.0 * 525.0 * (30000.0 / 1001.0) * (909.0 / 4.0);

// Colour subcarrier frequencies. PAL fSC = 283.75 * 15625 + 25 Hz; NTSC fSC =
// 315 MHz / 88; PAL-M fSC = 5 MHz * (63/88) * (909/910). Match the values
// already initialised in src/metadata/core.cpp's VideoSystemDefaults so the
// CVBS path and the TBC path agree.
static constexpr double PAL_FSC  = 283.75 * 15625.0 + 25.0;
static constexpr double NTSC_FSC = 315.0e6 / 88.0;
static constexpr double PALM_FSC = 5.0e6 * (63.0 / 88.0) * (909.0 / 910.0);

static constexpr VideoStandardPreset PRESETS[] = {
    // PAL: 625 lines, 4-frame colour sequence. Per the spec the per-line
    // average is 1135.0064 (non-orthogonal); the normative invariant is the
    // 709,379-sample frame total.
    {
        VideoStandard::PAL,  "PAL",  chd::metadata::PAL,
        PAL_SAMPLE_RATE, PAL_FSC,
        1135.0 + 1.0 / 156.25,   // = 1135.0064; spec value
        709379,
        625,
        4,
        /*zeroHBlankingStartRow*/ 957.5 - 948.0,
        /*burstStartFromZeroH*/   5.6e-6 * PAL_SAMPLE_RATE,
        /*burstLengthSamples*/    10 * 4,
        /*digitalActiveSamples*/  948,
        PAL_LEVELS,
    },
    // NTSC: 525 lines, exact 910 samples/line, 2-frame colour sequence.
    {
        VideoStandard::NTSC, "NTSC", chd::metadata::NTSC,
        NTSC_SAMPLE_RATE, NTSC_FSC,
        910.0,
        477750,
        525,
        2,
        /*zeroHBlankingStartRow*/ 784.0 + 33.0 / 90.0 - 768.0,
        /*burstStartFromZeroH*/   19.0 * 4,
        /*burstLengthSamples*/    9 * 4,
        /*digitalActiveSamples*/  768,
        ST244_LEVELS,
    },
    // PAL_M: 525 lines, exact 909 samples/line, 4-frame colour sequence.
    // 0H carries the SMPTE 244M digital-blanking timing to the 909-sample
    // line (the spec's PAL_M preset adopts SMPTE-244M-compatible coding).
    {
        VideoStandard::PAL_M, "PAL_M", chd::metadata::PAL_M,
        PALM_SAMPLE_RATE, PALM_FSC,
        909.0,
        477225,
        525,
        4,
        /*zeroHBlankingStartRow*/ (784.0 + 33.0 / 90.0 - 768.0) * (909.0 / 910.0),
        /*burstStartFromZeroH*/   19.0 * 4,
        /*burstLengthSamples*/    9 * 4,
        /*digitalActiveSamples*/  768,
        ST244_LEVELS,
    },
};

const VideoStandardPreset *findVideoStandardByName(const std::string &name)
{
    for (const auto &preset : PRESETS) {
        if (name == preset.name) return &preset;
    }
    return nullptr;
}

const VideoStandardPreset &getVideoStandard(VideoStandard standard)
{
    return PRESETS[static_cast<size_t>(standard)];
}

chd::metadata::LdDecodeMetaData::VideoParameters
makeVideoParameters(const VideoStandardPreset &preset,
                    bool isSubcarrierLocked,
                    HorizontalAlignment alignment)
{
    chd::metadata::LdDecodeMetaData::VideoParameters vp;

    vp.system = preset.videoSystem;
    vp.isSubcarrierLocked = isSubcarrierLocked;
    vp.isWidescreen = false;

    vp.sampleRate = preset.sampleRateHz;
    vp.fSC        = preset.fSC;

    // Sample-domain values are 10-bit values shifted left 6 (i.e. multiplied
    // by 64), matching the canonical TBC convention so existing decoders can
    // consume the data unchanged.
    vp.blanking16bIre = preset.levels.blanking * 64;
    vp.black16bIre    = preset.levels.black    * 64;
    vp.white16bIre    = preset.levels.white    * 64;

    // Field width: for the orthogonal-line simplification PAL is
    // treated as if orthogonal at 1135 samples/line so PALcolour's existing
    // assumption holds. NTSC and PAL_M are genuinely orthogonal.
    const int32_t samplesPerLine =
        (preset.standard == VideoStandard::PAL)
            ? 1135
            : static_cast<int32_t>(preset.samplesPerLineAvg);
    const int32_t linesPerField = preset.linesPerFrame / 2 + 1;  // 263 for 525, 313 for 625

    vp.fieldWidth  = samplesPerLine;
    vp.fieldHeight = linesPerField;

    // Active region defaults mirror VIDEO_SYSTEM_DEFAULTS in src/metadata/core.cpp
    // (keep the two in sync). first/last active frame lines are inclusive; these
    // values intentionally differ from decode-orc and tbc-tools' exclusive
    // defaults. The decoder pipeline calls processLineParameters() later to
    // populate firstActiveFrameLine / lastActiveFrameLine; CVBS-opened captures
    // don't yet have a metadata sidecar that overrides these, so populate the
    // defaults explicitly so chd_video_get_info reports sensible values.
    if (preset.standard == VideoStandard::PAL) {
        vp.firstActiveFrameLine = 44;
        vp.lastActiveFrameLine  = 619;
    } else {
        vp.firstActiveFrameLine = 39;
        vp.lastActiveFrameLine  = 524;
    }

    // Colour-burst and active-video windows are row positions, so they follow
    // the horizontal alignment. The default active-video crop is the digital
    // active line of the interface standard (SMPTE ST 244 for 525-line, EBU
    // Tech 3280-E for 625-line): digitalActiveSamples wide, positioned so it
    // ends at the standard's 0H. This is wider than the analogue picture and
    // includes the blanking transitions on each side; callers who want a
    // tighter crop set CHD_OPT_FIRST/LAST_ACTIVE_SAMPLE. Burst windows are
    // unchanged: SYNC_START keeps the ld-decode convention values, BLANKING_START
    // derives them from the row-local 0H with the same arithmetic
    // ld-chroma-encoder uses (PAL: burst 109..149).
    if (alignment == HorizontalAlignment::SYNC_START) {
        // Row starts at 0H; the standard's own 0H sits digitalActiveSamples +
        // zeroHBlankingStartRow into its digital line, so the digital active
        // line ends that far before this row's end.
        const double zeroHInRow = preset.digitalActiveSamples + preset.zeroHBlankingStartRow;
        vp.activeVideoStart = samplesPerLine - static_cast<int32_t>(std::ceil(zeroHInRow));
        vp.colourBurstStart =
            (preset.standard == VideoStandard::PAL) ? 98 : 75;
        vp.colourBurstEnd =
            (preset.standard == VideoStandard::PAL) ? 138 : 95;
    } else {
        // Row starts at the first digital blanking sample, so the digital
        // active line follows it and runs to the row end.
        const double zeroH = preset.zeroHBlankingStartRow;
        vp.colourBurstStart = static_cast<int32_t>(
            std::lrint(zeroH + preset.burstStartFromZeroH));
        vp.colourBurstEnd = static_cast<int32_t>(
            std::lrint(zeroH + preset.burstStartFromZeroH + preset.burstLengthSamples));
        vp.activeVideoStart = samplesPerLine - preset.digitalActiveSamples;
    }
    vp.activeVideoEnd = vp.activeVideoStart + preset.digitalActiveSamples;

    // The picture inside that digital line, where a .tbc of the same signal
    // would report it: the ld-decode window, or ld-chroma-encoder's.
    if (preset.standard == VideoStandard::NTSC) {
        const bool syncStart = alignment == HorizontalAlignment::SYNC_START;
        vp.pictureStart = syncStart ? 134 : 147;
        vp.pictureEnd   = syncStart ? 894 : 905;
    }

    vp.isMapped = false;
    vp.isValid  = true;
    return vp;
}

chd::metadata::LdDecodeMetaData::VideoParameters
makeCvbsVideoParameters(const VideoStandardPreset &preset,
                        SampleEncoding encoding,
                        std::optional<int32_t> blackLevelOverride,
                        FrameLayout layout,
                        std::optional<bool> subcarrierLockedOverride,
                        std::optional<HorizontalAlignment> alignmentOverride)
{
    // Subcarrier lock is a lattice property, not the signal_state burst lock:
    // a burst-phase-stable line-locked capture is still not subcarrier-locked.
    // Frame-native PAL is definitionally locked; a field raster defaults to
    // line-locked unless the caller marks it otherwise.
    const bool subcarrierLocked =
        (layout == FrameLayout::FRAME_NATIVE && preset.standard == VideoStandard::PAL) ||
        subcarrierLockedOverride.value_or(false);
    // Derived alignment: subcarrier-locked rasters carry the blanking-start
    // cut, plain field rasters the line-locked sync-start cut.
    // Frame-native sources resolve theirs from the 0H measurement and pass
    // it via alignmentOverride.
    const HorizontalAlignment alignment = alignmentOverride.value_or(
        (layout == FrameLayout::FRAME_NATIVE || subcarrierLockedOverride.value_or(false))
            ? HorizontalAlignment::BLANKING_START
            : HorizontalAlignment::SYNC_START);
    auto vp = makeVideoParameters(preset, subcarrierLocked, alignment);
    // A `.meta` black_level override replaces the preset default. The stored
    // value is in the encoding's integer domain, so map it through the same
    // canonical conversion the samples use.
    if (blackLevelOverride.has_value()) {
        vp.black16bIre = convertCompositeSampleToCanonical(
            encoding, static_cast<int16_t>(*blackLevelOverride), preset.levels.blanking);
    }
    return vp;
}

int32_t fieldPackedSamplesPerFrame(const VideoStandardPreset &preset)
{
    const int32_t samplesPerLine =
        (preset.standard == VideoStandard::PAL)
            ? 1135
            : static_cast<int32_t>(preset.samplesPerLineAvg);
    return samplesPerLine * (preset.linesPerFrame / 2 + 1) * 2;
}

FrameLayout resolveFrameLayout(FrameLayout overrideLayout,
                               const VideoStandardPreset &preset,
                               SignalState signalState,
                               int32_t bytesPerSample,
                               int64_t fileSize,
                               std::optional<int64_t> declaredFrames)
{
    if (overrideLayout != FrameLayout::UNKNOWN) return overrideLayout;

    // Frame totals are normative only for TBC-applied standard-rate states
    // (video-standard-presets.md "Exact frame size").
    const SignalStatePreset &ss = getSignalState(signalState);
    if (!ss.standardRate || !ss.tbcApplied) return FrameLayout::FIELD_RASTER;

    const int64_t nativeBytes =
        static_cast<int64_t>(preset.samplesPerFrame) * bytesPerSample;
    const int64_t packedBytes =
        static_cast<int64_t>(fieldPackedSamplesPerFrame(preset)) * bytesPerSample;

    // A sidecar-declared frame count divides the file size exactly for the
    // real layout; the modulo test below cannot distinguish the two totals
    // when both divide (they collide every 526 native frames on the 525-line
    // standards).
    if (declaredFrames.has_value() && *declaredFrames > 0) {
        if (fileSize == *declaredFrames * nativeBytes) return FrameLayout::FRAME_NATIVE;
        if (fileSize == *declaredFrames * packedBytes) return FrameLayout::FIELD_RASTER;
    }

    const bool nativeFits = fileSize > 0 && (fileSize % nativeBytes) == 0;
    const bool packedFits = fileSize > 0 && (fileSize % packedBytes) == 0;
    if (nativeFits && !packedFits) return FrameLayout::FRAME_NATIVE;
    if (nativeFits && packedFits) {
        chd::log::warn() << "resolveFrameLayout: file size matches both the frame-native"
                         << "and field-raster totals; assuming field raster";
    } else if (!packedFits) {
        chd::log::warn() << "resolveFrameLayout: file size matches neither the"
                         << "frame-native nor the field-raster total; assuming field raster";
    }
    return FrameLayout::FIELD_RASTER;
}

std::optional<double> measureRowZeroH(const uint16_t *samples,
                                      int32_t numRows,
                                      int32_t rowWidth,
                                      const SampleLevels10b &levels)
{
    // Half-amplitude threshold between blanking and sync tip, in the
    // canonical x64 domain. A clean falling edge has a sustained run of
    // blanking-or-above before it and a sustained run of sync-tip level
    // after it, which excludes equalising-pulse regions and picture content.
    const int32_t threshold = ((levels.blanking + levels.syncTip) / 2) * 64;
    constexpr int32_t RUN = 20;

    // Rows are one full line period, so the runs wrap: an edge near the row
    // start (blanking-start rows put 0H at ~9.5) checks its preceding
    // samples at the row end.
    std::vector<double> positions;
    for (int32_t row = 0; row < numRows; ++row) {
        const uint16_t *r = samples + static_cast<int64_t>(row) * rowWidth;
        auto at = [&](int32_t i) { return r[((i % rowWidth) + rowWidth) % rowWidth]; };
        for (int32_t i = 0; i < rowWidth; ++i) {
            if (at(i - 1) <= threshold || at(i) > threshold) continue;
            bool clean = true;
            for (int32_t j = i - RUN; j < i && clean; ++j) clean = at(j) > threshold;
            for (int32_t j = i; j < i + RUN && clean; ++j) clean = at(j) <= threshold;
            if (!clean) continue;
            const double num = static_cast<double>(at(i - 1)) - threshold;
            const double den = static_cast<double>(at(i - 1)) - at(i);
            double pos = (i - 1) + (den > 0.0 ? num / den : 0.5);
            if (pos < 0.0) pos += rowWidth;
            positions.push_back(pos);
            break;
        }
    }
    if (positions.size() * 2 < static_cast<size_t>(numRows)) return std::nullopt;
    std::nth_element(positions.begin(), positions.begin() + positions.size() / 2,
                     positions.end());
    return positions[positions.size() / 2];
}

HorizontalAlignment resolveFrameNativeAlignment(const VideoStandardPreset &preset,
                                                const std::optional<double> &measuredZeroH,
                                                const char *sourceName)
{
    if (!measuredZeroH.has_value()) {
        chd::log::debug() << sourceName
                          << ": no measurable sync edge; assuming sync-start rows";
        return HorizontalAlignment::SYNC_START;
    }
    constexpr double TOLERANCE = 8.0;
    const int32_t rowWidth =
        (preset.standard == VideoStandard::PAL)
            ? 1135
            : static_cast<int32_t>(preset.samplesPerLineAvg);
    const double m = *measuredZeroH;
    const double blankingStart = preset.zeroHBlankingStartRow;
    const double activeStart = blankingStart + preset.digitalActiveSamples;
    // 0H at the row start wraps: a capture triggering just before sync
    // measures slightly below rowWidth.
    const double distToRowStart = std::min(m, rowWidth - m);

    if (std::abs(m - blankingStart) <= TOLERANCE) {
        chd::log::debug().nospace()
            << sourceName << ": blanking-start rows (0H at " << m << ")";
        return HorizontalAlignment::BLANKING_START;
    }
    if (distToRowStart <= TOLERANCE) {
        chd::log::debug().nospace()
            << sourceName << ": sync-start rows (0H at " << m << ")";
        return HorizontalAlignment::SYNC_START;
    }
    if (std::abs(m - activeStart) <= TOLERANCE) {
        chd::log::warn().nospace()
            << sourceName << ": rows appear active-start aligned (0H at " << m
            << "); not servable without a row re-cut, using sync-start windows;"
            << " colour decoding will misbehave";
    } else {
        chd::log::warn().nospace()
            << sourceName << ": row 0H measured at " << m
            << ", matching no known cut; using sync-start windows";
    }
    return HorizontalAlignment::SYNC_START;
}

std::optional<ChromaPorchSignature> measure625ChromaPorchSignature(
    const uint16_t *rows, int32_t numRows, int32_t rowWidth,
    int32_t porchStart, int32_t porchEnd, double sampleRateHz,
    int32_t blanking16bIre, int32_t white16bIre)
{
    // Per row, estimate the dominant porch frequency from the lag-1
    // autocorrelation of the mean-removed window: for a near-tone,
    // r1/r0 = cos(2*pi*f/fs). Reject rows whose porch RMS is under 1% of
    // the blanking-to-white range (no reference carrier).
    const double minRms = 0.01 * (white16bIre - blanking16bIre);
    std::vector<double> freqs;
    freqs.reserve(numRows);
    std::vector<int32_t> parities;
    parities.reserve(numRows);

    for (int32_t row = 0; row < numRows; ++row) {
        const uint16_t *r = rows + static_cast<int64_t>(row) * rowWidth;
        double mean = 0.0;
        int32_t n = 0;
        for (int32_t x = porchStart; x < porchEnd && x < rowWidth; ++x) {
            mean += r[x];
            n++;
        }
        if (n < 16) return std::nullopt;
        mean /= n;
        double r0 = 0.0, r1 = 0.0;
        double prev = 0.0;
        bool havePrev = false;
        for (int32_t x = porchStart; x < porchEnd && x < rowWidth; ++x) {
            const double v = r[x] - mean;
            r0 += v * v;
            if (havePrev) r1 += v * prev;
            prev = v;
            havePrev = true;
        }
        if (std::sqrt(r0 / n) < minRms || r0 <= 0.0) continue;
        const double c = std::clamp(r1 / r0, -1.0, 1.0);
        freqs.push_back(std::acos(c) * sampleRateHz / (2.0 * M_PI));
        parities.push_back(row & 1);
    }
    if (freqs.size() < 24) return std::nullopt;

    double sum[2] = {0.0, 0.0};
    int32_t count[2] = {0, 0};
    for (size_t i = 0; i < freqs.size(); i++) {
        sum[parities[i]] += freqs[i];
        count[parities[i]]++;
    }
    if (count[0] == 0 || count[1] == 0) return std::nullopt;

    ChromaPorchSignature sig;
    sig.meanHz = (sum[0] + sum[1]) / (count[0] + count[1]);
    sig.alternationHz = std::abs(sum[0] / count[0] - sum[1] / count[1]);
    return sig;
}

std::optional<bool> measureNtscFieldBurstPolarity(const uint16_t *rows,
                                                  int32_t firstRow,
                                                  int32_t numRows,
                                                  int32_t rowWidth,
                                                  int32_t burstStart,
                                                  int32_t burstEnd,
                                                  int32_t blanking16bIre,
                                                  int32_t white16bIre)
{
    // Quadrature-sum the burst with the same per-sample sign pattern the NTSC
    // comb's splitIQ uses (h%4 -> +Q, -I, -Q, +I), giving the burst vector in
    // the comb's I/Q frame for an un-negated (linePhase == false) line. On a
    // correctly phased line that vector is the burst's 180-degree vectorscope
    // angle projected onto the I/Q axes at +123 and +33 degrees.
    constexpr double kBurstRefI = 0.5446;   // cos(180 - 123 deg)
    constexpr double kBurstRefQ = -0.8387;  // cos(180 - 33 deg)
    // Reject lines whose burst vector is shorter than 5 IRE: blank synthetic
    // data, chroma-free content, or amplitude scales that carry no burst.
    const double minMagnitude = 5.0 * (white16bIre - blanking16bIre) / 100.0;

    int32_t votesPositive = 0;
    int32_t votesValid = 0;
    for (int32_t row = 0; row < numRows; ++row) {
        const uint16_t *r = rows + static_cast<int64_t>(row) * rowWidth;
        double sums[4] = {0.0, 0.0, 0.0, 0.0};
        int32_t counts[4] = {0, 0, 0, 0};
        for (int32_t h = burstStart; h < burstEnd && h < rowWidth; ++h) {
            sums[h % 4] += r[h] - blanking16bIre;
            counts[h % 4]++;
        }
        if (counts[0] == 0 || counts[1] == 0 || counts[2] == 0 || counts[3] == 0) {
            return std::nullopt;
        }
        const double i = sums[3] / counts[3] - sums[1] / counts[1];
        const double q = sums[0] / counts[0] - sums[2] / counts[2];
        if (std::sqrt(i * i + q * q) < minMagnitude) continue;
        const bool linePhase = (kBurstRefI * i + kBurstRefQ * q) < 0.0;
        // Burst flips 180 degrees per line, so translate each line's phase to
        // the field-level flag via the field-line parity.
        const bool positiveOnEvenLines =
            (((firstRow + row) % 2) == 0) ? linePhase : !linePhase;
        if (positiveOnEvenLines) votesPositive++;
        votesValid++;
    }
    if (votesValid < 4) return std::nullopt;
    // A lattice-locked burst votes unanimously; anything mixed is not a
    // signal this measurement can classify.
    if (votesPositive * 4 >= votesValid * 3) return true;
    if ((votesValid - votesPositive) * 4 >= votesValid * 3) return false;
    return std::nullopt;
}

FrameNativeFieldRead planFrameNativeFieldRead(const VideoStandardPreset &preset,
                                              int32_t fieldIndex,
                                              int32_t firstRow0,
                                              int32_t lastRow0,
                                              int32_t bytesPerSample)
{
    // Flat cut, matching ld-chroma-encoder's field layout: the two field
    // buffers are consecutive windows of the continuous native frame, and
    // the part of the second buffer past the native total is padding. The
    // first field holds temporal lines 0..fieldHeight-1; the second holds
    // the rest. For subcarrier-locked PAL each row's 0H drifts +4/625 of a
    // sample per line within the uniform 1135-sample rows, which is what
    // puts the two fields 2 samples apart (removed downstream by
    // source_field's gated shift); the 4 leftover samples per frame land at
    // the start of the second field's final row.
    const int32_t lineWidth =
        (preset.standard == VideoStandard::PAL)
            ? 1135
            : static_cast<int32_t>(preset.samplesPerLineAvg);
    const int64_t bufferSamples =
        static_cast<int64_t>(lineWidth) * (preset.linesPerFrame / 2 + 1);
    const int64_t fieldBase = (fieldIndex % 2 == 0) ? 0 : bufferSamples;
    const int64_t frameBase =
        static_cast<int64_t>(fieldIndex / 2) * preset.samplesPerFrame;

    const int64_t startSample = fieldBase + static_cast<int64_t>(firstRow0) * lineWidth;
    const int64_t endSample   = fieldBase + static_cast<int64_t>(lastRow0 + 1) * lineWidth;
    const int64_t realStart   = std::min<int64_t>(startSample, preset.samplesPerFrame);
    const int64_t realEnd     = std::min<int64_t>(endSample, preset.samplesPerFrame);

    FrameNativeFieldRead plan;
    plan.startByte  = (frameBase + realStart) * bytesPerSample;
    plan.numBytes   = (realEnd - realStart) * bytesPerSample;
    plan.padSamples = static_cast<int32_t>(endSample - realEnd);
    return plan;
}

}  // namespace chd::format
