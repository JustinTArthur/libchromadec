// SPDX-License-Identifier: GPL-3.0-or-later
//
// SECAM decoder core against a synthetic SECAM generator (the correctness
// authority: no external golden reference decodes SECAM). The generator
// modulates known D'B/D'R bars per BT.1700 Part C Table 4 including both
// pre-corrections (LF shelf and HF bell), an ME-SECAM-style carrier offset,
// per-line AGC scaling (vhs-decode applies one), and back-porch reference
// carriers. The decoder must recover the colour differences, the per-line
// component lattice, and per-field carrier calibration. A second pass
// drives the same signal through the public Y/C ABI to a 4:4:0 float frame.

#include <chromadec/decoder.h>
#include <chromadec/dropout.h>
#include <chromadec/frame.h>
#include <chromadec/video.h>

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../../src/decoders/secam/secam_decoder.h"
#include "../../src/decoders/source_field.h"
#include "../../src/metadata/core.h"
#include "../../src/output/component_frame.h"

namespace fs = std::filesystem;

namespace {

#define REQUIRE(cond)                                                            \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #cond \
                      << "\n";                                                   \
            return 1;                                                            \
        }                                                                        \
    } while (0)

constexpr int32_t kWidth = 1135;
constexpr int32_t kHeight = 313;
constexpr double kSampleRate = 17734475.0;
constexpr double kFob = 4250000.0;
constexpr double kFor = 4406250.0;
constexpr double kDevDb = 230000.0;
constexpr double kDevDr = 280000.0;
constexpr double kBellF0 = 4286000.0;
constexpr double kLfF1 = 85000.0;
// Emulates the ME-SECAM converter arithmetic offset measured on the
// vhs-decode fixture: both carriers sit above nominal; calibration must
// absorb it.
constexpr double kCarrierOffset = 108000.0;

constexpr int32_t kActiveStart = 185;
constexpr int32_t kActiveEnd = 1107;
constexpr int32_t kNumBars = 8;

// Bar values and adjacencies chosen so the LF pre-corrected transient stays
// inside the Table 4 deviation limits (D'B in [-1.52, +2.20], D'R in
// [-1.81, +1.25]; smallest margin 0.16 by direct simulation of the
// generator chain): the transmitter limiter, the decoder's deviation rail,
// and click detection are all no-ops on the clean fixture, so bar centres
// decode exactly through the default pipeline.
constexpr double kDbBars[kNumBars] = {0.0, 0.8, 1.2, 0.55, 0.1, -0.35, -0.75, -0.4};
constexpr double kDrBars[kNumBars] = {0.0, -0.55, -0.95, -0.6, -0.1, 0.3, 0.65, 0.25};

chd::metadata::LdDecodeMetaData::VideoParameters secamParams() {
    chd::metadata::LdDecodeMetaData::VideoParameters vp;
    vp.system = chd::metadata::SECAM;
    vp.fSC = kBellF0;
    vp.fieldWidth = kWidth;
    vp.fieldHeight = kHeight;
    vp.sampleRate = kSampleRate;
    vp.black16bIre = 16384;
    vp.white16bIre = 54016;
    vp.blanking16bIre = 16384;
    vp.colourBurstStart = 98;
    vp.colourBurstEnd = 138;
    vp.activeVideoStart = kActiveStart;
    vp.activeVideoEnd = kActiveEnd;
    vp.firstActiveFrameLine = 44;
    vp.lastActiveFrameLine = 619;
    vp.numberOfSequentialFields = 2;
    vp.isValid = true;
    return vp;
}

int32_t barIndexFor(int32_t x) {
    const int32_t barWidth = (kActiveEnd - kActiveStart) / kNumBars;
    return std::min((x - kActiveStart) / barWidth, kNumBars - 1);
}

// Apply a frequency response to a real signal with one whole-signal FFT
// (hermitian responses keep the result real).
template <typename ResponseFn>
void applyResponse(std::vector<double> &signal, ResponseFn response) {
    const int32_t n = static_cast<int32_t>(signal.size());
    fftw_complex *in = fftw_alloc_complex(n);
    fftw_complex *out = fftw_alloc_complex(n);
    for (int32_t i = 0; i < n; i++) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }
    fftw_plan fwd = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fwd);
    for (int32_t k = 0; k < n; k++) {
        const double f = (k <= n / 2 ? k : k - n) * kSampleRate / n;
        const std::complex<double> h = response(f);
        const std::complex<double> v = std::complex<double>(out[k][0], out[k][1]) * h;
        out[k][0] = v.real();
        out[k][1] = v.imag();
    }
    fftw_plan bwd = fftw_plan_dft_1d(n, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(bwd);
    for (int32_t i = 0; i < n; i++) {
        signal[i] = in[i][0] / n;
    }
    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);
    fftw_free(in);
    fftw_free(out);
}

// Colour-difference bandwidth limit (Table 4: D' attenuation <= 3 dB at
// 1.3 MHz): Gaussian, zero-phase. Applied before the LF pre-correction so
// bar transitions carry physical bandwidth; without it the pre-corrected
// edges would sweep the FM carrier outside the chroma band entirely.
std::complex<double> colourDiffLowpass(double f) {
    return {std::exp(-M_LN2 * (f / 1300000.0) * (f / 1300000.0)), 0.0};
}

std::complex<double> lfPrecorrection(double f) {
    // A_LFP(f) = (1 + jf/f1) / (1 + jf/3f1)
    return std::complex<double>(1.0, f / kLfF1)
           / std::complex<double>(1.0, f / (3.0 * kLfF1));
}

std::complex<double> hfBell(double f, double centre) {
    // A_HFP(f) = (1 + j16F) / (1 + j1.26F), F = f/f0 - f0/f; hermitian via
    // the odd F. The centre follows the carrier offset: a converter offset
    // arises after broadcast encoding and translates the whole FM block,
    // the encoder's bell shaping included.
    if (f == 0.0) return {1.0, 0.0};
    const double F = f / centre - centre / f;
    return std::complex<double>(1.0, 16.0 * F) / std::complex<double>(1.0, 1.26 * F);
}

// Generator options. Field line l carries component (l + anchor) & 1
// (0 = Db, 1 = Dr). With porchCarrier the undeviated reference runs across
// blanking (line identification); without it, chroma exists only over the
// picture (and the bottle plateaus when enabled).
struct GenOpts {
    int32_t anchor = 0;
    int32_t firstChromaRow = 16;
    bool porchCarrier = true;
    bool bottles = false;
    double carrierOffset = kCarrierOffset;
    // Uniform sample noise (16-bit counts) from a deterministic LCG.
    double noiseAmplitude = 0.0;
    uint32_t noiseSeed = 1;
    // Model the transmitter's deviation limiter. The decoder holds the
    // same Table 4 bounds (deviation rail, click detection), so the
    // fixture must be compliant or legal pre-correction overshoots read
    // as deviation excursions; bar centres are steady-state and exact
    // either way.
    bool limiter = true;
};

// Field-ident bottle geometry (vertical-interval trapezoids).
constexpr int32_t kBottleFirstRow = 6;
constexpr int32_t kBottleLastRow = 14;
constexpr int32_t kBottleStartX = 250;
constexpr int32_t kBottleEndX = 1000;

std::vector<uint16_t> generateField(const GenOpts &opts) {
    const size_t n = static_cast<size_t>(kWidth) * kHeight;

    // Baseband colour-difference signal per line: bars over the active
    // region, zero (undeviated carrier) elsewhere; bottle plateaus on the
    // early vertical-interval lines when enabled.
    std::vector<double> baseband(n, 0.0);
    for (int32_t row = opts.firstChromaRow; row < kHeight; row++) {
        const int32_t component = (row + opts.anchor) & 1;
        for (int32_t x = kActiveStart; x < kActiveEnd; x++) {
            const int32_t bar = barIndexFor(x);
            baseband[static_cast<size_t>(row) * kWidth + x] =
                (component == 0) ? kDbBars[bar] : kDrBars[bar];
        }
    }
    if (opts.bottles) {
        for (int32_t row = kBottleFirstRow; row <= kBottleLastRow; row++) {
            const int32_t component = (row + opts.anchor) & 1;
            for (int32_t x = kBottleStartX; x < kBottleEndX; x++) {
                baseband[static_cast<size_t>(row) * kWidth + x] =
                    (component == 1) ? 1.25 : -1.52;
            }
        }
    }
    applyResponse(baseband, colourDiffLowpass);
    applyResponse(baseband, lfPrecorrection);

    // The transmitter's limiter: clamp the pre-corrected signal to the
    // Table 4 maximum deviations for each line's component.
    for (int32_t row = 0; opts.limiter && row < kHeight; row++) {
        const int32_t component = (row + opts.anchor) & 1;
        const double lo = (component == 0) ? -350000.0 / kDevDb : -506000.0 / kDevDr;
        const double hi = (component == 0) ? 506000.0 / kDevDb : 350000.0 / kDevDr;
        for (int32_t x = 0; x < kWidth; x++) {
            double &v = baseband[static_cast<size_t>(row) * kWidth + x];
            v = std::clamp(v, lo, hi);
        }
    }

    // Amplitude gate: full-line carrier with porch ident, else picture and
    // bottle spans only.
    const auto chromaOn = [&](int32_t row, int32_t x) {
        if (opts.bottles && row >= kBottleFirstRow && row <= kBottleLastRow) {
            return x >= kBottleStartX - 20 && x < kBottleEndX + 20;
        }
        if (opts.porchCarrier) return row >= opts.firstChromaRow;
        return row >= 20 && x >= kActiveStart && x < kActiveEnd;
    };

    // FM-modulate with continuous phase across the field.
    std::vector<double> chroma(n, 0.0);
    const double amplitude = 0.115 * (54016 - 16384);  // 2M0 = 23% of blanking-to-white
    double phase = 0.0;
    for (int32_t row = 0; row < kHeight; row++) {
        const int32_t component = (row + opts.anchor) & 1;
        const double carrier = (component == 0 ? kFob : kFor) + opts.carrierOffset;
        const double deviation = (component == 0) ? kDevDb : kDevDr;
        for (int32_t x = 0; x < kWidth; x++) {
            const size_t i = static_cast<size_t>(row) * kWidth + x;
            const double f = carrier + deviation * baseband[i];
            phase += 2.0 * M_PI * f / kSampleRate;
            if (phase > 2.0 * M_PI) phase -= 2.0 * M_PI;
            chroma[i] = chromaOn(row, x) ? amplitude * std::cos(phase) : 0.0;
        }
    }
    applyResponse(chroma, [&](double f) {
        return hfBell(f, kBellF0 + opts.carrierOffset);
    });

    // Per-line envelope scaling emulating vhs-decode's chroma AGC; the FM
    // information is in frequency, so the decoder must not care.
    std::vector<uint16_t> samples(n, 32767);
    uint32_t lcg = opts.noiseSeed;
    for (int32_t row = 0; row < kHeight; row++) {
        const double agc = 0.6 + 0.05 * ((row * 7) % 13);
        for (int32_t x = 0; x < kWidth; x++) {
            const size_t i = static_cast<size_t>(row) * kWidth + x;
            lcg = lcg * 1664525u + 1013904223u;
            const double noise =
                opts.noiseAmplitude * ((lcg >> 8) / 8388607.5 - 1.0);
            const double v = 32767.0 + chroma[i] * agc + noise;
            samples[i] = static_cast<uint16_t>(std::clamp(v, 0.0, 65535.0));
        }
    }
    return samples;
}

// Median over the central samples of a bar on one decoded row.
double barMedian(const double *rowData, int32_t bar) {
    const int32_t barWidth = (kActiveEnd - kActiveStart) / kNumBars;
    const int32_t centre = kActiveStart + bar * barWidth + barWidth / 2;
    std::vector<double> v;
    for (int32_t x = centre - 10; x <= centre + 10; x++) v.push_back(rowData[x]);
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

constexpr double kBReduction = 0.49211104112248356308804691718185;
constexpr double kRReduction = 0.87728321993817866838972487283129;
constexpr double kUvRange = 54016.0 - 16384.0;

// Build the standard two-field frame with the given generator options
// (field 2 continues the transmitted-line alternation: anchor ^ 1).
std::vector<chd::decoders::SourceField> makeFields(GenOpts opts) {
    std::vector<chd::decoders::SourceField> fields(2);
    fields[0].field.seqNo = 1;
    fields[0].field.isFirstField = true;
    fields[0].data = generateField(opts);
    opts.anchor ^= 1;
    fields[1].field.seqNo = 2;
    fields[1].field.isFirstField = false;
    fields[1].data = generateField(opts);
    return fields;
}

int testDecoderCore() {
    const auto vp = secamParams();
    const auto fields = makeFields(GenOpts{});

    chd::decoders::secam::SecamDecoder::SecamConfiguration cfg;
    chd::decoders::secam::SecamDecoder decoder(cfg);
    REQUIRE(decoder.configure(vp));

    std::vector<chd::output::ComponentFrame> frames(1);
    decoder.decodeFrames(fields, 0, 2, frames);
    const auto &frame = frames[0];

    REQUIRE(frame.chromaIdent.valid);
    REQUIRE(frame.chromaIdent.confidence > 0.95);
    REQUIRE(frame.chromaIdent.mechanism == 0);  // porch line identification

    const double uExpectScale = kBReduction * kUvRange / 1.505;
    const double vExpectScale = -kRReduction * kUvRange / 1.902;

    int32_t checkedDb = 0, checkedDr = 0;
    for (int32_t frameRow = 100; frameRow <= 560; frameRow += 7) {
        const int32_t fieldLine = frameRow / 2;
        const int32_t anchor = (frameRow % 2 == 0) ? 0 : 1;
        const int32_t expectedComponent = (fieldLine + anchor) & 1;
        REQUIRE(frame.chromaRowComponents[frameRow] == expectedComponent);

        for (int32_t bar = 0; bar < kNumBars; bar++) {
            if (expectedComponent == 0) {
                const double got = barMedian(frame.u(frameRow), bar);
                const double want = kDbBars[bar] * uExpectScale;
                REQUIRE(std::abs(got - want) < std::abs(want) * 0.03 + 0.03 * uExpectScale);
                checkedDb++;
            } else {
                const double got = barMedian(frame.v(frameRow), bar);
                const double want = kDrBars[bar] * vExpectScale;
                REQUIRE(std::abs(got - want) < std::abs(want) * 0.03
                                                   + 0.03 * std::abs(vExpectScale));
                checkedDr++;
            }
        }
    }
    REQUIRE(checkedDb > 100);
    REQUIRE(checkedDr > 100);
    return 0;
}

// ── End-to-end through the public Y/C ABI to a 4:4:0 float frame ────────────

const char *kPalJsonSidecar = R"({
  "videoParameters": {
    "activeVideoEnd": 1107,
    "activeVideoStart": 185,
    "black16bIre": 16384,
    "colourBurstEnd": 138,
    "colourBurstStart": 98,
    "fieldHeight": 313,
    "fieldWidth": 1135,
    "isMapped": false,
    "isSubcarrierLocked": false,
    "isWidescreen": false,
    "numberOfSequentialFields": 2,
    "sampleRate": 17734475,
    "system": "PAL",
    "white16bIre": 54016
  },
  "fields": [
    {"seqNo": 1, "isFirstField": true,  "syncConf": 100, "medianBurstIRE": 25.0},
    {"seqNo": 2, "isFirstField": false, "syncConf": 100, "medianBurstIRE": 25.0}
  ]
})";

// Shared lattice check: every sampled row matches the generator's
// transmitted-line alternation (or its complement when flipped is set).
int checkLattice(const chd::output::ComponentFrame &frame, bool flipped) {
    for (int32_t frameRow = 100; frameRow <= 560; frameRow += 7) {
        const int32_t fieldLine = frameRow / 2;
        const int32_t anchor = (frameRow % 2 == 0) ? 0 : 1;
        const int32_t expected = ((fieldLine + anchor) & 1) ^ (flipped ? 1 : 0);
        REQUIRE(frame.chromaRowComponents[frameRow] == expected);
    }
    return 0;
}

int decodeWith(const chd::decoders::secam::SecamDecoder::SecamConfiguration &cfg,
               const std::vector<chd::decoders::SourceField> &fields,
               std::vector<chd::output::ComponentFrame> &frames) {
    chd::decoders::secam::SecamDecoder decoder(cfg);
    REQUIRE(decoder.configure(secamParams()));
    frames.assign(1, {});
    decoder.decodeFrames(fields, 0, 2, frames);
    return 0;
}

int testManualMode() {
    const auto fields = makeFields(GenOpts{});
    using IdentMode = chd::decoders::secam::SecamDecoder::IdentMode;

    // db_first matches the generator (first active field line 22 is Db when
    // the anchor is 0); the porch-calibrated carriers still apply, so bar
    // values stay exact.
    chd::decoders::secam::SecamDecoder::SecamConfiguration cfg;
    cfg.identMode = IdentMode::Manual;
    cfg.manualFirstComponent = 0;
    std::vector<chd::output::ComponentFrame> frames;
    if (int rc = decodeWith(cfg, fields, frames)) return rc;
    REQUIRE(frames[0].chromaIdent.mechanism == 3);
    REQUIRE(frames[0].chromaIdent.confidence == 1.0);
    if (int rc = checkLattice(frames[0], false)) return rc;
    const double uExpectScale = kBReduction * kUvRange / 1.505;
    const double got = barMedian(frames[0].u(100), 1);
    REQUIRE(std::abs(got - kDbBars[1] * uExpectScale) < 0.05 * uExpectScale);

    // dr_first anchors the complementary lattice.
    cfg.manualFirstComponent = 1;
    if (int rc = decodeWith(cfg, fields, frames)) return rc;
    REQUIRE(frames[0].chromaIdent.mechanism == 3);
    if (int rc = checkLattice(frames[0], true)) return rc;
    return 0;
}

int testBottlesFallback() {
    GenOpts opts;
    opts.porchCarrier = false;
    opts.bottles = true;
    opts.carrierOffset = 0.0;  // no porch to calibrate from; nominal carriers
    const auto fields = makeFields(opts);

    chd::decoders::secam::SecamDecoder::SecamConfiguration cfg;
    std::vector<chd::output::ComponentFrame> frames;
    if (int rc = decodeWith(cfg, fields, frames)) return rc;
    REQUIRE(frames[0].chromaIdent.mechanism == 1);  // bottles
    REQUIRE(frames[0].chromaIdent.confidence > 0.9);
    if (int rc = checkLattice(frames[0], false)) return rc;
    const double uExpectScale = kBReduction * kUvRange / 1.505;
    const double got = barMedian(frames[0].u(100), 1);
    REQUIRE(std::abs(got - kDbBars[1] * uExpectScale) < 0.05 * uExpectScale);

    // Explicit bottles mode gives the same answer.
    cfg.identMode = chd::decoders::secam::SecamDecoder::IdentMode::Bottles;
    if (int rc = decodeWith(cfg, fields, frames)) return rc;
    REQUIRE(frames[0].chromaIdent.mechanism == 1);
    if (int rc = checkLattice(frames[0], false)) return rc;
    return 0;
}

int testContentFallback() {
    GenOpts opts;
    opts.porchCarrier = false;
    opts.carrierOffset = 0.0;
    const auto fields = makeFields(opts);

    chd::decoders::secam::SecamDecoder::SecamConfiguration cfg;
    std::vector<chd::output::ComponentFrame> frames;
    if (int rc = decodeWith(cfg, fields, frames)) return rc;
    REQUIRE(frames[0].chromaIdent.mechanism == 2);  // content statistics
    REQUIRE(frames[0].chromaIdent.confidence > 0.9);
    if (int rc = checkLattice(frames[0], false)) return rc;
    return 0;
}

// Swept-level click concealment against injected dropouts and noise: the
// stage bypasses at level 0, detects the injected envelope collapse at
// every level > 0, and does not fire on clean-but-noisy content.
int testClickConcealment() {
    GenOpts opts;
    opts.noiseAmplitude = 200.0;
    opts.limiter = true;
    auto fields = makeFields(opts);
    // Injected dropout: the chroma envelope collapses on field-0 row 100
    // (frame row 200, a Db line) for samples [460, 580).
    for (int32_t x = 460; x < 580; x++) {
        fields[0].data[static_cast<size_t>(100) * kWidth + x] = 32767;
    }

    chd::decoders::secam::SecamDecoder::SecamConfiguration cfg;
    std::vector<chd::output::ComponentFrame> frames;

    // Level 0: the stage is bypassed entirely.
    cfg.clickNrLevel = 0.0;
    if (int rc = decodeWith(cfg, fields, frames)) return rc;
    REQUIRE(frames[0].chromaConcealedSpans.empty());
    REQUIRE(!frames[0].chromaClick.valid);

    for (const double level : {0.25, 0.5, 0.9}) {
        cfg.clickNrLevel = level;
        if (int rc = decodeWith(cfg, fields, frames)) return rc;
        REQUIRE(frames[0].chromaClick.valid);

        int32_t injectedCoverage = 0;
        int32_t otherRows = 0;
        for (const auto &span : frames[0].chromaConcealedSpans) {
            if (span.frameRow == 200) {
                injectedCoverage += std::max(0, std::min(span.xEnd, 580)
                                                    - std::max(span.xStart, 460));
            } else {
                otherRows++;
            }
        }
        REQUIRE(injectedCoverage >= 60);
        // Clean (noisy) content must not trip the thresholds.
        REQUIRE(otherRows <= 2);

        // The concealed samples carry interpolated, in-range chroma.
        const double uExpectScale = kBReduction * kUvRange / 1.505;
        const double *u = frames[0].u(200);
        for (int32_t x = 480; x < 560; x++) {
            REQUIRE(std::isfinite(u[x]));
            REQUIRE(std::abs(u[x]) < 1.2 * uExpectScale);
        }
    }

    // Expert overrides pin the reported thresholds absolutely.
    cfg.clickNrLevel = 0.5;
    cfg.clickEnvDipDbOverride = 9.0;
    cfg.clickFreqOvershootOverride = 1.7;
    if (int rc = decodeWith(cfg, fields, frames)) return rc;
    REQUIRE(frames[0].chromaClick.envDipDb == 9.0);
    REQUIRE(frames[0].chromaClick.freqOvershoot == 1.7);
    return 0;
}

int testYcAbiDecode(const fs::path &dir) {
    const std::string luma    = (dir / "gen.tbc").string();
    const std::string chroma  = (dir / "gen_chroma.tbc").string();
    const std::string sidecar = (dir / "gen.tbc.json").string();

    // Chroma plane: the generated FM fields. Luma plane: flat mid-grey.
    {
        std::ofstream f(chroma, std::ios::binary);
        REQUIRE(f.good());
        for (int32_t anchor : {0, 1}) {
            GenOpts opts;
            opts.anchor = anchor;
            const auto field = generateField(opts);
            f.write(reinterpret_cast<const char *>(field.data()), field.size() * 2);
        }
    }
    {
        std::ofstream f(luma, std::ios::binary);
        REQUIRE(f.good());
        const std::vector<uint16_t> grey(static_cast<size_t>(kWidth) * kHeight, 35200);
        for (int32_t i = 0; i < 2; i++) {
            f.write(reinterpret_cast<const char *>(grey.data()), grey.size() * 2);
        }
    }
    {
        std::ofstream f(sidecar);
        REQUIRE(f.good());
        f << kPalJsonSidecar;
    }

    chd_video_params_t params{};
    params.standard = CHD_STD_SECAM;
    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_yc(luma.c_str(), chroma.c_str(), nullptr, &params, &video)
            == CHD_OK);

    chd_decoder_t *dec = nullptr;
    REQUIRE(chd_decoder_create(video, CHD_DEC_AUTO, &dec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "yuv440ps") == CHD_OK);
    REQUIRE(chd_decoder_commit(dec) == CHD_OK);

    chd_frame_t *frame = nullptr;
    REQUIRE(chd_decode_frame(dec, 0, &frame) == CHD_OK);
    chd_frame_info_t info{};
    REQUIRE(chd_frame_get_info(frame, &info) == CHD_OK);
    REQUIRE(info.format == CHD_PIXEL_YUV440PS);
    REQUIRE(info.width == kActiveEnd - kActiveStart);
    REQUIRE(info.height == 576);

    chd_plane_info_t cbInfo{}, crInfo{};
    REQUIRE(chd_frame_get_plane_info(frame, CHD_PLANE_CB, &cbInfo) == CHD_OK);
    REQUIRE(chd_frame_get_plane_info(frame, CHD_PLANE_CR, &crInfo) == CHD_OK);
    REQUIRE(cbInfo.height + crInfo.height == 576);
    REQUIRE(cbInfo.height == crInfo.height);

    // Luma: flat grey through the Mono kind.
    const float *yData = nullptr;
    ptrdiff_t yStride = 0;
    REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_Y, &yData, &yStride) == CHD_OK);
    const double expectY = (35200.0 - 16384.0) / (54016.0 - 16384.0);
    REQUIRE(std::abs(yData[100 * info.width + 200] - expectY) < 0.02);

    // Component lattice matches the generator's anchors, frame-row indexed
    // over the active region (frame row = active row + 44).
    chd_chroma_row_component_t comp;
    for (int32_t outRow = 20; outRow < 550; outRow += 13) {
        const int32_t frameRow = outRow + 44;
        const int32_t fieldLine = frameRow / 2;
        const int32_t anchor = (frameRow % 2 == 0) ? 0 : 1;
        const auto expected = ((fieldLine + anchor) & 1) == 0 ? CHD_CHROMA_ROW_DB
                                                              : CHD_CHROMA_ROW_DR;
        REQUIRE(chd_frame_chroma_row_component(frame, outRow, &comp) == CHD_OK);
        REQUIRE(comp == expected);
    }

    // Chroma plane values: walk the Cb plane against the output-row lattice
    // and check bar 1 (D'B = 0.8) at its centre. E'Cb = D'B / (1.505 * 1.772).
    // The plane is woven by output-row parity, so a Db line at output row r
    // sits at plane row 2j+p, where p = r & 1 (no top border here) and j
    // counts the Db lines of that parity above it.
    const float *cbData = nullptr;
    ptrdiff_t cbStride = 0;
    REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_CB, &cbData, &cbStride) == CHD_OK);
    const int32_t barWidth = (kActiveEnd - kActiveStart) / kNumBars;
    const int32_t barX = barWidth + barWidth / 2;  // bar 1 centre, active-relative
    const double expectECb = 0.8 / (1.505 * 2.0 * (1.0 - 0.114));
    int32_t seen[2] = {0, 0};
    int32_t checked = 0;
    for (int32_t outRow = 0; outRow < info.height; outRow++) {
        if (chd_frame_chroma_row_component(frame, outRow, &comp) != CHD_OK) continue;
        if (comp != CHD_CHROMA_ROW_DB) continue;
        const int32_t parity = outRow & 1;
        const int32_t planeRow = 2 * seen[parity] + parity;
        seen[parity]++;
        if (outRow >= 40 && outRow < 540) {
            const float got = cbData[planeRow * info.width + barX];
            REQUIRE(std::abs(got - expectECb) < 0.03 * expectECb + 0.01);
            checked++;
        }
    }
    REQUIRE(seen[0] + seen[1] == cbInfo.height);
    REQUIRE(seen[0] == seen[1]);
    REQUIRE(checked > 200);

    chd_chroma_ident_report_t report{};
    REQUIRE(chd_frame_get_chroma_ident(frame, &report) == CHD_OK);
    REQUIRE(report.mechanism == CHD_CHROMA_IDENT_PORCH);
    REQUIRE(report.confidence > 0.95);

    chd_frame_free(frame);
    chd_decoder_free(dec);

    // Ident option surface: gating, value validation, and the manual-mode
    // cross-option requirement.
    chd_decoder_t *vdec = nullptr;
    REQUIRE(chd_decoder_create(video, CHD_DEC_SECAM, &vdec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(vdec, CHD_OPT_OUTPUT_FORMAT, "yuv440ps") == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(vdec, CHD_OPT_CHROMA_IDENT_MODE, "manual") == CHD_OK);
    REQUIRE(chd_decoder_commit(vdec) != CHD_OK);  // manual requires the anchor
    chd_decoder_free(vdec);

    REQUIRE(chd_decoder_create(video, CHD_DEC_SECAM, &vdec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(vdec, CHD_OPT_OUTPUT_FORMAT, "yuv440ps") == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(vdec, CHD_OPT_CHROMA_IDENT_MANUAL, "db_first")
            == CHD_OK);
    REQUIRE(chd_decoder_commit(vdec) != CHD_OK);  // anchor without manual mode
    chd_decoder_free(vdec);

    REQUIRE(chd_decoder_create(video, CHD_DEC_SECAM, &vdec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(vdec, CHD_OPT_OUTPUT_FORMAT, "yuv440ps") == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(vdec, CHD_OPT_CHROMA_IDENT_MODE, "always") == CHD_OK);
    REQUIRE(chd_decoder_commit(vdec) != CHD_OK);  // unknown mode value
    chd_decoder_free(vdec);

    REQUIRE(chd_decoder_create(video, CHD_DEC_SECAM, &vdec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(vdec, CHD_OPT_OUTPUT_FORMAT, "yuv440ps") == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(vdec, CHD_OPT_CHROMA_IDENT_MODE, "manual") == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(vdec, CHD_OPT_CHROMA_IDENT_MANUAL, "db_first")
            == CHD_OK);
    REQUIRE(chd_decoder_commit(vdec) == CHD_OK);
    chd_frame_t *manualFrame = nullptr;
    REQUIRE(chd_decode_frame(vdec, 0, &manualFrame) == CHD_OK);
    REQUIRE(chd_frame_get_chroma_ident(manualFrame, &report) == CHD_OK);
    REQUIRE(report.mechanism == CHD_CHROMA_IDENT_MANUAL);
    REQUIRE(report.confidence == 1.0);
    chd_frame_free(manualFrame);
    chd_decoder_free(vdec);

    // The ident options are SECAM-scoped.
    REQUIRE(chd_decoder_create(video, CHD_DEC_PAL_2D, &vdec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(vdec, CHD_OPT_CHROMA_IDENT_MODE, "auto")
            == CHD_E_INVALID_ARG);
    chd_decoder_free(vdec);

    chd_video_free(video);

    fs::remove(luma);
    fs::remove(chroma);
    fs::remove(sidecar);
    return 0;
}

// Click concealment through the public ABI: the origin-tagged unified span
// stream and the effective-threshold diagnostics.
int testClickAbi(const fs::path &dir) {
    const std::string luma    = (dir / "click.tbc").string();
    const std::string chroma  = (dir / "click_chroma.tbc").string();
    const std::string sidecar = (dir / "click.tbc.json").string();
    {
        std::ofstream f(chroma, std::ios::binary);
        REQUIRE(f.good());
        GenOpts opts;
        opts.noiseAmplitude = 200.0;
        opts.limiter = true;
        auto field0 = generateField(opts);
        for (int32_t x = 460; x < 580; x++) {
            field0[static_cast<size_t>(100) * kWidth + x] = 32767;
        }
        opts.anchor = 1;
        const auto field1 = generateField(opts);
        f.write(reinterpret_cast<const char *>(field0.data()), field0.size() * 2);
        f.write(reinterpret_cast<const char *>(field1.data()), field1.size() * 2);
    }
    {
        std::ofstream f(luma, std::ios::binary);
        REQUIRE(f.good());
        const std::vector<uint16_t> grey(static_cast<size_t>(kWidth) * kHeight, 35200);
        for (int32_t i = 0; i < 2; i++) {
            f.write(reinterpret_cast<const char *>(grey.data()), grey.size() * 2);
        }
    }
    {
        std::ofstream f(sidecar);
        REQUIRE(f.good());
        f << kPalJsonSidecar;
    }

    chd_video_params_t params{};
    params.standard = CHD_STD_SECAM;
    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_yc(luma.c_str(), chroma.c_str(), nullptr, &params, &video)
            == CHD_OK);

    // An expert override with the stage explicitly disabled is rejected at
    // commit; without the level it rides the enabled-by-default stage.
    chd_decoder_t *dec = nullptr;
    REQUIRE(chd_decoder_create(video, CHD_DEC_SECAM, &dec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "yuv440ps") == CHD_OK);
    REQUIRE(chd_decoder_set_option_f64(dec, CHD_OPT_CHROMA_CLICK_NR_LEVEL, 0.0) == CHD_OK);
    REQUIRE(chd_decoder_set_option_f64(dec, CHD_OPT_CHROMA_CLICK_ENV_DIP_DB, 12.0)
            == CHD_OK);
    REQUIRE(chd_decoder_commit(dec) != CHD_OK);
    chd_decoder_free(dec);

    REQUIRE(chd_decoder_create(video, CHD_DEC_SECAM, &dec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "yuv440ps") == CHD_OK);
    REQUIRE(chd_decoder_set_option_f64(dec, CHD_OPT_CHROMA_CLICK_ENV_DIP_DB, 12.0)
            == CHD_OK);
    REQUIRE(chd_decoder_commit(dec) == CHD_OK);
    chd_decoder_free(dec);

    REQUIRE(chd_decoder_create(video, CHD_DEC_SECAM, &dec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "yuv440ps") == CHD_OK);
    REQUIRE(chd_decoder_set_option_f64(dec, CHD_OPT_CHROMA_CLICK_NR_LEVEL, 0.5) == CHD_OK);
    REQUIRE(chd_decoder_commit(dec) == CHD_OK);

    // Thresholds report only after a decode has run.
    double dipDb = 0.0, overshoot = 0.0;
    REQUIRE(chd_decoder_get_chroma_click_thresholds(dec, &dipDb, &overshoot)
            == CHD_E_UNSUPPORTED);

    chd_frame_t *frame = nullptr;
    REQUIRE(chd_decode_frame(dec, 0, &frame) == CHD_OK);
    chd_frame_free(frame);

    REQUIRE(chd_decoder_get_chroma_click_thresholds(dec, &dipDb, &overshoot) == CHD_OK);
    REQUIRE(std::abs(dipDb - 9.0) < 1e-9);  // 12 - 6 * 0.5
    REQUIRE(overshoot > 1.7);                // 2.6 - 1.6 * 0.5 + noise term

    // The unified span stream carries the concealment span, origin-tagged,
    // in output coordinates (frame row 200 - 44, x - activeVideoStart).
    chd_dropout_span_t *spans = nullptr;
    size_t count = 0;
    REQUIRE(chd_decoder_get_dropout_spans(dec, 0, CHD_DROPOUT_DETECTED, &spans, &count)
            == CHD_OK);
    int32_t coverage = 0;
    for (size_t i = 0; i < count; i++) {
        if (spans[i].origin != CHD_DROPOUT_ORIGIN_DECODER_CONCEALMENT) continue;
        if (spans[i].y != 156) continue;
        coverage += std::max(0, std::min(spans[i].x_end, 395)
                                    - std::max(spans[i].x_start, 275));
    }
    REQUIRE(coverage >= 50);
    chd_dropout_spans_free(spans);

    chd_decoder_free(dec);
    chd_video_free(video);
    fs::remove(luma);
    fs::remove(chroma);
    fs::remove(sidecar);
    return 0;
}

// A composite field in the CVBS canonical domain: blanking 16384, a flat
// mid-grey luma pedestal over the active region, and the FM chroma block
// as an excursion on top.
constexpr double kPedestal16b = 18816.0;  // E'Y = 0.5 against the PAL levels

std::vector<uint16_t> compositeFromChroma(const std::vector<uint16_t> &chromaField) {
    std::vector<uint16_t> out(chromaField.size());
    for (int32_t row = 0; row < kHeight; row++) {
        for (int32_t x = 0; x < kWidth; x++) {
            const size_t i = static_cast<size_t>(row) * kWidth + x;
            const double base =
                16384.0 + ((x >= kActiveStart && x < kActiveEnd) ? kPedestal16b : 0.0);
            const double v = base + static_cast<double>(chromaField[i]) - 32767.0;
            out[i] = static_cast<uint16_t>(std::clamp(v, 0.0, 65535.0));
        }
    }
    return out;
}

int checkCvbsDecode(chd_video_t *video) {
    chd_video_info_t vinfo{};
    REQUIRE(chd_video_get_info(video, &vinfo) == CHD_OK);
    REQUIRE(vinfo.standard == CHD_STD_SECAM);
    REQUIRE(vinfo.fsc_hz == 4286000.0);
    REQUIRE(vinfo.is_subcarrier_locked == 0);

    chd_decoder_t *dec = nullptr;
    REQUIRE(chd_decoder_create(video, CHD_DEC_AUTO, &dec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "yuv440ps") == CHD_OK);
    REQUIRE(chd_decoder_commit(dec) == CHD_OK);

    chd_frame_t *frame = nullptr;
    REQUIRE(chd_decode_frame(dec, 0, &frame) == CHD_OK);
    chd_frame_info_t info{};
    REQUIRE(chd_frame_get_info(frame, &info) == CHD_OK);
    REQUIRE(info.format == CHD_PIXEL_YUV440PS);

    chd_chroma_ident_report_t report{};
    REQUIRE(chd_frame_get_chroma_ident(frame, &report) == CHD_OK);
    REQUIRE(report.mechanism == CHD_CHROMA_IDENT_PORCH);
    REQUIRE(report.confidence > 0.9);

    // Luma: the flat pedestal recovered subtractively from the composite.
    const float *yData = nullptr;
    ptrdiff_t stride = 0;
    REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_Y, &yData, &stride) == CHD_OK);
    const int32_t midX = info.width / 2;
    REQUIRE(std::abs(yData[100 * info.width + midX] - 0.5) < 0.05);

    // Chroma: bar 1 (D'B = 0.8) at its centre, walking the Cb plane against
    // the row lattice. Bar positions are generator-absolute; the preset
    // decides the active crop, so translate through first_active_sample.
    chd_plane_info_t cbInfo{};
    REQUIRE(chd_frame_get_plane_info(frame, CHD_PLANE_CB, &cbInfo) == CHD_OK);
    const float *cbData = nullptr;
    REQUIRE(chd_frame_get_plane_float(frame, CHD_PLANE_CB, &cbData, &stride) == CHD_OK);
    const int32_t barWidth = (kActiveEnd - kActiveStart) / kNumBars;
    const int32_t barX = kActiveStart + barWidth + barWidth / 2 - vinfo.first_active_sample;
    REQUIRE(barX > 0 && barX < info.width);
    const double expectECb = 0.8 / (1.505 * 2.0 * (1.0 - 0.114));
    chd_chroma_row_component_t comp;
    int32_t seen[2] = {0, 0};
    int32_t checked = 0;
    for (int32_t outRow = 0; outRow < info.height; outRow++) {
        if (chd_frame_chroma_row_component(frame, outRow, &comp) != CHD_OK) continue;
        if (comp != CHD_CHROMA_ROW_DB) continue;
        const int32_t parity = outRow & 1;
        const int32_t planeRow = 2 * seen[parity] + parity;
        seen[parity]++;
        if (outRow >= 40 && outRow < 540) {
            const float got = cbData[planeRow * info.width + barX];
            REQUIRE(std::abs(got - expectECb) < 0.05 * expectECb + 0.01);
            checked++;
        }
    }
    REQUIRE(seen[0] + seen[1] == cbInfo.height);
    REQUIRE(seen[0] == seen[1]);
    REQUIRE(checked > 200);

    chd_frame_free(frame);
    chd_decoder_free(dec);
    return 0;
}

// SECAM over the CVBS composite path: stored under the byte-compatible PAL
// preset at a NONSTANDARD signal state, declared SECAM at open time.
int testCvbsComposite(const fs::path &dir) {
    const std::string path = (dir / "gen.cvbs").string();
    {
        std::ofstream f(path, std::ios::binary);
        REQUIRE(f.good());
        for (int32_t anchor : {0, 1}) {
            GenOpts opts;
            opts.anchor = anchor;
            opts.carrierOffset = 0.0;
            const auto composite = compositeFromChroma(generateField(opts));
            f.write(reinterpret_cast<const char *>(composite.data()),
                    composite.size() * 2);
        }
    }

    chd_video_params_t params{};
    params.standard = CHD_STD_SECAM;
    params.encoding = CHD_ENC_CVBS_U16_4FSC;
    params.signal_state = CHD_SIG_NONSTANDARD_TBC_UNLOCKED;
    params.layout = CHD_FRAME_LAYOUT_FIELD_RASTER;
    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_composite(path.c_str(), nullptr, &params, &video) == CHD_OK);
    if (int rc = checkCvbsDecode(video)) return rc;
    chd_video_free(video);
    fs::remove(path);
    return 0;
}

// SECAM over the CVBS .y/.c pair: the .c holds the FM block oscillating
// about the centred-chroma zero; the reader recombines a composite.
int testCvbsYcPair(const fs::path &dir) {
    const std::string yPath = (dir / "gen.cvbsy").string();
    const std::string cPath = (dir / "gen.cvbsc").string();
    {
        std::ofstream fy(yPath, std::ios::binary);
        std::ofstream fc(cPath, std::ios::binary);
        REQUIRE(fy.good());
        REQUIRE(fc.good());
        std::vector<uint16_t> luma(static_cast<size_t>(kWidth) * kHeight);
        for (int32_t row = 0; row < kHeight; row++) {
            for (int32_t x = 0; x < kWidth; x++) {
                luma[static_cast<size_t>(row) * kWidth + x] = static_cast<uint16_t>(
                    16384.0
                    + ((x >= kActiveStart && x < kActiveEnd) ? kPedestal16b : 0.0));
            }
        }
        for (int32_t anchor : {0, 1}) {
            GenOpts opts;
            opts.anchor = anchor;
            opts.carrierOffset = 0.0;
            const auto chroma = generateField(opts);
            fy.write(reinterpret_cast<const char *>(luma.data()), luma.size() * 2);
            fc.write(reinterpret_cast<const char *>(chroma.data()), chroma.size() * 2);
        }
    }

    chd_video_params_t params{};
    params.standard = CHD_STD_SECAM;
    params.encoding = CHD_ENC_CVBS_U16_4FSC;
    params.signal_state = CHD_SIG_NONSTANDARD_TBC_UNLOCKED;
    params.layout = CHD_FRAME_LAYOUT_FIELD_RASTER;
    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_yc(yPath.c_str(), cPath.c_str(), nullptr, &params, &video)
            == CHD_OK);
    if (int rc = checkCvbsDecode(video)) return rc;
    chd_video_free(video);
    fs::remove(yPath);
    fs::remove(cPath);
    return 0;
}

// Tape-fixture checks against a local vhs-decode ME-SECAM capture (a Y/C
// pair with a shared sidecar). Local/manual like the other tape fixtures:
// self-skips unless CHD_TEST_MESECAM_TBC points at the luma .tbc.
int testLocalMesecamFixture() {
    const char *lumaPath = std::getenv("CHD_TEST_MESECAM_TBC");
    if (lumaPath == nullptr) {
        std::cout << "CHD_TEST_MESECAM_TBC unset; skipping tape-fixture checks\n";
        return 0;
    }
    std::string chromaPath = lumaPath;
    const size_t ext = chromaPath.rfind(".tbc");
    REQUIRE(ext != std::string::npos);
    chromaPath.insert(ext, "_chroma");

    chd_video_params_t params{};
    params.standard = CHD_STD_SECAM;
    chd_video_t *video = nullptr;
    REQUIRE(chd_video_open_yc(lumaPath, chromaPath.c_str(), nullptr, &params, &video)
            == CHD_OK);

    chd_decoder_t *dec = nullptr;
    REQUIRE(chd_decoder_create(video, CHD_DEC_AUTO, &dec) == CHD_OK);
    REQUIRE(chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "yuv440ps") == CHD_OK);
    REQUIRE(chd_decoder_commit(dec) == CHD_OK);

    // Frame 0: porch ident with solid confidence, honest 4:4:0 geometry.
    chd_frame_t *frame0 = nullptr;
    REQUIRE(chd_decode_frame(dec, 0, &frame0) == CHD_OK);
    chd_chroma_ident_report_t report{};
    REQUIRE(chd_frame_get_chroma_ident(frame0, &report) == CHD_OK);
    REQUIRE(report.mechanism == CHD_CHROMA_IDENT_PORCH);
    REQUIRE(report.confidence > 0.8);
    chd_frame_info_t info{};
    REQUIRE(chd_frame_get_info(frame0, &info) == CHD_OK);
    chd_plane_info_t cbInfo{}, crInfo{};
    REQUIRE(chd_frame_get_plane_info(frame0, CHD_PLANE_CB, &cbInfo) == CHD_OK);
    REQUIRE(chd_frame_get_plane_info(frame0, CHD_PLANE_CR, &crInfo) == CHD_OK);
    REQUIRE(cbInfo.height + crInfo.height == info.height);
    REQUIRE(cbInfo.height == crInfo.height);

    // Frame parity: 625 is odd, so a given output row's component flips
    // frame to frame (the BR.469 four-field cycle).
    chd_frame_t *frame1 = nullptr;
    REQUIRE(chd_decode_frame(dec, 1, &frame1) == CHD_OK);
    int32_t flipped = 0, sampled = 0;
    for (int32_t row = 40; row < info.height - 40; row += 11) {
        chd_chroma_row_component_t c0, c1;
        if (chd_frame_chroma_row_component(frame0, row, &c0) != CHD_OK) continue;
        if (chd_frame_chroma_row_component(frame1, row, &c1) != CHD_OK) continue;
        if (c0 != c1) flipped++;
        sampled++;
    }
    REQUIRE(sampled > 30);
    REQUIRE(flipped * 10 >= sampled * 9);

    chd_frame_free(frame0);
    chd_frame_free(frame1);
    chd_decoder_free(dec);
    chd_video_free(video);
    std::cout << "tape-fixture checks ran against " << lumaPath << "\n";
    return 0;
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "chd_secam_decoder_test";
    fs::create_directories(dir);

    int rc = testDecoderCore();
    if (rc == 0) rc = testManualMode();
    if (rc == 0) rc = testBottlesFallback();
    if (rc == 0) rc = testContentFallback();
    if (rc == 0) rc = testClickConcealment();
    if (rc == 0) rc = testYcAbiDecode(dir);
    if (rc == 0) rc = testClickAbi(dir);
    if (rc == 0) rc = testCvbsComposite(dir);
    if (rc == 0) rc = testCvbsYcPair(dir);
    if (rc == 0) rc = testLocalMesecamFixture();

    fs::remove_all(dir);
    if (rc == 0) std::cout << "PASS\n";
    return rc;
}
