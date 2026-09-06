// SPDX-License-Identifier: GPL-3.0-or-later
//
// HVD decoder bridge against a synthetic NTSC generator built from the
// physical definitions the engine assumes (the same construction as
// hvd-core's own chroma_reference_test):
//
//   S = Y + U sin(phi) + V cos(phi),  phi = theta[line] + (pi/2) x
//
// with the burst carried on the -U axis and theta advancing 180 deg per
// field line. A flat known colour must come back through configure ->
// decodeFrames as composite-scale Y/U/V in the ComponentFrame, which pins
// the whole bridge: code -> true-IRE conversion (blanking-referenced),
// field weave and active-crop row mapping, chi = V - iU sign conventions,
// and the caller-applied chroma_gain * ACC gain. Also checks the
// SECAM-rejection arm of configure.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../../src/decoders/hvd/hvd_decoder.h"
#include "../../src/decoders/source_field.h"
#include "../../src/metadata/core.h"
#include "../../src/output/component_frame.h"

#include "engine/ntsc_geometry.h"

namespace {

#define REQUIRE(cond)                                                            \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #cond \
                      << "\n";                                                   \
            return 1;                                                            \
        }                                                                        \
    } while (0)

constexpr double kPi = 3.14159265358979323846;

// Compact synthetic geometry (hvd-core's engine tests use the same shape);
// theta is synthesized at the tabulated NTSC 180 deg/line advance rather
// than derived from the shortened line, matching the engine's phase model.
constexpr int32_t kFieldWidth = 300;
constexpr int32_t kFieldHeight = 40;
constexpr int32_t kBurstStart = 20;
constexpr int32_t kBurstEnd = 60;
constexpr int32_t kActiveStart = 80;
constexpr int32_t kActiveEnd = 280;
constexpr int32_t kFirstActiveFrameLine = 10;
constexpr int32_t kLastActiveFrameLine = 69;

// Real NTSC-M 16-bit levels: blanking-referenced true IRE, black 7.5 IRE up.
constexpr int32_t kBlanking16 = 16384;
constexpr int32_t kWhite16 = 54016;
constexpr double kCodesPerIre = (kWhite16 - kBlanking16) / 100.0;
constexpr int32_t kBlack16 = kBlanking16 + static_cast<int32_t>(7.5 * kCodesPerIre);

// The flat test colour (IRE) and the nominal NTSC burst (so ACC gain is 1).
constexpr double kY = 50.0;
constexpr double kU = -18.0;
constexpr double kV = 25.0;
constexpr double kBurstIre = 20.0;

chd::metadata::LdDecodeMetaData::VideoParameters ntscParams() {
    chd::metadata::LdDecodeMetaData::VideoParameters vp;
    vp.system = chd::metadata::NTSC;
    vp.fSC = hvd::kFscNtsc;
    vp.sampleRate = hvd::kFs4Fsc;
    vp.fieldWidth = kFieldWidth;
    vp.fieldHeight = kFieldHeight;
    vp.colourBurstStart = kBurstStart;
    vp.colourBurstEnd = kBurstEnd;
    vp.activeVideoStart = kActiveStart;
    vp.activeVideoEnd = kActiveEnd;
    vp.firstActiveFrameLine = kFirstActiveFrameLine;
    vp.lastActiveFrameLine = kLastActiveFrameLine;
    vp.white16bIre = kWhite16;
    vp.black16bIre = kBlack16;
    vp.blanking16bIre = kBlanking16;
    vp.numberOfSequentialFields = 2;
    vp.isValid = true;
    return vp;
}

// One field in 16-bit codes. `theta0` sets the field's carrier phase at
// line 0; the burst rides the -U axis exactly as an encoder emits it.
chd::decoders::SourceField makeField(double theta0, bool isFirstField) {
    chd::decoders::SourceField sf;
    sf.field.isFirstField = isFirstField;
    sf.data.assign(static_cast<size_t>(kFieldWidth) * kFieldHeight, 0);
    for (int32_t line = 0; line < kFieldHeight; line++) {
        const double theta = theta0 + kPi * line;
        for (int32_t x = 0; x < kFieldWidth; x++) {
            const double phi = theta + (kPi / 2.0) * x;
            double ire = 0.0;
            if (x >= kBurstStart && x < kBurstEnd) {
                ire = -kBurstIre * std::sin(phi);
            }
            if (x >= kActiveStart && x < kActiveEnd) {
                ire = kY + kU * std::sin(phi) + kV * std::cos(phi);
            }
            const double code = kBlanking16 + ire * kCodesPerIre;
            sf.data[static_cast<size_t>(line) * kFieldWidth + x] =
                static_cast<uint16_t>(std::lround(code));
        }
    }
    return sf;
}

}  // namespace

int main() {
    const auto vp = ntscParams();

    // SECAM is the one system with no HVD carrier model.
    {
        chd::decoders::hvd::HvdDecoder::HvdConfiguration hc;
        chd::decoders::hvd::HvdDecoder rejecting(hc);
        auto secamVp = vp;
        secamVp.system = chd::metadata::SECAM;
        REQUIRE(!rejecting.configure(secamVp));
    }

    chd::decoders::hvd::HvdDecoder::HvdConfiguration hc;
    chd::decoders::hvd::HvdDecoder decoder(hc);
    REQUIRE(decoder.configure(vp));

    std::vector<chd::decoders::SourceField> fields;
    fields.push_back(makeField(0.35, true));
    fields.push_back(makeField(0.75, false));

    std::vector<chd::output::ComponentFrame> frames(1);
    decoder.decodeFrames(fields, 0, 2, frames);

    // Interior of the active picture, away from the arbitration's edge
    // support and the chroma bandwidth crop's horizontal transient.
    const int32_t yLo = kFirstActiveFrameLine + 8;
    const int32_t yHi = kLastActiveFrameLine - 8;
    const int32_t xLo = kActiveStart + 24;
    const int32_t xHi = kActiveEnd - 24;

    const double expectY = kBlanking16 + kY * kCodesPerIre;
    const double expectU = kU * kCodesPerIre;
    const double expectV = kV * kCodesPerIre;
    const double tolerance = 1.0 * kCodesPerIre;  // 1 IRE

    for (int32_t y = yLo; y <= yHi; y++) {
        const double *rowY = frames[0].y(y);
        const double *rowU = frames[0].u(y);
        const double *rowV = frames[0].v(y);
        for (int32_t x = xLo; x < xHi; x++) {
            REQUIRE(std::abs(rowY[x] - expectY) < tolerance);
            REQUIRE(std::abs(rowU[x] - expectU) < tolerance);
            REQUIRE(std::abs(rowV[x] - expectV) < tolerance);
        }
    }

    // Outside the active frame-line crop nothing may be written: init()
    // leaves black luma and zero chroma there.
    {
        const double *rowU = frames[0].u(kFirstActiveFrameLine - 1);
        const double *rowV = frames[0].v(kLastActiveFrameLine + 1);
        for (int32_t x = kActiveStart; x < kActiveEnd; x++) {
            REQUIRE(rowU[x] == 0.0);
            REQUIRE(rowV[x] == 0.0);
        }
    }

    // The 3D kind: same flat colour through the field-granularity sequence
    // pipeline. Static content, so the cross-field terms must agree with the
    // 2D answer. The field window mirrors what SourceField::loadFields hands
    // a look-behind/look-ahead of one frame: three real frames, core in the
    // middle; each field gets its own arbitrary carrier phase, since the
    // engine measures per-line phase from the burst rather than assuming
    // continuity.
    {
        chd::decoders::hvd::HvdDecoder::HvdConfiguration hc3;
        hc3.temporal3d = true;
        chd::decoders::hvd::HvdDecoder decoder3(hc3);
        REQUIRE(decoder3.configure(vp));
        REQUIRE(decoder3.getLookBehind() == 1);
        REQUIRE(decoder3.getLookAhead() == 1);

        std::vector<chd::decoders::SourceField> window;
        for (int32_t i = 0; i < 6; i++) {
            window.push_back(makeField(0.35 + 0.4 * i, (i % 2) == 0));
        }

        std::vector<chd::output::ComponentFrame> coreFrames(1);
        decoder3.decodeFrames(window, 2, 4, coreFrames);

        for (int32_t y = yLo; y <= yHi; y++) {
            const double *rowY = coreFrames[0].y(y);
            const double *rowU = coreFrames[0].u(y);
            const double *rowV = coreFrames[0].v(y);
            for (int32_t x = xLo; x < xHi; x++) {
                REQUIRE(std::abs(rowY[x] - expectY) < tolerance);
                REQUIRE(std::abs(rowU[x] - expectU) < tolerance);
                REQUIRE(std::abs(rowV[x] - expectV) < tolerance);
            }
        }
    }

    std::cout << "hvd_decoder: all checks passed\n";
    return 0;
}
