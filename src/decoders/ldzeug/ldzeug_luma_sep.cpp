// SPDX-License-Identifier: GPL-3.0-or-later

#include "ldzeug_luma_sep.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../../common/log.h"
#include "../../nn/inference_engine.h"
#include "../filter/deemp.h"
#include "../filter/firfilter.h"
#include "../ntsc_burst.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chd::decoders::ldzeug {

namespace {

constexpr double IQ_BASE_ANGLE_DEG = 33.0;
constexpr double FROM_NORMALIZED   = 65535.0;

inline void applyUvFromIq(double i, double q, double bp, double bq,
                          double *u, double *v) {
    *u = -bp * i + bq * q;
    *v =  bq * i + bp * q;
}

// NTSC line phase, matches Comb::FrameBuffer::getLinePhase. `line` is the
// 0-based interlaced-frame line; even lines come from the first field,
// odd from the second.
inline bool getLinePhase(int32_t line, int32_t firstFieldPhaseID, int32_t secondFieldPhaseID) {
    const bool isFirstField = ((line % 2) == 0);
    const int32_t fieldID = isFirstField ? firstFieldPhaseID : secondFieldPhaseID;
    const bool isPositivePhaseOnEvenLines = (fieldID == 1) || (fieldID == 4);
    const int32_t fieldLine = line / 2;
    const bool isEvenLine = (fieldLine % 2) == 0;
    return isEvenLine ? isPositivePhaseOnEvenLines : !isPositivePhaseOnEvenLines;
}

struct FramePhaseIDs { int32_t first; int32_t second; };

FramePhaseIDs framePhaseIDs(const chd::decoders::SourceField &fa,
                            const chd::decoders::SourceField &fb) {
    if (fa.getOffset() == 0) {
        return { fa.field.fieldPhaseID, fb.field.fieldPhaseID };
    }
    return { fb.field.fieldPhaseID, fa.field.fieldPhaseID };
}

// Per-row I/Q demod + optional bandpass + uv_from_iq rotation. The Y row
// is written for every pixel; U/V are zero outside the horizontal active
// range. Mirrors Comb::FrameBuffer::splitIQ but operates on `cvbs − Y`
// rather than raw composite.
//
// `burstDev` rotates the demodulated I/Q to undo the line's measured
// departure from the nominal subcarrier phase. The default identity
// rotation leaves the nominal-phase result untouched.
void demodChromaRow(const uint16_t *cvbsRow,
                    const float    *yNormRow,
                    int32_t fieldWidth,
                    int32_t activeStart, int32_t activeEnd,
                    int32_t frameLine,
                    int32_t firstFieldPhaseID, int32_t secondFieldPhaseID,
                    bool chromaBandpass,
                    const chd::decoders::BurstDeviation &burstDev,
                    double bp, double bq,
                    std::vector<double> &iWork,
                    std::vector<double> &qWork,
                    std::vector<double> &iFilt,
                    std::vector<double> &qFilt,
                    double *yRow, double *uRow, double *vRow)
{
    for (int32_t x = 0; x < fieldWidth; ++x) {
        yRow[x] = static_cast<double>(yNormRow[x]) * FROM_NORMALIZED;
        uRow[x] = 0.0;
        vRow[x] = 0.0;
    }
    if (activeStart >= activeEnd) return;

    const int32_t activeLen = activeEnd - activeStart;
    if (static_cast<int32_t>(iWork.size()) < activeLen) {
        iWork.resize(activeLen);
        qWork.resize(activeLen);
        iFilt.resize(activeLen);
        qFilt.resize(activeLen);
    }

    const bool linePhase = getLinePhase(frameLine, firstFieldPhaseID, secondFieldPhaseID);

    // Quadrature switch mirroring Comb::FrameBuffer::splitIQ. si/sq carry
    // across the row so every pixel ends up with the most recent I and Q
    // samples; the FIR LP step (when enabled) smooths the resulting steps.
    double si = 0.0, sq = 0.0;
    for (int32_t h = activeStart; h < activeEnd; ++h) {
        const double C    = static_cast<double>(cvbsRow[h]) - yRow[h];
        const double cavg = linePhase ? -C : C;
        switch (h % 4) {
            case 0: sq =  cavg; break;
            case 1: si = -cavg; break;
            case 2: sq = -cavg; break;
            case 3: si =  cavg; break;
            default: break;
        }
        iWork[h - activeStart] = si;
        qWork[h - activeStart] = sq;
    }

    // Undo the line's measured phase error before the bandpass, so the FIR
    // sees the same burst-locked axes the rotation targets.
    if (burstDev.valid) {
        for (int32_t idx = 0; idx < activeLen; ++idx) {
            chd::decoders::correctDemodulatedIQ(burstDev, &iWork[idx], &qWork[idx]);
        }
    }

    const double *iSrc = iWork.data();
    const double *qSrc = qWork.data();
    if (chromaBandpass) {
        auto iqFilter = chd::decoders::filter::makeFIRFilter(chd::decoders::filter::c_colorlp_b);
        iqFilter.apply(iWork.data(), iFilt.data(), activeLen);
        iqFilter.apply(qWork.data(), qFilt.data(), activeLen);
        iSrc = iFilt.data();
        qSrc = qFilt.data();
    }

    for (int32_t h = activeStart; h < activeEnd; ++h) {
        const int32_t idx = h - activeStart;
        applyUvFromIq(iSrc[idx], qSrc[idx], bp, bq, &uRow[h], &vRow[h]);
    }
}

}  // namespace

void LdzeugLumaSepDecoder::decodeFrames(
    const std::vector<chd::decoders::SourceField> &inputFields,
    int32_t startIndex, int32_t endIndex,
    std::vector<chd::output::ComponentFrame> &componentFrames)
{
    if (session_ == nullptr) {
        throw std::runtime_error("ldzeug2_luma_sep: no NN model bound");
    }

    const auto &vp = videoParameters;
    const int32_t fieldWidth  = vp.fieldWidth;
    const int32_t fieldHeight = vp.fieldHeight;
    const int32_t frameHeight = fieldHeight * 2;
    const size_t  fieldStride = static_cast<size_t>(fieldWidth) * fieldHeight;
    const size_t  frameStride = fieldStride * 2;

    const int32_t activeStart     = vp.activeVideoStart;
    const int32_t activeEnd       = vp.activeVideoEnd;
    const int32_t firstActiveLine = vp.firstActiveFrameLine;
    const int32_t lastActiveLine  = vp.lastActiveFrameLine;

    const double theta = (IQ_BASE_ANGLE_DEG + chromaPhase_) * M_PI / 180.0;
    const double bp    = std::sin(theta) * chromaGain_;
    const double bq    = std::cos(theta) * chromaGain_;

    std::vector<float> fieldBuffer(fieldStride);
    std::vector<float> frameBuffer;
    if (mode_ == Mode::Frame) frameBuffer.resize(frameStride);

    std::vector<double> iWork, qWork, iFilt, qFilt;

    const int32_t inputCount = static_cast<int32_t>(inputFields.size());
    const int32_t outFrameCap = static_cast<int32_t>(componentFrames.size());

    for (int32_t frameIdx = startIndex;
         frameIdx + 1 < endIndex && frameIdx + 1 < inputCount;
         frameIdx += 2) {
        const int32_t outFrame = (frameIdx - startIndex) / 2;
        if (outFrame >= outFrameCap) break;
        chd::output::ComponentFrame &out = componentFrames[outFrame];
        out.init(vp);
        // The network runs over the full frameHeight raster, but ComponentFrame
        // drops the second field's trailing padding row. Demod only the rows it
        // stores; the last frame line is discarded, as the comb path does.
        const int32_t storedHeight = out.getHeight();

        const auto phaseIDs = framePhaseIDs(inputFields[frameIdx], inputFields[frameIdx + 1]);

        auto demodLine = [&](int32_t frameLine,
                             const uint16_t *cvbsRow,
                             const float    *yNormRow) {
            const bool inActive = frameLine >= firstActiveLine && frameLine <= lastActiveLine;
            const int32_t hStart = inActive ? activeStart : 0;
            const int32_t hEnd   = inActive ? activeEnd   : 0;
            chd::decoders::BurstDeviation burstDev;
            if (phaseCompensation_ && inActive) {
                // The deviation is measured against the carrier sign the line
                // was modulated with: +1 on positive-phase lines. The switch's
                // negation of those lines is a demodulator convention and must
                // not flip the burst's expected orientation.
                const double nominalSign =
                    getLinePhase(frameLine, phaseIDs.first, phaseIDs.second) ? 1.0 : -1.0;
                burstDev = chd::decoders::detectBurstDeviation(cvbsRow, vp, nominalSign);
            }
            demodChromaRow(cvbsRow, yNormRow, fieldWidth, hStart, hEnd,
                           frameLine, phaseIDs.first, phaseIDs.second,
                           chromaBandpass_, burstDev, bp, bq,
                           iWork, qWork, iFilt, qFilt,
                           out.y(frameLine), out.u(frameLine), out.v(frameLine));
        };

        if (mode_ == Mode::Frame) {
            // Weave both fields into a single interlaced frame input.
            for (int32_t sub = 0; sub < 2; ++sub) {
                const chd::decoders::SourceField &field = inputFields[frameIdx + sub];
                const int32_t yOffset = field.getOffset();
                const auto   *src = field.data.data();
                constexpr float scale = 1.0f / 65535.0f;
                for (int32_t y = 0; y < fieldHeight; ++y) {
                    const int32_t frameLine = y * 2 + yOffset;
                    if (frameLine >= frameHeight) break;
                    float *dst = frameBuffer.data() + frameLine * fieldWidth;
                    for (int32_t x = 0; x < fieldWidth; ++x) {
                        dst[x] = static_cast<float>(src[y * fieldWidth + x]) * scale;
                    }
                }
            }

            chd::nn::TensorSpec input;
            input.data  = frameBuffer.data();
            input.shape = { 1, 1, frameHeight, fieldWidth };
            auto outputs = session_->run({ input });
            requireOutputShape(*outputs, "ldzeug2_luma_sep", 1, frameHeight, fieldWidth);
            const float *yOut = outputs->data(0);

            // Per-line demod: cvbs row from the source field, Y row from
            // the NN output (which is in frame coordinates).
            for (int32_t sub = 0; sub < 2; ++sub) {
                const chd::decoders::SourceField &field = inputFields[frameIdx + sub];
                const int32_t yOffset = field.getOffset();
                const auto *src = field.data.data();
                for (int32_t y = 0; y < fieldHeight; ++y) {
                    const int32_t frameLine = y * 2 + yOffset;
                    if (frameLine >= storedHeight) break;
                    demodLine(frameLine,
                              src + y * fieldWidth,
                              yOut + frameLine * fieldWidth);
                }
            }
        } else {
            // Field mode: one inference per field; demod inline.
            for (int32_t sub = 0; sub < 2; ++sub) {
                const chd::decoders::SourceField &field = inputFields[frameIdx + sub];
                const int32_t yOffset = field.getOffset();
                const auto   *src = field.data.data();
                constexpr float scale = 1.0f / 65535.0f;
                for (int32_t y = 0; y < fieldHeight; ++y) {
                    for (int32_t x = 0; x < fieldWidth; ++x) {
                        fieldBuffer[y * fieldWidth + x] =
                            static_cast<float>(src[y * fieldWidth + x]) * scale;
                    }
                }

                chd::nn::TensorSpec input;
                input.data  = fieldBuffer.data();
                input.shape = { 1, 1, fieldHeight, fieldWidth };
                auto outputs = session_->run({ input });
                requireOutputShape(*outputs, "ldzeug2_luma_sep", 1, fieldHeight, fieldWidth);
                const float *yOut = outputs->data(0);
                for (int32_t y = 0; y < fieldHeight; ++y) {
                    const int32_t frameLine = y * 2 + yOffset;
                    if (frameLine >= storedHeight) break;
                    demodLine(frameLine,
                              src + y * fieldWidth,
                              yOut + y * fieldWidth);
                }
            }
        }
    }
}

}  // namespace chd::decoders::ldzeug
