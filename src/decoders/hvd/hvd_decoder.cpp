// SPDX-License-Identifier: GPL-3.0-or-later
#include "hvd_decoder.h"

#include <cmath>

#include "engine/colour.h"
#include "engine/engine.h"
#include "engine/lockin.h"
#include "engine/plane.h"
#include "engine/sequence.h"

#include "../../common/log.h"

namespace chd::decoders::hvd {

namespace {

double nominalFsc(::hvd::VideoStandard standard) {
    switch (standard) {
        case ::hvd::VideoStandard::kPal:  return ::hvd::kFscPal;
        case ::hvd::VideoStandard::kPalM: return ::hvd::kFscPalM;
        default:                          return ::hvd::kFscNtsc;
    }
}

// One full field in blanking-referenced IRE (blanking, not black: the NTSC
// setup pedestal is luma). Sync/blanking/burst stay in for the engine's own
// burst lock-in.
::hvd::FieldInput fieldToIre(const SourceField &field,
                             const chd::metadata::LdDecodeMetaData::VideoParameters &vp,
                             bool isFirstField) {
    const float blanking = static_cast<float>(vp.blanking16bIre);
    const float white = static_cast<float>(vp.white16bIre);
    ::hvd::FieldInput out;
    out.is_first_field = isFirstField;
    out.samples = ::hvd::Plane(vp.fieldHeight, vp.fieldWidth);
    for (int32_t line = 0; line < vp.fieldHeight; line++) {
        const uint16_t *in = field.data.data() + static_cast<size_t>(line) * vp.fieldWidth;
        float *dst = out.samples.data() + static_cast<size_t>(line) * vp.fieldWidth;
        for (int32_t x = 0; x < vp.fieldWidth; x++) {
            dst[x] = ::hvd::SampleToIre(static_cast<float>(in[x]), blanking, white);
        }
    }
    return out;
}

}  // namespace

HvdDecoder::HvdDecoder(const HvdConfiguration &hvdConfig)
    : config(hvdConfig), engine(std::make_unique<::hvd::HvdEngine>())
{
}

HvdDecoder::~HvdDecoder() = default;

bool HvdDecoder::configure(const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters)
{
    switch (videoParameters.system) {
        case chd::metadata::NTSC:  geometry.standard = ::hvd::VideoStandard::kNtsc; break;
        case chd::metadata::PAL:   geometry.standard = ::hvd::VideoStandard::kPal;  break;
        case chd::metadata::PAL_M: geometry.standard = ::hvd::VideoStandard::kPalM; break;
        default:
            chd::log::fail() << "The HVD decoder is for NTSC, PAL, and PAL-M video sources only";
            return false;
    }

    config.videoParameters = videoParameters;

    geometry.field_width = videoParameters.fieldWidth;
    geometry.field_height = videoParameters.fieldHeight;
    geometry.active_video_start = videoParameters.activeVideoStart;
    geometry.active_video_end = videoParameters.activeVideoEnd;
    geometry.colour_burst_start = videoParameters.colourBurstStart;
    geometry.colour_burst_end = videoParameters.colourBurstEnd;
    // Field-line crop covering the frame-line crop on both parities. The
    // engine decodes whole field lines and the write-back maps its rows
    // against the exact inclusive frame crop, so an odd-parity edge leaves a
    // row unused rather than shifting the picture.
    geometry.first_active_field_line = videoParameters.firstActiveFrameLine / 2;
    geometry.last_active_field_line = videoParameters.lastActiveFrameLine / 2 + 1;
    geometry.sample_rate = videoParameters.sampleRate;
    // The phase model is exact for the nominal subcarrier on the standard's
    // 4fsc grid (tabulated per-line advances, including 625-line PAL's
    // 270.576 deg). Setting subcarrier_hz replaces those tables with derived
    // values, so only a measurably deviating carrier opts in.
    geometry.subcarrier_hz = 0.0;
    if (videoParameters.fSC > 0.0
        && std::abs(videoParameters.fSC - nominalFsc(geometry.standard)) > 1.0) {
        geometry.subcarrier_hz = videoParameters.fSC;
    }

    engineConfig = ::hvd::HvdConfig{};
    engineConfig.chroma_phase_deg = static_cast<float>(config.chromaPhase);
    if (config.cgIterations >= 0) {
        engineConfig.cg_iterations = config.cgIterations;
    }
    if (config.temporal3d) {
        engineConfig.temporal_strength = static_cast<float>(config.temporalStrength);
    }
    // chromaGain is applied at write-back with the measured ACC gain, as
    // hvd-core's own callers do. is_pal is set by the engine from the geometry.

    // One decoder per pool worker, so the engine must not fan its FFTs across
    // cores on top of the frame-level parallelism. A no-op against the
    // single-threaded fftw3f this build links, but the contract is the
    // engine's, not the FFTW build's.
    engine->SetFftThreads(1);

    return true;
}

int32_t HvdDecoder::getLookBehind() const {
    return config.temporal3d ? engineConfig.chunk_overlap : 0;
}

int32_t HvdDecoder::getLookAhead() const {
    return config.temporal3d ? engineConfig.chunk_overlap : 0;
}

void HvdDecoder::decodeFrames(const std::vector<SourceField> &inputFields,
                              int32_t startIndex, int32_t endIndex,
                              std::vector<chd::output::ComponentFrame> &componentFrames)
{
    if (config.temporal3d) {
        decodeFrames3d(inputFields, startIndex, endIndex, componentFrames);
        return;
    }
    for (int32_t fieldIndex = startIndex, frameIndex = 0; fieldIndex < endIndex;
         fieldIndex += 2, frameIndex++) {
        decodeFrame(inputFields[fieldIndex], inputFields[fieldIndex + 1],
                    componentFrames[frameIndex]);
    }
}

void HvdDecoder::decodeFrame(const SourceField &firstField, const SourceField &secondField,
                             chd::output::ComponentFrame &componentFrame)
{
    const chd::metadata::LdDecodeMetaData::VideoParameters &vp = config.videoParameters;
    const float blanking = static_cast<float>(vp.blanking16bIre);
    const float white = static_cast<float>(vp.white16bIre);
    const float codesPerIre = ::hvd::CodesPerIre(blanking, white);

    // The first field carries the even frame rows.
    const ::hvd::FieldInput top = fieldToIre(firstField, vp, true);
    const ::hvd::FieldInput bottom = fieldToIre(secondField, vp, false);

    const ::hvd::FrameYc yc = engine->DecodeFrame(top, bottom, geometry, engineConfig);

    componentFrame.init(vp);

    // FrameYc covers the woven active picture: row r is frame line
    // 2 * first_active_field_line + r. Y goes back to composite codes; the
    // phasor chi = V - iU is IRE amplitude, so U/V scale by codes per IRE,
    // the composite scale the OutputWriter expects.
    const int32_t wovenFirstRow = 2 * geometry.first_active_field_line;
    const double gain = config.chromaGain * static_cast<double>(yc.acc_gain);
    for (int32_t y = vp.firstActiveFrameLine; y <= vp.lastActiveFrameLine; y++) {
        const int32_t r = y - wovenFirstRow;
        if (r < 0 || r >= yc.luma.height()) continue;
        double *outY = componentFrame.y(y);
        double *outU = componentFrame.u(y);
        double *outV = componentFrame.v(y);
        for (int32_t x = vp.activeVideoStart; x < vp.activeVideoEnd; x++) {
            const int32_t c = x - vp.activeVideoStart;
            outY[x] = static_cast<double>(::hvd::IreToSample(yc.luma.at(r, c), blanking, white));
            const ::hvd::Complex chi = yc.chroma_phasor.at(r, c);
            outV[x] = gain * static_cast<double>(chi.real()) * codesPerIre;
            outU[x] = gain * static_cast<double>(-chi.imag()) * codesPerIre;
        }
    }
}

void HvdDecoder::decodeFrames3d(const std::vector<SourceField> &inputFields,
                                int32_t startIndex, int32_t endIndex,
                                std::vector<chd::output::ComponentFrame> &componentFrames)
{
    const chd::metadata::LdDecodeMetaData::VideoParameters &vp = config.videoParameters;
    const float blanking = static_cast<float>(vp.blanking16bIre);
    const float white = static_cast<float>(vp.white16bIre);
    const float codesPerIre = ::hvd::CodesPerIre(blanking, white);

    // The whole window, look-behind/ahead context included, in one pass: the
    // sequence pipeline's cross-field terms need every field's active picture
    // and measured carrier. Fields alternate strictly from a first (top) field
    // at startIndex, so spatial parity follows index parity, black boundary
    // padding included.
    std::vector<::hvd::FieldObs> observations;
    observations.reserve(inputFields.size());
    std::vector<float> burstAmps(inputFields.size(), 0.0F);
    for (size_t i = 0; i < inputFields.size(); i++) {
        const int parity = static_cast<int>((i + static_cast<size_t>(startIndex)) & 1);
        const ::hvd::FieldInput field = fieldToIre(inputFields[i], vp, parity == 0);
        burstAmps[i] = ::hvd::BurstAmplitudeIre(field.samples, geometry);
        observations.push_back(
            ::hvd::PrepareFieldObs(field, geometry, engineConfig, parity));
    }

    const std::vector<::hvd::DecodedField> decoded =
        engine->DecodeSequenceWindow(observations, geometry, engineConfig);

    // ACC per output frame from its own two fields, as the frame path does,
    // not upstream's whole-window median: the window is small and its boundary
    // context can be dummy black fields with no burst, which would drag a
    // window-wide median toward the clamp.
    for (int32_t fieldIndex = startIndex, frameIndex = 0; fieldIndex < endIndex;
         fieldIndex += 2, frameIndex++) {
        chd::output::ComponentFrame &componentFrame = componentFrames[frameIndex];
        componentFrame.init(vp);

        float accGain = 1.0F;
        if (engineConfig.acc) {
            accGain = ::hvd::AccGain(
                0.5F * (burstAmps[fieldIndex] + burstAmps[fieldIndex + 1]),
                geometry.nominal_burst_ire());
        }
        const double gain = config.chromaGain * static_cast<double>(accGain);

        // DecodedField planes cover one field's active picture: frame line y
        // maps to row y / 2 - first_active_field_line in the decode of the
        // field with y's parity. Same conversions as the frame path.
        for (int32_t y = vp.firstActiveFrameLine; y <= vp.lastActiveFrameLine; y++) {
            const ::hvd::DecodedField &field = decoded[fieldIndex + (y % 2)];
            const int32_t r = y / 2 - geometry.first_active_field_line;
            if (r < 0 || r >= field.luma.height()) continue;
            double *outY = componentFrame.y(y);
            double *outU = componentFrame.u(y);
            double *outV = componentFrame.v(y);
            for (int32_t x = vp.activeVideoStart; x < vp.activeVideoEnd; x++) {
                const int32_t c = x - vp.activeVideoStart;
                outY[x] = static_cast<double>(
                    ::hvd::IreToSample(field.luma.at(r, c), blanking, white));
                const ::hvd::Complex chi = field.chroma.at(r, c);
                outV[x] = gain * static_cast<double>(chi.real()) * codesPerIre;
                outU[x] = gain * static_cast<double>(-chi.imag()) * codesPerIre;
            }
        }
    }
}

}  // namespace chd::decoders::hvd
