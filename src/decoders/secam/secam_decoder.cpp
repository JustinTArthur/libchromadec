// SPDX-License-Identifier: GPL-3.0-or-later

#include "secam_decoder.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "../../common/log.h"

namespace chd::decoders::secam {

namespace {

// BT.1700 Part C Table 4 (identical in BT.470-6 Table 2).
constexpr double kNominalFob = 4250000.0;   // Db undeviated subcarrier, 272*fH
constexpr double kNominalFor = 4406250.0;   // Dr undeviated subcarrier, 282*fH
constexpr double kDeviationDb = 230000.0;   // nominal deviation for D'B = 1
constexpr double kDeviationDr = 280000.0;   // nominal deviation for D'R = 1
constexpr double kBellCentre = 4286000.0;   // HF pre-correction f0
constexpr double kLfBreak = 85000.0;        // LF pre-correction f1
constexpr double kDbScale = 1.505;          // D'B = +1.505 (E'B - E'Y)
constexpr double kDrScale = -1.902;         // D'R = -1.902 (E'R - E'Y)

// Chroma band limits (Hz): flat over the FM block's nominal occupancy,
// raised-cosine transitions. Start point is the JVC BPF-1 takeoff curve
// (3.38-5.25 MHz, -3 dB at 3.9/4.75).
constexpr double kBandStopLo = 2900000.0;
constexpr double kBandPassLo = 3500000.0;
constexpr double kBandPassHi = 5100000.0;
constexpr double kBandStopHi = 5700000.0;

double bandGain(double f) {
    const double af = std::abs(f);
    if (af <= kBandStopLo || af >= kBandStopHi) return 0.0;
    if (af < kBandPassLo) {
        const double t = (af - kBandStopLo) / (kBandPassLo - kBandStopLo);
        return 0.5 - 0.5 * std::cos(M_PI * t);
    }
    if (af > kBandPassHi) {
        const double t = (kBandStopHi - af) / (kBandStopHi - kBandPassHi);
        return 0.5 - 0.5 * std::cos(M_PI * t);
    }
    return 1.0;
}

// The receiver's HF "bell" (cloche) network: the closed-form inverse of
// the encoder's HF pre-correction (Table 4 item 10d, the "anti-bell"),
// A_HFP(f) = (1 + j16F) / (1 + j1.26F), F = f/f0 - f0/f. First-order
// networks invert exactly; evaluated at true analog frequencies. `centre`
// is where the encoder's f0 sits in this capture: heterodyne converters
// (VHS colour-under) translate the whole FM block, anti-bell shaping
// included, so the bell must follow the measured carriers, not nominal.
std::complex<double> bellNetwork(double f, double centre) {
    if (f <= 0.0) return {0.0, 0.0};
    const double F = f / centre - centre / f;
    const std::complex<double> num(1.0, 1.26 * F);
    const std::complex<double> den(1.0, 16.0 * F);
    return num / den;
}

// Exact inverse of the LF pre-correction (Table 4 item 8):
// A_LFP(f) = (1 + jf/f1) / (1 + jf/3f1)  ->  inverse swaps the factors.
// Hermitian by construction when evaluated at signed frequencies.
std::complex<double> inverseLf(double f) {
    const std::complex<double> num(1.0, f / (3.0 * kLfBreak));
    const std::complex<double> den(1.0, f / kLfBreak);
    return num / den;
}

}  // namespace

SecamDecoder::SecamDecoder(const SecamConfiguration &secamConfig)
{
    config = secamConfig;
}

// Manual-mode lattice: extend the configured frame-0 anchor across the
// capture with the deterministic transmitted-line alternation. Field i
// (0-based) starts ceil(i/2) odd-length fields after field 0, so its
// anchor advances by (i + 1) / 2 (the BR.469 four-field cycle).
int32_t SecamDecoder::manualAnchorForField(int32_t seqNo) const
{
    const auto &vp = config.videoParameters;
    const int32_t firstActiveLine = (vp.firstActiveFrameLine + 1) / 2;
    const int32_t anchor0 = (firstActiveLine & 1) ^ config.manualFirstComponent;
    const int32_t i = std::max(seqNo - 1, 0);
    return (anchor0 + (i + 1) / 2) & 1;
}

SecamDecoder::~SecamDecoder()
{
    if (forwardPlan != nullptr) fftw_destroy_plan(forwardPlan);
    if (inversePlan != nullptr) fftw_destroy_plan(inversePlan);
    if (fftIn != nullptr) fftw_free(fftIn);
    if (fftFreq != nullptr) fftw_free(fftFreq);
    if (fftWork != nullptr) fftw_free(fftWork);
    if (fftOut != nullptr) fftw_free(fftOut);
}

bool SecamDecoder::configure(const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters)
{
    if (videoParameters.system != chd::metadata::SECAM) {
        chd::log::fail() << "This decoder is for SECAM video sources only";
        return false;
    }

    config.videoParameters = videoParameters;
    width = videoParameters.fieldWidth;
    height = videoParameters.fieldHeight;
    sampleRate = videoParameters.sampleRate;

    // Block geometry: power of two with at least 256 samples of context on
    // each side of a row (the longest filter memory here is the bell
    // network's ~16/(2*pi*f0) tail, well under a microsecond).
    blockSize = 1;
    while (blockSize < width + 512) blockSize <<= 1;
    margin = (blockSize - width) / 2;

    if (forwardPlan != nullptr) fftw_destroy_plan(forwardPlan);
    if (inversePlan != nullptr) fftw_destroy_plan(inversePlan);
    if (fftIn != nullptr) fftw_free(fftIn);
    if (fftFreq != nullptr) fftw_free(fftFreq);
    if (fftWork != nullptr) fftw_free(fftWork);
    if (fftOut != nullptr) fftw_free(fftOut);
    fftIn   = fftw_alloc_complex(blockSize);
    fftFreq = fftw_alloc_complex(blockSize);
    fftWork = fftw_alloc_complex(blockSize);
    fftOut  = fftw_alloc_complex(blockSize);
    forwardPlan = fftw_plan_dft_1d(blockSize, fftIn, fftFreq, FFTW_FORWARD, FFTW_ESTIMATE);
    inversePlan = fftw_plan_dft_1d(blockSize, fftWork, fftOut, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Frequency-domain masks. The chroma/band masks depend on where the
    // capture's FM block actually sits, so they are rebuilt whenever the
    // per-field carrier calibration moves the measured centre.
    carrierOffset = 0.0;
    buildChromaMasks();
    maskDeemphasis.assign(blockSize, {0.0, 0.0});
    for (int32_t k = 0; k < blockSize; k++) {
        const double f = (k <= blockSize / 2 ? k : k - blockSize) * sampleRate / blockSize;
        maskDeemphasis[k] = inverseLf(f);
    }

    // Differentiating FIR, applied after the analytic signal is mixed down
    // to near-DC: any differentiator's fractional gain error is multiplied
    // by the absolute frequency it operates at, so discriminating at the
    // carrier would leak carrier-scale error into the small deviations.
    // At baseband a maximally-flat central difference (error O(w^9)) is
    // exact to sub-Hz across the +-1 MHz the mixed-down FM block spans.
    constexpr int32_t K = 4;
    constexpr double kMaxFlat[K] = {4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0};
    diffTaps.assign(2 * K + 1, 0.0);
    for (int32_t k = 1; k <= K; k++) {
        diffTaps[K - k] = kMaxFlat[k - 1];
        diffTaps[K + k] = -kMaxFlat[k - 1];
    }

    // Porch measurement window: the tail of the back porch, directly before
    // active video. The reference carrier runs across the whole porch, but
    // the early porch still carries the sweep from the previous line's
    // carrier switch (measured on tape: early-window medians of both
    // components pull toward each other), so measure where the carrier has
    // had the longest to settle.
    if (videoParameters.activeVideoStart > 70
        && videoParameters.activeVideoStart < width) {
        porchEnd = videoParameters.activeVideoStart - 5;
        porchStart = porchEnd - 60;
    } else {
        porchStart = static_cast<int32_t>(width * 0.085);
        porchEnd = static_cast<int32_t>(width * 0.155);
    }

    const size_t fieldSamples = static_cast<size_t>(width) * height;
    analytic.assign(fieldSamples, {0.0, 0.0});
    analyticRef.assign(fieldSamples, {0.0, 0.0});
    chromaRecon.assign(fieldSamples, 0.0);
    fInst.assign(fieldSamples, 0.0);
    fInstRef.assign(fieldSamples, 0.0);
    demod.assign(fieldSamples, 0.0);
    deemph.assign(fieldSamples, 0.0);

    lastGoodFob = 0.0;
    lastGoodFor = 0.0;
    haveLastGoodCalibration = false;

    configurationSet = true;
    return true;
}

// (Re)build the carrier-relative frequency masks with the FM block centred
// at nominal + carrierOffset. Heterodyne converters translate the whole
// block (anti-bell shaping included), so the band, the bell network, and
// the discriminator's mix-down all follow the measured carrier pair.
void SecamDecoder::buildChromaMasks()
{
    maskChroma.assign(blockSize, {0.0, 0.0});
    maskBand.assign(blockSize, 0.0);
    for (int32_t k = 0; k < blockSize; k++) {
        const double f = (k <= blockSize / 2 ? k : k - blockSize) * sampleRate / blockSize;
        // Analytic-signal factor: keep positive frequencies doubled, drop
        // negative ones (exact Hilbert via the block FFT).
        double analyticFactor = 0.0;
        if (k == 0 || k == blockSize / 2) analyticFactor = 1.0;
        else if (k < blockSize / 2)       analyticFactor = 2.0;
        const double fShifted = std::abs(f) - carrierOffset;
        maskChroma[k] = analyticFactor * bandGain(fShifted)
                        * bellNetwork(std::abs(f), kBellCentre + carrierOffset);
        // Bell-free analytic band: its real part is the plain band-filtered
        // signal (the analytic factor restores the Hermitian half exactly),
        // and the complex signal is the calibration reference. Keeping the
        // bell out of this path matters: the bell network shapes noise and
        // switch transients asymmetrically around its centre, which biases
        // the discriminated porch medians and ties them to the mask state.
        maskBand[k] = analyticFactor * bandGain(fShifted);
    }
    mixFrequency = 0.5 * (kNominalFob + kNominalFor) + carrierOffset;
}

void SecamDecoder::filterRowAnalytic(const uint16_t *fieldData, int32_t numSamples,
                                     int32_t row, std::complex<double> *analyticRow,
                                     std::complex<double> *analyticRefRow,
                                     double *chromaRow)
{
    const int32_t base = row * width;
    for (int32_t i = 0; i < blockSize; i++) {
        int32_t s = base - margin + i;
        s = std::clamp(s, 0, numSamples - 1);
        fftIn[i][0] = static_cast<double>(fieldData[s]);
        fftIn[i][1] = 0.0;
    }
    fftw_execute(forwardPlan);

    const double scale = 1.0 / blockSize;
    for (int32_t k = 0; k < blockSize; k++) {
        const std::complex<double> s(fftFreq[k][0], fftFreq[k][1]);
        const std::complex<double> v = s * maskChroma[k];
        fftWork[k][0] = v.real();
        fftWork[k][1] = v.imag();
    }
    fftw_execute(inversePlan);
    // Mix the analytic chroma down to near-DC (phase-continuous across the
    // field: rows are contiguous in time, so the global sample index is the
    // time base).
    const double wMix = 2.0 * M_PI * mixFrequency / sampleRate;
    const std::complex<double> phasor0 =
        std::polar(1.0, -std::fmod(wMix * base, 2.0 * M_PI));
    const std::complex<double> step = std::polar(1.0, -wMix);
    std::complex<double> phasor = phasor0;
    for (int32_t x = 0; x < width; x++) {
        analyticRow[x] = std::complex<double>(fftOut[margin + x][0], fftOut[margin + x][1])
                         * scale * phasor;
        phasor *= step;
    }

    // Bell-free band pass: the real part is the reconstituted chroma the
    // luma path subtracts, the mixed-down complex signal is the reference
    // the carrier calibration discriminates.
    for (int32_t k = 0; k < blockSize; k++) {
        fftWork[k][0] = fftFreq[k][0] * maskBand[k];
        fftWork[k][1] = fftFreq[k][1] * maskBand[k];
    }
    fftw_execute(inversePlan);
    phasor = phasor0;
    for (int32_t x = 0; x < width; x++) {
        const std::complex<double> v(fftOut[margin + x][0], fftOut[margin + x][1]);
        chromaRow[x] = v.real() * scale;
        analyticRefRow[x] = v * scale * phasor;
        phasor *= step;
    }
}

void SecamDecoder::filterRowDeemphasis(const double *demodData, int32_t numSamples,
                                       int32_t row, double *deemphRow)
{
    const int32_t base = row * width;
    for (int32_t i = 0; i < blockSize; i++) {
        int32_t s = base - margin + i;
        s = std::clamp(s, 0, numSamples - 1);
        fftIn[i][0] = demodData[s];
        fftIn[i][1] = 0.0;
    }
    fftw_execute(forwardPlan);

    const double scale = 1.0 / blockSize;
    for (int32_t k = 0; k < blockSize; k++) {
        const std::complex<double> s(fftFreq[k][0], fftFreq[k][1]);
        const std::complex<double> v = s * maskDeemphasis[k];
        fftWork[k][0] = v.real();
        fftWork[k][1] = v.imag();
    }
    fftw_execute(inversePlan);
    for (int32_t x = 0; x < width; x++) {
        deemphRow[x] = fftOut[margin + x][0] * scale;
    }
}

void SecamDecoder::discriminateRow(const std::complex<double> *analyticData,
                                   int32_t numSamples, int32_t row, double *fInstRow)
{
    const int32_t K = (static_cast<int32_t>(diffTaps.size()) - 1) / 2;
    const int32_t base = row * width;
    const double freqScale = sampleRate / (2.0 * M_PI);

    for (int32_t x = 0; x < width; x++) {
        const std::complex<double> a = analyticData[base + x];
        const double mag2 = std::norm(a);
        if (mag2 < 1e-9) {
            fInstRow[x] = 0.0;
            continue;
        }
        std::complex<double> d(0.0, 0.0);
        for (int32_t m = -K; m <= K; m++) {
            if (m == 0) continue;
            const int32_t s = std::clamp(base + x - m, 0, numSamples - 1);
            d += diffTaps[m + K] * analyticData[s];
        }
        // Im(conj(a) * a') / |a|^2 is the exact phase derivative: the
        // envelope's own derivative lands in the real part, so a separate
        // limiter stage is unnecessary and per-line AGC scaling cancels.
        fInstRow[x] = mixFrequency
                      + freqScale * (a.real() * d.imag() - a.imag() * d.real()) / mag2;
    }
}

void SecamDecoder::decodeField(const SourceField &inputField,
                               chd::output::ComponentFrame &componentFrame,
                               FieldIdent &identOut)
{
    const auto &vp = config.videoParameters;
    const uint16_t *data = inputField.data.data();
    const int32_t rows = std::min<int32_t>(
        height, static_cast<int32_t>(inputField.data.size()) / width);
    const int32_t numSamples = rows * width;
    if (rows <= 0) return;

    // Central-window per-row stats over the picture region: the reference
    // envelope scale for validity gating, and the frequency inputs the
    // bottle/content mechanisms read. Reads the bell-free reference buffers,
    // whatever they hold at call time.
    const auto &vpEarly = config.videoParameters;
    const int32_t centreStart = vpEarly.activeVideoStart + 30;
    const int32_t centreEnd = std::max(centreStart + 64, vpEarly.activeVideoEnd - 30);
    const auto centreStats = [&](int32_t row, double &freqOut, double &envOut) {
        double freqSum = 0.0, envSum = 0.0;
        int32_t n = 0;
        for (int32_t x = centreStart; x < centreEnd && x < width; x++) {
            const size_t i = static_cast<size_t>(row) * width + x;
            freqSum += fInstRef[i];
            envSum += std::abs(analyticRef[i]);
            n++;
        }
        freqOut = (n > 0) ? freqSum / n : 0.0;
        envOut = (n > 0) ? envSum / n : 0.0;
    };

    std::vector<double> porchFreq(rows, 0.0);
    std::vector<double> porchEnvelope(rows, 0.0);
    std::vector<bool> rowValid(rows, false);
    std::vector<double> validFreqs;
    double activeEnvMedian = 0.0;
    double fob = kNominalFob;
    double for_ = kNominalFor;
    bool calibrated = false;

    // Filter, discriminate, and calibrate; when the measured carrier pair
    // shows the FM block sits away from where the masks were built (VHS
    // colour-under converters translate the whole block, anti-bell included),
    // recentre the masks on the measurement and redo the field once. The
    // porch measurement runs on the bell-free band path, where a steady
    // in-band carrier is untouched by mask recentring (only the distant band
    // edges move), so one recentre settles it; later fields reuse the masks
    // until the offset drifts past the threshold again.
    for (int32_t pass = 0; pass < 2; pass++) {
        for (int32_t row = 0; row < rows; row++) {
            filterRowAnalytic(data, numSamples, row,
                              analytic.data() + static_cast<size_t>(row) * width,
                              analyticRef.data() + static_cast<size_t>(row) * width,
                              chromaRecon.data() + static_cast<size_t>(row) * width);
        }
        for (int32_t row = 0; row < rows; row++) {
            discriminateRow(analytic.data(), numSamples, row,
                            fInst.data() + static_cast<size_t>(row) * width);
            discriminateRow(analyticRef.data(), numSamples, row,
                            fInstRef.data() + static_cast<size_t>(row) * width);
        }

        // Back-porch reference measurement: per row, the median discriminated
        // frequency and mean analytic envelope over the porch window, both
        // from the bell-free reference path. The median rides out FM clicks
        // and window-edge transients on noisy tape; the flat band response
        // keeps the noise around each carrier symmetric, so the median sits
        // on the carrier instead of being dragged toward the bell centre.
        std::vector<double> windowFreqs;
        for (int32_t row = 0; row < rows; row++) {
            windowFreqs.clear();
            double envSum = 0.0;
            int32_t n = 0;
            for (int32_t x = porchStart; x < porchEnd; x++) {
                const size_t i = static_cast<size_t>(row) * width + x;
                windowFreqs.push_back(fInstRef[i]);
                envSum += std::abs(analyticRef[i]);
                n++;
            }
            if (n > 0) {
                std::sort(windowFreqs.begin(), windowFreqs.end());
                porchFreq[row] = windowFreqs[windowFreqs.size() / 2];
                porchEnvelope[row] = envSum / n;
            }
        }

        std::vector<double> activeEnv;
        {
            const int32_t fa = inputField.getFirstActiveLine(vpEarly);
            const int32_t la = std::min(inputField.getLastActiveLine(vpEarly), rows);
            for (int32_t row = fa; row < la; row++) {
                double f, e;
                centreStats(row, f, e);
                activeEnv.push_back(e);
            }
        }
        activeEnvMedian = 0.0;
        if (!activeEnv.empty()) {
            std::vector<double> sortedEnv = activeEnv;
            std::sort(sortedEnv.begin(), sortedEnv.end());
            activeEnvMedian = sortedEnv[sortedEnv.size() / 2];
        }

        // Rows with a usable porch carrier: the reference carrier has the
        // same order of amplitude as the picture FM block, so gate against
        // the picture envelope. A blanked porch leaves only filter leakage,
        // which this rejects.
        const double envThreshold = 0.25 * activeEnvMedian;
        rowValid.assign(rows, false);
        validFreqs.clear();
        for (int32_t row = 0; row < rows; row++) {
            if (envThreshold > 0.0 && porchEnvelope[row] > envThreshold) {
                rowValid[row] = true;
                validFreqs.push_back(porchFreq[row]);
            }
        }

        // Carrier calibration: split the per-line porch frequencies into the
        // two undeviated carriers. The measured pair absorbs converter
        // offsets (ME-SECAM LO arithmetic), so absolute positions are never
        // assumed. A field whose pair fails the separation gate reuses the
        // last good field's pair: the block shift is a property of the
        // capture, not the field, so a stale measurement stays close while
        // the nominal carriers can sit a full deviation away.
        fob = haveLastGoodCalibration ? lastGoodFob : kNominalFob;
        for_ = haveLastGoodCalibration ? lastGoodFor : kNominalFor;
        calibrated = false;
        if (validFreqs.size() >= 8) {
            std::sort(validFreqs.begin(), validFreqs.end());
            const size_t half = validFreqs.size() / 2;
            // Per-cluster medians: FM clicks put long tails on the porch
            // frequency distributions, and a tail-dragged carrier estimate
            // becomes a constant colour-difference offset downstream.
            const double lowMedian = validFreqs[half / 2];
            const double highMedian = validFreqs[half + (validFreqs.size() - half) / 2];
            const double separation = highMedian - lowMedian;
            if (separation > 90000.0 && separation < 230000.0) {
                fob = lowMedian;
                for_ = highMedian;
                calibrated = true;
            }
        }

        if (pass == 0 && calibrated) {
            const double measuredOffset =
                0.5 * (fob + for_) - 0.5 * (kNominalFob + kNominalFor);
            if (std::abs(measuredOffset - carrierOffset) > 10000.0) {
                carrierOffset = measuredOffset;
                buildChromaMasks();
                continue;
            }
        }
        break;
    }

    const double split = 0.5 * (fob + for_);
    const int32_t firstActive = inputField.getFirstActiveLine(vp);
    const int32_t lastActive = std::min(inputField.getLastActiveLine(vp), rows);
    identOut.fob = fob;
    identOut.for_ = for_;
    if (calibrated) {
        lastGoodFob = fob;
        lastGoodFor = for_;
        haveLastGoodCalibration = true;
    } else if (haveLastGoodCalibration) {
        chd::log::info() << "SecamDecoder: porch carrier pair not measurable;"
                         << "reusing the last measured subcarriers";
    } else {
        chd::log::warn() << "SecamDecoder: porch carrier pair not measurable;"
                         << "using nominal subcarriers";
    }

    // Strict-alternation majority fit over labelled rows: single-line
    // measurement errors self-heal, and the fit degrades gracefully to a
    // low confidence instead of a scrambled lattice.
    const auto alternationFit = [](const std::vector<int32_t> &rowsIn,
                                   const std::vector<int32_t> &labels) {
        AnchorFit fit;
        int32_t agree[2] = {0, 0};
        for (size_t i = 0; i < rowsIn.size(); i++) {
            agree[(labels[i] ^ rowsIn[i]) & 1]++;
        }
        const int32_t total = static_cast<int32_t>(rowsIn.size());
        if (total == 0) return fit;
        fit.valid = true;
        fit.anchor = (agree[1] > agree[0]) ? 1 : 0;
        fit.confidence = static_cast<double>(agree[fit.anchor]) / total;
        fit.measuredRows = total;
        return fit;
    };

    // Porch ("line identification"): label each measurable row by its
    // undeviated carrier, higher = Dr.
    AnchorFit porchFit;
    {
        std::vector<int32_t> fitRows, labels;
        for (int32_t row = firstActive; row < lastActive; row++) {
            if (!rowValid[row]) continue;
            fitRows.push_back(row);
            labels.push_back(porchFreq[row] > split ? 1 : 0);
        }
        porchFit = alternationFit(fitRows, labels);
    }

    // Field-ident bottles (BT.470-6 option B): trapezoids on the early
    // vertical-interval lines, Dr lines deviated far positive and Db lines
    // far negative, so the carrier split separates them trivially.
    const auto fitBottles = [&]() {
        AnchorFit fit;
        std::vector<int32_t> fitRows, labels;
        const int32_t last = std::min<int32_t>(15, rows - 1);
        for (int32_t row = 5; row <= last; row++) {
            double f, e;
            centreStats(row, f, e);
            if (e < 0.5 * activeEnvMedian || activeEnvMedian <= 0.0) continue;
            fitRows.push_back(row);
            labels.push_back(f > split ? 1 : 0);
        }
        if (static_cast<int32_t>(fitRows.size()) < 4) return fit;
        fit = alternationFit(fitRows, labels);
        if (fit.confidence < 0.75) fit.valid = false;
        return fit;
    };

    // Content statistics: a line's mean discriminated frequency sits near
    // its own carrier (real content averages toward low saturation), so a
    // median split over the active rows recovers the alternation without
    // trusting absolute carrier positions.
    const auto fitContent = [&]() {
        AnchorFit fit;
        std::vector<int32_t> fitRows;
        std::vector<double> freqs;
        for (int32_t row = firstActive; row < lastActive; row++) {
            double f, e;
            centreStats(row, f, e);
            if (e < 0.3 * activeEnvMedian || activeEnvMedian <= 0.0) continue;
            fitRows.push_back(row);
            freqs.push_back(f);
        }
        if (static_cast<int32_t>(fitRows.size()) < 16) return fit;
        std::vector<double> sorted = freqs;
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted[sorted.size() / 2];
        std::vector<int32_t> labels;
        for (const double f : freqs) labels.push_back(f > median ? 1 : 0);
        return alternationFit(fitRows, labels);
    };

    switch (config.identMode) {
        case IdentMode::Manual:
            identOut.anchor = manualAnchorForField(inputField.field.seqNo);
            identOut.confidence = 1.0;
            identOut.mechanism = 3;
            break;
        case IdentMode::Porch:
            identOut.anchor = porchFit.anchor;
            identOut.confidence = porchFit.valid ? porchFit.confidence : 0.0;
            identOut.mechanism = 0;
            break;
        case IdentMode::Bottles: {
            const AnchorFit fit = fitBottles();
            if (!fit.valid) {
                chd::log::warn() << "SecamDecoder: no field-ident bottles measurable;"
                                 << "line identity defaults to a Db-first lattice";
            }
            identOut.anchor = fit.valid ? fit.anchor : 0;
            identOut.confidence = fit.valid ? fit.confidence : 0.0;
            identOut.mechanism = 1;
            break;
        }
        case IdentMode::Auto:
        default: {
            const AnchorFit bottles = fitBottles();
            if (porchFit.valid && porchFit.measuredRows >= 8 && porchFit.confidence >= 0.6) {
                identOut.anchor = porchFit.anchor;
                identOut.confidence = porchFit.confidence;
                identOut.mechanism = 0;
                if (bottles.valid && bottles.anchor != porchFit.anchor) {
                    chd::log::warn() << "SecamDecoder: field-ident bottles disagree with"
                                     << "the porch line identification";
                }
            } else if (bottles.valid) {
                identOut.anchor = bottles.anchor;
                identOut.confidence = bottles.confidence;
                identOut.mechanism = 1;
            } else {
                const AnchorFit content = fitContent();
                identOut.anchor = content.valid ? content.anchor : 0;
                identOut.confidence = content.valid ? content.confidence : 0.0;
                identOut.mechanism = 2;
            }
            break;
        }
    }

    // Demodulate against the per-component carrier zero and deviation, then
    // invert the LF pre-correction.
    for (int32_t row = 0; row < rows; row++) {
        const int32_t component = (row + identOut.anchor) & 1;
        const double zero = (component == 0) ? fob : for_;
        const double deviation = (component == 0) ? kDeviationDb : kDeviationDr;
        const size_t base = static_cast<size_t>(row) * width;
        for (int32_t x = 0; x < width; x++) {
            demod[base + x] = (fInst[base + x] - zero) / deviation;
        }
    }
    // FM click concealment ("SECAM fire" suppression): detect discriminator
    // samples where the analytic envelope collapses or the deviation exits
    // the Table 4 maxima, then conceal before de-emphasis by interpolating
    // across the span, or substituting the previous same-component line for
    // spans too wide to interpolate. Level 0 bypasses the stage entirely.
    if (config.clickNrLevel > 0.0) {
        // Chroma noise floor from the same porch windows used for ident and
        // calibration: median absolute deviation of the per-line porch
        // frequencies about their component's carrier.
        std::vector<double> porchDeviations;
        for (int32_t row = firstActive; row < lastActive; row++) {
            if (!rowValid[row]) continue;
            const double centre = porchFreq[row] > split ? for_ : fob;
            porchDeviations.push_back(std::abs(porchFreq[row] - centre));
        }
        double noiseMadHz = 0.0;
        if (!porchDeviations.empty()) {
            std::sort(porchDeviations.begin(), porchDeviations.end());
            noiseMadHz = porchDeviations[porchDeviations.size() / 2];
        }

        // Frozen threshold formula (documented in the API reference):
        // envelope dip 12 - 6*level dB below the row's median envelope,
        // deviation overshoot max(2.6 - 1.6*level, 1.15) + 6*noise/506kHz
        // in max-deviation multiples. Endpoints calibrated with the
        // swept-level study on the synthetic generator: the band filter's
        // own leakage keeps even a dead carrier's envelope near a quarter
        // of nominal inside a dropout, so a deeper dip threshold misses
        // real holes. The overshoot floor keeps a limiter flat-top riding
        // exactly at the Table 4 bounds from flagging on its own ripple;
        // the deviation rail owns everything between the bound and the
        // floor.
        const double dipDb = (config.clickEnvDipDbOverride > 0.0)
                                 ? config.clickEnvDipDbOverride
                                 : 12.0 - 6.0 * config.clickNrLevel;
        const double overshoot =
            (config.clickFreqOvershootOverride > 0.0)
                ? config.clickFreqOvershootOverride
                : std::max(2.6 - 1.6 * config.clickNrLevel, 1.15)
                      + 6.0 * noiseMadHz / 506000.0;
        componentFrame.chromaClick.valid = true;
        componentFrame.chromaClick.envDipDb = dipDb;
        componentFrame.chromaClick.freqOvershoot = overshoot;
        const double envRatio = std::pow(10.0, -dipDb / 20.0);

        const int32_t fieldOffset = inputField.getOffset();
        const int32_t detectStart = std::max<int32_t>(0, vp.activeVideoStart - 8);
        const int32_t detectEnd = std::min<int32_t>(width, vp.activeVideoEnd + 8);
        std::vector<char> flagged(width);
        std::vector<double> rowEnv;
        for (int32_t row = firstActive; row < lastActive; row++) {
            const size_t base = static_cast<size_t>(row) * width;
            const int32_t component = (row + identOut.anchor) & 1;
            const double dev = (component == 0) ? kDeviationDb : kDeviationDr;
            const double loBound = ((component == 0) ? -350000.0 : -506000.0) / dev;
            const double hiBound = ((component == 0) ? 506000.0 : 350000.0) / dev;

            rowEnv.clear();
            for (int32_t x = detectStart; x < detectEnd; x++) {
                rowEnv.push_back(std::abs(analytic[base + x]));
            }
            if (rowEnv.empty()) continue;
            std::sort(rowEnv.begin(), rowEnv.end());
            const double envRef = rowEnv[rowEnv.size() / 2];
            if (envRef <= 0.0) continue;
            const double envFloor = envRef * envRatio;

            std::fill(flagged.begin(), flagged.end(), 0);
            bool any = false;
            for (int32_t x = detectStart; x < detectEnd; x++) {
                const double dValue = demod[base + x];
                if (dValue < loBound * overshoot || dValue > hiBound * overshoot
                    || std::abs(analytic[base + x]) < envFloor) {
                    flagged[x] = 1;
                    any = true;
                }
            }
            if (!any) continue;

            int32_t x = detectStart;
            while (x < detectEnd) {
                if (!flagged[x]) {
                    x++;
                    continue;
                }
                int32_t runEnd = x;
                while (runEnd < detectEnd && flagged[runEnd]) runEnd++;
                const int32_t spanStart = std::max<int32_t>(0, x - 3);
                const int32_t spanEnd = std::min<int32_t>(width, runEnd + 3);

                if (spanEnd - spanStart <= 48 && spanStart > 0 && spanEnd < width) {
                    const double before = demod[base + spanStart - 1];
                    const double after = demod[base + spanEnd];
                    const int32_t n = spanEnd - spanStart + 1;
                    for (int32_t i = spanStart; i < spanEnd; i++) {
                        demod[base + i] =
                            before + (after - before) * (i - spanStart + 1) / n;
                    }
                } else if (row >= 2) {
                    // Rows are processed top to bottom, so the substituted
                    // line has already been concealed itself if needed.
                    const size_t prev = base - 2 * static_cast<size_t>(width);
                    for (int32_t i = spanStart; i < spanEnd; i++) {
                        demod[base + i] = demod[prev + i];
                    }
                } else {
                    for (int32_t i = spanStart; i < spanEnd; i++) {
                        demod[base + i] = 0.0;
                    }
                }
                componentFrame.chromaConcealedSpans.push_back(
                    {row * 2 + fieldOffset, spanStart, spanEnd});
                x = spanEnd;
            }
        }
    }

    // Deviation rail: the transmitter clips the pre-corrected signal to the
    // Table 4 maximum deviations, so demodulated excursions beyond them are
    // never signal. Clamping before de-emphasis bounds whatever clicks the
    // concealment stage left under its thresholds (or all of them when it
    // is bypassed).
    for (int32_t row = 0; row < rows; row++) {
        const int32_t component = (row + identOut.anchor) & 1;
        const double dev = (component == 0) ? kDeviationDb : kDeviationDr;
        const double loBound = ((component == 0) ? -350000.0 : -506000.0) / dev;
        const double hiBound = ((component == 0) ? 506000.0 : 350000.0) / dev;
        const size_t base = static_cast<size_t>(row) * width;
        for (int32_t x = 0; x < width; x++) {
            demod[base + x] = std::clamp(demod[base + x], loBound, hiBound);
        }
    }

    for (int32_t row = 0; row < rows; row++) {
        filterRowDeemphasis(demod.data(), numSamples, row,
                            deemph.data() + static_cast<size_t>(row) * width);
    }

    // Write the decoded lines: Y from the input minus the reconstituted
    // chroma band, and one colour-difference component per line, scaled to
    // the composite-domain U/V the OutputWriter expects. The reduction factors
    // are the ones the writer will divide back out, so they cancel: SECAM's
    // own Db/Dr scalings (kDbScale/kDrScale) are what the signal actually
    // carries, and the broadcast_scaling_precision option leaves the picture
    // untouched here by construction.
    const int32_t offset = inputField.getOffset();
    const double uvRange = vp.white16bIre - vp.black16bIre;
    const double uScale = config.colorConversion.uReduction * uvRange * config.chromaGain / kDbScale;
    const double vScale = config.colorConversion.vReduction * uvRange * config.chromaGain / kDrScale;
    const int32_t frameHeight = componentFrame.getHeight();

    for (int32_t fieldLine = firstActive; fieldLine < lastActive; fieldLine++) {
        const int32_t frameRow = fieldLine * 2 + offset;
        if (frameRow < 0 || frameRow >= frameHeight) continue;
        const size_t base = static_cast<size_t>(fieldLine) * width;
        const int32_t component = (fieldLine + identOut.anchor) & 1;

        double *outY = componentFrame.y(frameRow);
        for (int32_t x = 0; x < width; x++) {
            outY[x] = static_cast<double>(data[base + x]) - chromaRecon[base + x];
        }
        if (component == 0) {
            double *outU = componentFrame.u(frameRow);
            for (int32_t x = 0; x < width; x++) {
                outU[x] = deemph[base + x] * uScale;
            }
        } else {
            double *outV = componentFrame.v(frameRow);
            for (int32_t x = 0; x < width; x++) {
                outV[x] = deemph[base + x] * vScale;
            }
        }
        componentFrame.chromaRowComponents[frameRow] = static_cast<int8_t>(component);
    }
}

void SecamDecoder::decodeFrames(const std::vector<SourceField> &inputFields,
                                int32_t startIndex, int32_t endIndex,
                                std::vector<chd::output::ComponentFrame> &componentFrames)
{
    assert(configurationSet);
    assert((componentFrames.size() * 2) == static_cast<size_t>(endIndex - startIndex));

    for (int32_t i = startIndex, k = 0; i < endIndex; i += 2, k++) {
        auto &frame = componentFrames[k];
        frame.init(config.videoParameters);
        frame.chromaRowComponents.assign(frame.getHeight(), -1);

        FieldIdent firstIdent, secondIdent;
        decodeField(inputFields[i], frame, firstIdent);
        decodeField(inputFields[i + 1], frame, secondIdent);

        frame.chromaIdent.valid = true;
        // Report the weaker field's deciding mechanism: it is the one an
        // archival consumer needs to know about.
        frame.chromaIdent.mechanism =
            (secondIdent.confidence < firstIdent.confidence) ? secondIdent.mechanism
                                                             : firstIdent.mechanism;
        frame.chromaIdent.fieldConfidence[0] = firstIdent.confidence;
        frame.chromaIdent.fieldConfidence[1] = secondIdent.confidence;
        frame.chromaIdent.confidence =
            0.5 * (firstIdent.confidence + secondIdent.confidence);
    }
}

}  // namespace chd::decoders::secam
