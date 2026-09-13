// SPDX-License-Identifier: GPL-3.0-or-later

#include "cvbs_composite_source.h"

#include <algorithm>
#include <stdexcept>

#include "../common/log.h"

namespace chd::reader {

CvbsCompositeSource::CvbsCompositeSource() = default;
CvbsCompositeSource::~CvbsCompositeSource()
{
    if (isOpen) inputFile.close();
}

bool CvbsCompositeSource::open(const std::string &path,
                               const chd::format::VideoStandardPreset &videoStandard,
                               chd::format::SampleEncoding sampleEncoding,
                               chd::format::SignalState signalState,
                               std::optional<int32_t> blackLevelOverride,
                               chd::format::FrameLayout layoutOverride,
                               std::optional<int64_t> declaredFrames,
                               std::optional<bool> subcarrierLockedOverride,
                               Plane whichPlane,
                               std::optional<chd::format::HorizontalAlignment> frameNativeAlignment)
{
    if (isOpen) {
        chd::log::fail() << "source is already open";
        return false;
    }

    standardEnum = videoStandard.standard;
    preset       = &videoStandard;
    encoding     = sampleEncoding;
    state        = signalState;
    plane        = whichPlane;
    bytesPerSample = chd::format::getSampleEncoding(encoding).bytesPerSample;

    inputFile.open(path, std::ios::binary);
    if (!inputFile.is_open()) {
        chd::log::fail() << "could not open" << path;
        return false;
    }
    inputFile.seekg(0, std::ios::end);
    fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    layout = chd::format::resolveFrameLayout(layoutOverride, videoStandard, signalState,
                                             bytesPerSample, fileSize, declaredFrames);

    videoParameters = chd::format::makeCvbsVideoParameters(
        videoStandard, sampleEncoding, blackLevelOverride, layout,
        subcarrierLockedOverride);
    rowAlignment = subcarrierLockedOverride.value_or(false)
                       ? chd::format::HorizontalAlignment::BLANKING_START
                       : chd::format::HorizontalAlignment::SYNC_START;
    fieldWidth   = videoParameters.fieldWidth;
    fieldHeight  = videoParameters.fieldHeight;
    fieldSamples = fieldWidth * fieldHeight;
    fieldByteSize = fieldSamples * bytesPerSample;

    if (fileSize <= 0 || fieldByteSize <= 0) {
        chd::log::fail() << "file is empty, or the standard implies a zero-length field";
        inputFile.close();
        return false;
    }
    if (layout == chd::format::FrameLayout::FRAME_NATIVE) {
        const int64_t frameByteSize =
            static_cast<int64_t>(videoStandard.samplesPerFrame) * bytesPerSample;
        numFields = static_cast<int32_t>(fileSize / frameByteSize) * 2;
    } else {
        numFields = static_cast<int32_t>(fileSize / fieldByteSize);
    }
    if (layout == chd::format::FrameLayout::FRAME_NATIVE) {
        // Resolve the horizontal alignment from the signal: measure 0H from
        // the sync edges of 50 post-VBI first-field rows, then rebuild the
        // burst/active windows for the cut the capture actually uses. A
        // chroma plane carries no sync, so it takes the luma plane's result.
        chd::format::HorizontalAlignment alignment;
        if (plane == Plane::CHROMA) {
            alignment = frameNativeAlignment.value_or(
                chd::format::HorizontalAlignment::SYNC_START);
        } else {
            std::optional<double> measured;
            if (numFields >= 2 &&
                chd::format::getSampleEncoding(encoding).hasStandardAmplitudeMapping) {
                const auto plan = chd::format::planFrameNativeFieldRead(
                    videoStandard, 0, 40, 89, bytesPerSample);
                const Data rows = readAndConvert(plan.startByte, plan.numBytes);
                measured = chd::format::measureRowZeroH(rows.data(), 50, fieldWidth,
                                                        videoStandard.levels);
            }
            alignment = chd::format::resolveFrameNativeAlignment(
                videoStandard, measured, "CvbsCompositeSource");
        }
        videoParameters = chd::format::makeCvbsVideoParameters(
            videoStandard, sampleEncoding, blackLevelOverride, layout,
            subcarrierLockedOverride, alignment);
        rowAlignment = alignment;
    }

    isOpen = true;
    chd::log::debug().nospace()
        << "CvbsCompositeSource::open(): " << path << " has " << numFields
        << " fields (" << fieldWidth << "x" << fieldHeight << " samples/field, "
        << (layout == chd::format::FrameLayout::FRAME_NATIVE ? "frame-native" : "field-raster")
        << (plane == Plane::CHROMA ? ", chroma plane" : "") << ")";
    return true;
}

void CvbsCompositeSource::close()
{
    if (!isOpen) return;
    inputFile.close();
    isOpen = false;
    fieldCache.clear();
}

void CvbsCompositeSource::redeclareVideoSystem(chd::metadata::VideoSystem system, double fSC)
{
    videoParameters.system = system;
    videoParameters.fSC = fSC;
    if (system == chd::metadata::SECAM) {
        videoParameters.isSubcarrierLocked = false;
    }
}

const chd::metadata::LdDecodeMetaData::VideoParameters &CvbsCompositeSource::parameters() const
{
    return videoParameters;
}

chd::format::SignalState CvbsCompositeSource::signalState() const
{
    return state;
}

chd::format::SampleEncoding CvbsCompositeSource::sampleEncoding() const
{
    return encoding;
}

chd::format::HorizontalAlignment CvbsCompositeSource::horizontalAlignment() const
{
    return rowAlignment;
}

chd::format::FrameLayout CvbsCompositeSource::frameLayout() const
{
    return layout;
}

bool CvbsCompositeSource::isSourceValid() const
{
    return isOpen;
}

int32_t CvbsCompositeSource::getNumberOfAvailableFields() const
{
    return numFields;
}

int32_t CvbsCompositeSource::getFieldLength() const
{
    return fieldSamples;
}

CvbsCompositeSource::Data CvbsCompositeSource::getVideoField(int32_t fieldNumber,
                                                             int32_t startFieldLine,
                                                             int32_t endFieldLine)
{
    if (!isOpen) {
        throw std::runtime_error("CvbsCompositeSource::getVideoField(): source not open");
    }
    // 1-based field number, matching the canonical TBC convention.
    const int32_t fieldIndex = fieldNumber - 1;
    if (fieldIndex < 0 || fieldIndex >= numFields) {
        throw std::runtime_error("CvbsCompositeSource::getVideoField(): field out of range");
    }

    std::lock_guard<std::mutex> lock(ioMutex);

    const bool wholeField = (startFieldLine == -1 && endFieldLine == -1);
    if (wholeField) {
        if (const Data *cached = fieldCache.find(fieldIndex)) return *cached;
    }
    // Lines are 1-based. Convert to 0-based and validate.
    const int32_t first0 = wholeField ? 0 : startFieldLine - 1;
    const int32_t last0  = wholeField ? fieldHeight - 1 : endFieldLine - 1;
    if (first0 < 0 || last0 < first0 || last0 >= fieldHeight) {
        throw std::runtime_error("CvbsCompositeSource::getVideoField(): line range out of bounds");
    }

    Data data;
    if (layout == chd::format::FrameLayout::FRAME_NATIVE) {
        // Conform from the frame-addressed native stream: pure line
        // re-blocking on the same sample grid, plus the dummy padding line.
        const auto plan = chd::format::planFrameNativeFieldRead(
            *preset, fieldIndex, first0, last0, bytesPerSample);
        if (plan.numBytes > 0) data = readAndConvert(plan.startByte, plan.numBytes);
        data.resize(data.size() + plan.padSamples,
                    static_cast<uint16_t>(videoParameters.blanking16bIre));
    } else {
        const int64_t lineByteSize = static_cast<int64_t>(fieldWidth) * 2;
        const int64_t startByte =
            static_cast<int64_t>(fieldByteSize) * fieldIndex + lineByteSize * first0;
        const int64_t numBytes = lineByteSize * (last0 - first0 + 1);
        data = readAndConvert(startByte, numBytes);
    }

    if (wholeField) {
        fieldCache.insert(fieldIndex, data);
    }
    return data;
}

CvbsCompositeSource::Data CvbsCompositeSource::readAndConvert(int64_t startByte, int64_t numBytes)
{
    inputFile.clear();
    inputFile.seekg(startByte);
    if (!inputFile.good()) {
        throw std::runtime_error("CvbsCompositeSource::readAndConvert(): seek failed");
    }

    // Read the raw on-disk samples as int16_t (signedness handled by the
    // per-encoding converter; we use a signed buffer here so two's-complement
    // sign extension matches the spec for signed encodings).
    const int64_t numSamples = numBytes / 2;
    std::vector<int16_t> raw(static_cast<size_t>(numSamples));
    inputFile.read(reinterpret_cast<char *>(raw.data()), numBytes);
    if (inputFile.gcount() != numBytes) {
        throw std::runtime_error("CvbsCompositeSource::readAndConvert(): short read");
    }

    Data out(static_cast<size_t>(numSamples));
    const int32_t blanking10 = videoParameters.blanking16bIre / 64;
    if (plane == Plane::CHROMA) {
        // Centred chroma excursion (× 64) re-centred on blanking, clamped to
        // the uint16_t canonical domain.
        const int32_t blanking = videoParameters.blanking16bIre;
        for (size_t i = 0; i < raw.size(); ++i) {
            const int32_t v = blanking + chd::format::convertChromaSampleToCenteredCanonical(
                                             encoding, raw[i], blanking10);
            out[i] = static_cast<uint16_t>(std::clamp(v, 0, 65535));
        }
        return out;
    }
    for (size_t i = 0; i < raw.size(); ++i) {
        out[i] = chd::format::convertCompositeSampleToCanonical(encoding, raw[i], blanking10);
    }
    return out;
}

}  // namespace chd::reader
