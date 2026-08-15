// SPDX-License-Identifier: GPL-3.0-or-later
//
// CvbsYcSource — ISource implementation for the CVBS file format
// specification's dual-file YC layout (`<basename>.cvbsy` + `<basename>.cvbsc`).
//
// The .cvbsy file follows the same level definitions as the composite output;
// the .cvbsc file uses a centred 10-bit representation with chroma zero at
// sample value 512 (spec: sample-encoding-presets.md). On read this source
// synthesises a composite-shaped sample:
//
//   composite[i] = luma_canonical[i] + chroma_centered_canonical[i]
//
// where luma_canonical is in the canonical TBC convention (10-bit × 64,
// blanking at 16384) and chroma_centered_canonical is the signed excursion
// around zero scaled × 64. The downstream decoder pipeline sees a
// composite-shaped uint16_t buffer indistinguishable from a real
// CvbsCompositeSource read.
//
// This always materialises the synthesised composite; a future optimisation
// can present pre-separated chroma directly to PALcolour/transform-PAL (plan
// §5 note).

#ifndef CHD_READER_CVBS_YC_SOURCE_H
#define CHD_READER_CVBS_YC_SOURCE_H

#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../format/sample_encoding.h"
#include "../format/signal_state.h"
#include "../format/video_standards.h"
#include "../metadata/core.h"
#include "source.h"

namespace chd::reader {

class CvbsYcSource : public ISource
{
public:
    using Data = chd::reader::Data;

    CvbsYcSource();
    ~CvbsYcSource() override;

    // Open a YC pair. Both files must have identical size and follow the
    // same preset triple. Layout resolution matches CvbsCompositeSource:
    // layoutOverride wins, else declaredFrames + file size decide.
    bool open(const std::string &yPath,
              const std::string &cPath,
              const chd::format::VideoStandardPreset &videoStandard,
              chd::format::SampleEncoding sampleEncoding,
              chd::format::SignalState signalState,
              std::optional<int32_t> blackLevelOverride = std::nullopt,
              chd::format::FrameLayout layoutOverride = chd::format::FrameLayout::UNKNOWN,
              std::optional<int64_t> declaredFrames = std::nullopt,
              std::optional<bool> subcarrierLockedOverride = std::nullopt);

    void close();

    // Re-declare the colour standard over the opened preset's; see
    // CvbsCompositeSource::redeclareVideoSystem.
    void redeclareVideoSystem(chd::metadata::VideoSystem system, double fSC);

    // ISource implementation -------------------------------------------------
    const chd::metadata::LdDecodeMetaData::VideoParameters &parameters() const override;
    chd::format::SignalState     signalState()    const override;
    chd::format::SampleEncoding  sampleEncoding() const override;
    chd::format::HorizontalAlignment horizontalAlignment() const override;
    chd::format::FrameLayout     frameLayout()    const override;

    bool isSourceValid() const override;
    int32_t getNumberOfAvailableFields() const override;
    int32_t getFieldLength() const override;

    Data getVideoField(int32_t fieldNumber,
                       int32_t startFieldLine = -1,
                       int32_t endFieldLine = -1) override;

private:
    Data readAndSynthesise(int64_t startByte, int64_t numBytes);

    std::ifstream yFile;
    std::ifstream cFile;
    bool isOpen = false;
    int64_t fileSize = 0;

    int32_t fieldWidth = 0;
    int32_t fieldHeight = 0;
    int32_t fieldSamples = 0;
    int32_t fieldByteSize = 0;
    int32_t numFields = 0;

    chd::format::FrameLayout layout = chd::format::FrameLayout::FIELD_RASTER;

    // Horizontal alignment served by parameters(): declared for a field
    // raster; for a frame-native file, measured once from early sync edges
    // when open() reads the file.
    chd::format::HorizontalAlignment rowAlignment = chd::format::HorizontalAlignment::SYNC_START;

    const chd::format::VideoStandardPreset *preset = nullptr;
    int32_t bytesPerSample = 2;

    chd::format::VideoStandard          standardEnum;
    chd::format::SampleEncoding         encoding;
    chd::format::SignalState            state;
    chd::metadata::LdDecodeMetaData::VideoParameters videoParameters;

    std::unordered_map<int32_t, Data> fieldCache;
    mutable std::mutex ioMutex;
};

}  // namespace chd::reader

#endif  // CHD_READER_CVBS_YC_SOURCE_H
