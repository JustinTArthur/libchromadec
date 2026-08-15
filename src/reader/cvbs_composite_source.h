// SPDX-License-Identifier: GPL-3.0-or-later
//
// CvbsCompositeSource — ISource implementation for the CVBS file format
// specification's `<basename>.cvbs` layout (single file containing
// luma+chroma combined into one signal).
//
// On open the source pairs the on-disk binary with a Video Standard Preset,
// Sample Encoding Preset, and Signal State Preset. Per-encoding amplitude
// conversion happens at load time so the decoder pipeline always sees
// canonical uint16_t samples in the canonical TBC convention (10-bit value × 64,
// blanking at 16384).
//
// Files are addressed per the resolved FrameLayout. A field-raster file is
// blocked like a .tbc (fixed rows of fieldWidth samples, both fields padded
// to the same height) and slices line-major per field. A frame-native file
// stores the spec's exact continuous frame totals; reads conform it onto the
// same field raster by re-blocking lines on the unchanged sample grid and
// blanking-filling past the native total, and open() measures the rows'
// horizontal alignment (0H) from early sync edges to select the matching
// burst/active windows.

#ifndef CHD_READER_CVBS_COMPOSITE_SOURCE_H
#define CHD_READER_CVBS_COMPOSITE_SOURCE_H

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

class CvbsCompositeSource : public ISource
{
public:
    using Data = chd::reader::Data;

    CvbsCompositeSource();
    ~CvbsCompositeSource() override;

    // Open a composite file with the given preset triple. The Video Standard
    // Preset fixes the field geometry; the Sample Encoding Preset selects
    // amplitude conversion; the Signal State Preset controls whether
    // normative sample-count constraints apply and is reported back via
    // signalState().
    //
    // layoutOverride forces the container layout; UNKNOWN resolves it from
    // declaredFrames (the `.meta` frame count) and the file size.
    // subcarrierLockedOverride marks a subcarrier-locked field raster.
    //
    // Returns true on success. On failure, the source is left invalid.
    bool open(const std::string &compositePath,
              const chd::format::VideoStandardPreset &videoStandard,
              chd::format::SampleEncoding sampleEncoding,
              chd::format::SignalState signalState,
              std::optional<int32_t> blackLevelOverride = std::nullopt,
              chd::format::FrameLayout layoutOverride = chd::format::FrameLayout::UNKNOWN,
              std::optional<int64_t> declaredFrames = std::nullopt,
              std::optional<bool> subcarrierLockedOverride = std::nullopt);

    void close();

    // Re-declare the colour standard over the opened preset's, for captures
    // the CVBS spec cannot yet name (SECAM stored under the byte-compatible
    // 625/50 PAL preset). Geometry is untouched; a SECAM re-declaration also
    // clears the subcarrier-lock flag (line-sequential FM chroma has no QAM
    // subcarrier to lock to).
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
    // Read the requested raw byte range from the file under ioMutex, then
    // return a freshly-allocated buffer of canonical-domain uint16_t samples
    // produced by per-encoding amplitude conversion.
    Data readAndConvert(int64_t startByte, int64_t numBytes);

    std::ifstream inputFile;
    bool isOpen = false;
    int64_t fileSize = 0;

    // Field geometry derived from the Video Standard Preset.
    int32_t fieldWidth = 0;          // samples per "line" (orthogonal-line simplification for PAL)
    int32_t fieldHeight = 0;         // lines per field (262/263 for NTSC/PAL_M; 312/313 for PAL)
    int32_t fieldSamples = 0;        // fieldWidth * fieldHeight
    int32_t fieldByteSize = 0;       // fieldSamples * 2

    int32_t numFields = 0;

    // Container addressing resolved at open time. For FRAME_NATIVE, fields
    // are conformed onto the field raster from whole native frames.
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

    // Whole-field cache (matches TbcSource's behaviour). Serialised by
    // ioMutex.
    std::unordered_map<int32_t, Data> fieldCache;
    mutable std::mutex ioMutex;
};

}  // namespace chd::reader

#endif  // CHD_READER_CVBS_COMPOSITE_SOURCE_H
