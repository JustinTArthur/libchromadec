// SPDX-License-Identifier: GPL-3.0-or-later
//
// CvbsCompositeSource — ISource implementation for a CVBS file format
// composite signal data file (`<basename>.cvbs`) or one plane of a
// dual-file Y/C layout (`<basename>.cvbsy` luma, `<basename>.cvbsc` chroma).
//
// On open the source pairs the on-disk binary with a Video Standard Preset,
// Sample Encoding Preset, and Signal State Preset. Per-encoding amplitude
// conversion happens at load time so the decoder pipeline always sees
// canonical uint16_t samples in the canonical TBC convention (10-bit value × 64,
// blanking at 16384). A `.cvbsy` follows the composite level definitions and
// reads exactly like a `.cvbs`. A `.cvbsc` holds the chroma excursion centred
// on 10-bit 512 (spec: sample-encoding-presets.md); in chroma-plane mode the
// excursion is re-centred on blanking so the colour decoders see the same
// chroma-only, composite-shaped plane a chroma `.tbc` has.
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
#include <vector>

#include "../format/sample_encoding.h"
#include "../format/signal_state.h"
#include "../format/video_standards.h"
#include "../metadata/core.h"
#include "field_cache.h"
#include "source.h"

namespace chd::reader {

class CvbsCompositeSource : public ISource
{
public:
    using Data = chd::reader::Data;

    // Which signal the file holds. COMPOSITE covers `.cvbs` and `.cvbsy`
    // (both use the composite level definitions); CHROMA is a `.cvbsc`.
    enum class Plane { COMPOSITE, CHROMA };

    CvbsCompositeSource();
    ~CvbsCompositeSource() override;

    // Open a sample file with the given preset triple. The Video Standard
    // Preset fixes the field geometry; the Sample Encoding Preset selects
    // amplitude conversion; the Signal State Preset controls whether
    // normative sample-count constraints apply and is reported back via
    // signalState().
    //
    // layoutOverride forces the container layout; UNKNOWN resolves it from
    // declaredFrames (the `.meta` frame count) and the file size.
    // subcarrierLockedOverride marks a subcarrier-locked field raster.
    //
    // A frame-native file's row alignment is measured from its sync edges. A
    // chroma plane has no sync, so a CHROMA open takes the alignment the
    // caller measured on the matching luma plane via frameNativeAlignment
    // (ignored for a field raster; a missing value falls back to sync-start).
    //
    // Returns true on success. On failure, the source is left invalid.
    bool open(const std::string &path,
              const chd::format::VideoStandardPreset &videoStandard,
              chd::format::SampleEncoding sampleEncoding,
              chd::format::SignalState signalState,
              std::optional<int32_t> blackLevelOverride = std::nullopt,
              chd::format::FrameLayout layoutOverride = chd::format::FrameLayout::UNKNOWN,
              std::optional<int64_t> declaredFrames = std::nullopt,
              std::optional<bool> subcarrierLockedOverride = std::nullopt,
              Plane plane = Plane::COMPOSITE,
              std::optional<chd::format::HorizontalAlignment> frameNativeAlignment =
                  std::nullopt);

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
    // produced by per-encoding amplitude conversion (composite levels, or the
    // centred chroma excursion re-centred on blanking for a CHROMA plane).
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
    Plane                               plane = Plane::COMPOSITE;
    chd::metadata::LdDecodeMetaData::VideoParameters videoParameters;

    // Whole-field cache (matches TbcSource's behaviour). Serialised by
    // ioMutex.
    FieldCache fieldCache;
    mutable std::mutex ioMutex;
};

}  // namespace chd::reader

#endif  // CHD_READER_CVBS_COMPOSITE_SOURCE_H
