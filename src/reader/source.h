// SPDX-License-Identifier: GPL-3.0-or-later
//
// ISource — abstract field reader interface for the decoder pipeline.
//
// As a reader abstraction over CVBS + TBC, the decoder pipeline
// only needs three things from its input:
//   - a set of video parameters (sample rate, line geometry, IRE levels)
//   - per-field sample data (uint16_t composite samples)
//   - the count of available fields
//
// One ISource interface, three concrete implementations:
//   - TbcSource             — ld-decode `.tbc` + sqlite/json sidecar
//   - CvbsCompositeSource   — CVBS spec `.cvbs` + `.meta` sqlite sidecar
//   - CvbsYcSource          — CVBS spec dual-file `.cvbsy` + `.cvbsc` + `.meta`
//
// The signal-state and sample-encoding accessors let decoders refuse
// incompatible inputs (raw-uncorrected signals) and let the I/O layer
// transparently fold per-encoding amplitude conversion at load time so the
// pipeline always sees samples in the canonical TBC convention (uint16_t,
// 10-bit value × 64, blanking at 16384).

#ifndef CHD_READER_SOURCE_H
#define CHD_READER_SOURCE_H

#include <cstdint>
#include <vector>

#include "../format/sample_encoding.h"
#include "../format/signal_state.h"
#include "../format/video_standards.h"
#include "../metadata/core.h"

namespace chd::reader {

// Canonical per-field sample buffer: uint16_t composite samples in the
// canonical TBC scaled-10-bit domain. This alias is what all decoder hot loops
// expect and matches the pre-Phase-D SourceVideo::Data type.
using Data = std::vector<uint16_t>;

class ISource {
public:
    virtual ~ISource() = default;

    // Disallow copying (sources own file handles + caches)
    ISource(const ISource &) = delete;
    ISource &operator=(const ISource &) = delete;

    // Static information about the video signal.
    virtual const chd::metadata::LdDecodeMetaData::VideoParameters &parameters() const = 0;
    virtual chd::format::SignalState     signalState()    const = 0;
    virtual chd::format::SampleEncoding  sampleEncoding() const = 0;

    // Horizontal alignment of the stored rows (where 0H sits within a row),
    // as served: the cut the burst/active windows in parameters() were built
    // for. Drives the standard-sample <-> row-sample conversion in the C ABI.
    virtual chd::format::HorizontalAlignment horizontalAlignment() const = 0;

    // Container addressing of the backing file(s). Every `.tbc` is a field
    // raster; CVBS sources report the layout resolved at open time.
    virtual chd::format::FrameLayout     frameLayout()    const = 0;

    // True if the source successfully opened its backing file(s).
    virtual bool isSourceValid() const = 0;

    // Number of fields the source can provide. Returns -1 if unknown.
    virtual int32_t getNumberOfAvailableFields() const = 0;

    // Length in samples of a single field (fieldWidth * fieldHeight).
    virtual int32_t getFieldLength() const = 0;

    // Retrieve a range of lines from one field, or the whole field if
    // startFieldLine and endFieldLine are both -1. fieldNumber is 1-based to
    // match the 1-based convention. Returns canonical composite samples.
    // Implementations must be safe to call from multiple threads concurrently
    // (worker threads may call it concurrently).
    virtual Data getVideoField(int32_t fieldNumber,
                               int32_t startFieldLine = -1,
                               int32_t endFieldLine = -1) = 0;

protected:
    ISource() = default;
};

}  // namespace chd::reader

#endif  // CHD_READER_SOURCE_H
