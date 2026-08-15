// SPDX-License-Identifier: GPL-3.0-or-later
//
// CVBS metadata sidecar reader (`<basename>.meta`).
//
// Reads the `cvbs_file` table defined in the CVBS file format specification
// (cvbs-file-format-specification/docs/index.md, user_version 7 to 10). The
// reader validates the preset triple against the format/ DATA tables and
// returns the resolved presets along with file-level fields like
// `number_of_sequential_frames` and `black_level`. Audio metadata (a sibling
// table from user_version 9 on) is not read.

#ifndef CHD_METADATA_CVBS_METADATA_SQLITE_H
#define CHD_METADATA_CVBS_METADATA_SQLITE_H

#include <cstdint>
#include <optional>
#include <string>

#include "../format/sample_encoding.h"
#include "../format/signal_state.h"
#include "../format/video_standards.h"

namespace chd::metadata {

// `signal_type` column values, declared by the CVBS spec.
enum class CvbsSignalType {
    Composite = 0,
    Yc,
};

// Resolved row from the `cvbs_file` table. All preset fields point into the
// static format/ tables so callers get stable references.
struct CvbsMetadata {
    const chd::format::VideoStandardPreset  *videoStandard;
    chd::format::SampleEncoding              sampleEncoding;
    chd::format::SignalState                 signalState;
    CvbsSignalType                           signalType;

    std::string decoder;
    std::optional<std::string>               gitBranch;
    std::optional<std::string>               gitCommit;
    std::optional<int64_t>                   numberOfSequentialFrames;
    std::optional<int32_t>                   blackLevelOverride;
    std::optional<bool>                      hasNonstandardValues;
    std::optional<std::string>               captureNotes;
};

// Read the CVBS `<basename>.meta` sidecar at metaPath. Returns nullopt and
// sets the thread-local error detail on failure. The reader rejects unknown
// preset names per the spec ("an unrecognised preset name must not be
// silently interpreted").
std::optional<CvbsMetadata> readCvbsMetadata(const std::string &metaPath);

}  // namespace chd::metadata

#endif  // CHD_METADATA_CVBS_METADATA_SQLITE_H
