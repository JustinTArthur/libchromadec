// SPDX-License-Identifier: GPL-3.0-or-later
//
// Corrected frames kept for reuse. A frame's correction depends only on that
// frame, the dropout options and the extra sources, so the several decode calls
// that see a frame in their look-around window can share one result.

#ifndef CHD_DROPOUT_CORRECTED_FRAME_CACHE_H
#define CHD_DROPOUT_CORRECTED_FRAME_CACHE_H

#include <cstdint>
#include <tuple>

#include "../common/single_flight_cache.h"
#include "../reader/source.h"
#include "dropout_corrector.h"

namespace chd::dropout {

struct CorrectedFrame {
    chd::reader::Data firstFieldData;
    chd::reader::Data secondFieldData;
    DropoutCorrectionStats stats;
};

struct CorrectedFrameKey {
    int32_t firstFieldNumber = 0;
    int32_t secondFieldNumber = 0;
    uint64_t correctionRevision = 0;

    bool operator<(const CorrectedFrameKey &other) const {
        return std::tie(firstFieldNumber, secondFieldNumber, correctionRevision)
             < std::tie(other.firstFieldNumber, other.secondFieldNumber, other.correctionRevision);
    }
};

using CorrectedFrameCache = chd::SingleFlightCache<CorrectedFrameKey, CorrectedFrame>;

}  // namespace chd::dropout

#endif  // CHD_DROPOUT_CORRECTED_FRAME_CACHE_H
