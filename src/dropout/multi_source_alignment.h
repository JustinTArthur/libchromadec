// SPDX-License-Identifier: GPL-3.0-or-later
/************************************************************************

    correctorpool.h

    ld-dropout-correct - Dropout correction for ld-decode
    Copyright (C) 2018-2020 Simon Inns
    Copyright (C) 2019-2020 Adam Sampson

    This file is part of ld-decode-tools.

    ld-dropout-correct is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#ifndef MULTI_SOURCE_ALIGNMENT_H
#define MULTI_SOURCE_ALIGNMENT_H

#include <cstdint>
#include <vector>

#include "../metadata/core.h"

namespace chd::dropout {

// Registers multiple captures of the same disc against one another using the
// CAV picture numbers / CLV time-codes carried in each field's VBI metadata,
// so frame N of the primary source can be matched to the same physical disc
// frame in additional captures even when those captures start at different
// points or skip frames. Sources that carry no usable VBI codes (for example
// CVBS captures) report hasVbi() == false; the caller aligns those positionally.
class MultiSourceAlignment
{
public:
    // One capture and the frame pairing order to read it with (see
    // LdDecodeMetaData::getNumberOfFrames).
    struct Source {
        chd::metadata::LdDecodeMetaData *metadata;
        bool firstFieldFirst;
    };

    // sources[0] is the primary capture; sources[1..] are additional captures.
    // Scans every field's VBI to determine each source's disc type and the
    // range of VBI frame numbers it covers.
    explicit MultiSourceAlignment(std::vector<Source> sources);

    // True if the source carried valid CAV picture numbers or CLV time-codes.
    bool hasVbi(int32_t sourceNumber) const;

    // The sequential frame number in `sourceNo` holding the same picture as
    // `frameNumber` in the primary source, or -1 if that source does not cover
    // it. Sources without VBI are matched positionally.
    int32_t sourceFrameForPrimaryFrame(int32_t frameNumber, int32_t sourceNo) const;

    int32_t convertSequentialFrameNumberToVbi(int32_t sequentialFrameNumber, int32_t sourceNumber) const;
    int32_t convertVbiFrameNumberToSequential(int32_t vbiFrameNumber, int32_t sourceNumber) const;
    std::vector<int32_t> getAvailableSourcesForFrame(int32_t vbiFrameNumber) const;

private:
    void setMinAndMaxVbiFrames();

    std::vector<Source> sources;

    // Per-source VBI frame information, indexed by source number.
    std::vector<bool> sourceHasVbi;
    std::vector<bool> sourceDiscTypeCav;
    std::vector<int32_t> sourceMinimumVbiFrame;
    std::vector<int32_t> sourceMaximumVbiFrame;
};

}  // namespace chd::dropout

#endif // MULTI_SOURCE_ALIGNMENT_H
