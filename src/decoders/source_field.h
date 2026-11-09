// SPDX-License-Identifier: GPL-3.0-or-later
/************************************************************************

    sourcefield.h

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2019 Adam Sampson

    This file is part of ld-decode-tools.

    ld-chroma-decoder is free software: you can redistribute it and/or
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

#ifndef CHD_DECODERS_SOURCE_FIELD_H
#define CHD_DECODERS_SOURCE_FIELD_H

#include <cstdint>
#include <optional>
#include <vector>

#include "../metadata/core.h"
#include "../reader/source.h"

namespace chd::decoders {

// A field read from the input, with metadata and data
struct SourceField {
    chd::metadata::LdDecodeMetaData::Field field;
    chd::reader::Data data;

    // Black placeholder for a frame outside the input.
    bool isPadding = false;

    // 1-based number of this field in the source, as loaded.
    int32_t fieldNumber = 0;

    // Which version of this field's samples `data` holds, for matching work
    // between decode calls. Empty if they can't be matched.
    std::optional<uint64_t> inputRevision;

    // Set inputRevision across a window of frames from loadFields. While
    // correction is on, `correctionRevision` identifies it, and a frame with
    // dropouts takes that value because its samples depend on the correction.
    // Other frames hold the source's own samples, which are revision 0.
    static void markInputRevisions(std::vector<SourceField> &fields, std::optional<uint64_t> correctionRevision);

    // With subcarrier-locked 4fSC PAL sampling, the two fields of a frame are
    // horizontally misaligned by two samples, so the second moves left to
    // compensate. Its dropout positions move with it.
    static bool secondFieldIsShifted(const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters);
    static void shiftSecondField(chd::reader::Data &data, chd::metadata::DropOuts &dropOuts, uint16_t black);

    // Load a sequence of frames from the input files.
    //
    // fields will contain {lookbehind fields... [startIndex] real fields... [endIndex] lookahead fields...}.
    // Fields requested outside the bounds of the file will have dummy metadata and black data.
    // `firstFieldFirst` is the frame pairing order (see LdDecodeMetaData::getNumberOfFrames).
    static void loadFields(chd::reader::ISource &sourceVideo, chd::metadata::LdDecodeMetaData &ldDecodeMetaData,
                           bool firstFieldFirst, int32_t firstFrameNumber, int32_t numFrames,
                           int32_t lookBehindFrames, int32_t lookAheadFrames,
                           std::vector<SourceField> &fields, int32_t &startIndex, int32_t &endIndex);

    // Return the vertical offset of this field within the interlaced frame
    // (i.e. 0 for the top field, 1 for the bottom field).
    int32_t getOffset() const {
        return field.isFirstField ? 0 : 1;
    }

    // Return the first/last active line numbers within this field's data,
    // given the video parameters. firstActiveFrameLine/lastActiveFrameLine are
    // inclusive frame rows; the returned field-line range is half-open
    // ([first, last)), so getLastActiveLine() is one past the field line that
    // carries lastActiveFrameLine.
    int32_t getFirstActiveLine(const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters) const {
        return (videoParameters.firstActiveFrameLine + 1 - getOffset()) / 2;
    }
    int32_t getLastActiveLine(const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters) const {
        return ((videoParameters.lastActiveFrameLine - getOffset()) / 2) + 1;
    }
};

}  // namespace chd::decoders

#endif  // CHD_DECODERS_SOURCE_FIELD_H
