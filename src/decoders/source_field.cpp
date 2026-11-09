// SPDX-License-Identifier: GPL-3.0-or-later
/************************************************************************

    sourcefield.cpp

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

#include "source_field.h"

#include "../common/log.h"
#include "../reader/source.h"

namespace chd::decoders {

void SourceField::loadFields(chd::reader::ISource &sourceVideo, chd::metadata::LdDecodeMetaData &ldDecodeMetaData,
                             bool firstFieldFirst, int32_t firstFrameNumber, int32_t numFrames,
                             int32_t lookBehindFrames, int32_t lookAheadFrames,
                             std::vector<SourceField> &fields, int32_t &startIndex, int32_t &endIndex)
{
    const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters = ldDecodeMetaData.getVideoParameters();

    // Work out indexes.
    // fields will contain {lookbehind fields... [startIndex] real fields... [endIndex] lookahead fields...}.
    startIndex = 2 * lookBehindFrames;
    endIndex = startIndex + (2 * numFrames);
    fields.resize(endIndex + (2 * lookAheadFrames));

    // Populate fields
    const int32_t numInputFrames = ldDecodeMetaData.getNumberOfFrames(firstFieldFirst);
    const int32_t numMetadataFields = ldDecodeMetaData.getNumberOfFields();
    const int32_t numSourceFields = sourceVideo.getNumberOfAvailableFields();
    bool warnedMetadataFieldRange = false;
    bool warnedSourceFieldRange = false;
    int32_t frameNumber = firstFrameNumber - lookBehindFrames;
    for (size_t i = 0; i < fields.size(); i += 2) {

        // Is this frame outside the bounds of the input file?
        // If so, use real metadata (from frame 1) and black fields.
        bool useBlankFrame = frameNumber < 1 || frameNumber > numInputFrames;

        // Resolve field numbers for this frame (using frame 1 if outside bounds).
        int32_t metadataFrame = useBlankFrame ? 1 : frameNumber;
        int32_t firstFieldNumber = ldDecodeMetaData.getFirstFieldNumber(metadataFrame, firstFieldFirst);
        int32_t secondFieldNumber = ldDecodeMetaData.getSecondFieldNumber(metadataFrame, firstFieldFirst);

        const auto fieldNumbersAreMetadataSafe = [numMetadataFields](int32_t firstField, int32_t secondField) {
            return firstField >= 1 && secondField >= 1 && firstField <= numMetadataFields
                   && secondField <= numMetadataFields;
        };

        bool metadataFieldRangeInvalid = !fieldNumbersAreMetadataSafe(firstFieldNumber, secondFieldNumber);
        const bool sourceFieldRangeInvalid =
            numSourceFields != -1 && (firstFieldNumber > numSourceFields || secondFieldNumber > numSourceFields);

        // Metadata and source files can become mismatched (for example if the source
        // TBC is truncated while metadata still references more fields). In that case,
        // fall back to blank frames instead of requesting out-of-range field data.
        if (!useBlankFrame && (metadataFieldRangeInvalid || sourceFieldRangeInvalid)) {
            if (metadataFieldRangeInvalid && !warnedMetadataFieldRange) {
                chd::log::warn() << "SourceField::loadFields(): Metadata field range invalid for frame" << frameNumber
                                 << "(first=" << firstFieldNumber << "second=" << secondFieldNumber
                                 << "metadata fields=" << numMetadataFields << "). Using blank fallback frames.";
                warnedMetadataFieldRange = true;
            }

            if (sourceFieldRangeInvalid && !warnedSourceFieldRange) {
                chd::log::warn() << "SourceField::loadFields(): Source file has fewer fields than metadata for frame"
                                 << frameNumber << "(first=" << firstFieldNumber << "second=" << secondFieldNumber
                                 << "source fields=" << numSourceFields
                                 << "). Source file appears damaged or truncated; using blank fallback frames.";
                warnedSourceFieldRange = true;
            }

            useBlankFrame = true;
            metadataFrame = 1;
            firstFieldNumber = ldDecodeMetaData.getFirstFieldNumber(metadataFrame, firstFieldFirst);
            secondFieldNumber = ldDecodeMetaData.getSecondFieldNumber(metadataFrame, firstFieldFirst);
            metadataFieldRangeInvalid = !fieldNumbersAreMetadataSafe(firstFieldNumber, secondFieldNumber);
        }

        // Fetch the input metadata
        if (!metadataFieldRangeInvalid) {
            fields[i].field = ldDecodeMetaData.getField(firstFieldNumber);
            fields[i + 1].field = ldDecodeMetaData.getField(secondFieldNumber);
        } else {
            // If metadata is irrecoverably inconsistent, synthesize minimal field
            // metadata and continue with blank picture data.
            fields[i].field = chd::metadata::LdDecodeMetaData::Field();
            fields[i + 1].field = chd::metadata::LdDecodeMetaData::Field();
            fields[i].field.isFirstField = true;
            fields[i + 1].field.isFirstField = false;
            useBlankFrame = true;
        }

        const uint16_t black = videoParameters.black16bIre;

        fields[i].isPadding = useBlankFrame;
        fields[i + 1].isPadding = useBlankFrame;

        if (useBlankFrame) {
            // Fill both fields with black
            fields[i].data.assign(sourceVideo.getFieldLength(), black);
            fields[i + 1].data.assign(sourceVideo.getFieldLength(), black);
        } else {
            // Fetch the input fields
            fields[i].data = sourceVideo.getVideoField(firstFieldNumber);
            fields[i + 1].data = sourceVideo.getVideoField(secondFieldNumber);

            if ((videoParameters.system == chd::metadata::PAL || videoParameters.system == chd::metadata::PAL_M) && videoParameters.isSubcarrierLocked) {
                // With subcarrier-locked 4fSC PAL sampling, we have four
                // "extra" samples over the course of the frame, so the two
                // fields will be horizontally misaligned by two samples. Shift
                // the second field to the left to compensate.
                //
                // XXX This should be done elsewhere, as it affects other tools
                // too.

                fields[i + 1].data.erase(fields[i + 1].data.begin(), fields[i + 1].data.begin() + 2);
                for (int j = 0; j < 2; j++) {
                    fields[i + 1].data.push_back(black);
                }
            }
        }

        frameNumber++;
    }
}

}  // namespace chd::decoders
