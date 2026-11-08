// SPDX-License-Identifier: GPL-3.0-or-later
/************************************************************************

    correctorpool.cpp

    ld-dropout-correct - Dropout correction for ld-decode
    Copyright (C) 2018-2025 Simon Inns
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

#include "multi_source_alignment.h"

#include <utility>

#include "../common/log.h"
#include "../metadata/vbi_decoder.h"

namespace chd::dropout {

using chd::metadata::LdDecodeMetaData;

MultiSourceAlignment::MultiSourceAlignment(std::vector<Source> sources)
    : sources(std::move(sources))
{
    setMinAndMaxVbiFrames();
}

bool MultiSourceAlignment::hasVbi(int32_t sourceNumber) const
{
    return sourceHasVbi[sourceNumber];
}

// For frame `frameNumber` in the primary source (source 0), return the
// sequential frame number that holds the same picture in source `sourceNo`,
// or -1 if that source does not cover it. The primary returns the frame
// itself; VBI-bearing sources are matched by disc frame number; sources
// without VBI codes (for example CVBS captures) align positionally.
int32_t MultiSourceAlignment::sourceFrameForPrimaryFrame(int32_t frameNumber, int32_t sourceNo) const
{
    // No need to perform VBI frame number mapping on the first source
    if (sourceNo == 0) return frameNumber;

    // Without VBI codes on both the primary and this source there is nothing
    // to register against, so align positionally (the same sequential frame).
    if (!sourceHasVbi[0] || !sourceHasVbi[sourceNo]) return frameNumber;

    // Get the current VBI frame number based on the first source
    int32_t currentVbiFrame = convertSequentialFrameNumberToVbi(frameNumber, 0);
    if (currentVbiFrame >= sourceMinimumVbiFrame[sourceNo] && currentVbiFrame <= sourceMaximumVbiFrame[sourceNo]) {
        // Use VBI frame number mapping to get the same frame from the
        // current additional source
        int32_t currentSourceFrameNumber = convertVbiFrameNumberToSequential(currentVbiFrame, sourceNo);
        return currentSourceFrameNumber;
    }
    return -1;
}

void MultiSourceAlignment::setMinAndMaxVbiFrames()
{
    // Determine the number of sources available
    int32_t numberOfSources = static_cast<int32_t>(sources.size());

    // Resize vectors
    sourceHasVbi.resize(numberOfSources);
    sourceDiscTypeCav.resize(numberOfSources);
    sourceMaximumVbiFrame.resize(numberOfSources);
    sourceMinimumVbiFrame.resize(numberOfSources);

    for (int32_t sourceNumber = 0; sourceNumber < numberOfSources; sourceNumber++) {
        // Determine the disc type and max/min VBI frame numbers
        VbiDecoder vbiDecoder;
        int32_t cavCount = 0;
        int32_t clvCount = 0;
        int32_t cavMin = 1000000;
        int32_t cavMax = 0;
        int32_t clvMin = 1000000;
        int32_t clvMax = 0;

        sourceMinimumVbiFrame[sourceNumber] = 0;
        sourceMaximumVbiFrame[sourceNumber] = 0;
        sourceDiscTypeCav[sourceNumber] = false;

        // Using sequential frame numbering starting from 1
        LdDecodeMetaData &meta = *sources[sourceNumber].metadata;
        const bool firstFieldFirst = sources[sourceNumber].firstFieldFirst;
        for (int32_t seqFrame = 1; seqFrame <= meta.getNumberOfFrames(firstFieldFirst); seqFrame++) {
            // Get the VBI data and then decode
            auto vbi1 = meta.getFieldVbi(meta.getFirstFieldNumber(seqFrame, firstFieldFirst)).vbiData;
            auto vbi2 = meta.getFieldVbi(meta.getSecondFieldNumber(seqFrame, firstFieldFirst)).vbiData;
            VbiDecoder::Vbi vbi = vbiDecoder.decodeFrame(vbi1[0], vbi1[1], vbi1[2], vbi2[0], vbi2[1], vbi2[2]);

            // Look for a complete, valid CAV picture number or CLV time-code
            if (vbi.picNo > 0) {
                cavCount++;

                if (vbi.picNo < cavMin) cavMin = vbi.picNo;
                if (vbi.picNo > cavMax) cavMax = vbi.picNo;
            }

            if (vbi.clvHr != -1 && vbi.clvMin != -1 &&
                    vbi.clvSec != -1 && vbi.clvPicNo != -1) {
                clvCount++;

                LdDecodeMetaData::ClvTimecode timecode;
                timecode.hours = vbi.clvHr;
                timecode.minutes = vbi.clvMin;
                timecode.seconds = vbi.clvSec;
                timecode.pictureNumber = vbi.clvPicNo;
                int32_t cvFrameNumber = meta.convertClvTimecodeToFrameNumber(timecode);

                if (cvFrameNumber < clvMin) clvMin = cvFrameNumber;
                if (cvFrameNumber > clvMax) clvMax = cvFrameNumber;
            }
        }
        chd::log::debug() << "MultiSourceAlignment::setMinAndMaxVbiFrames(): Got" << cavCount << "CAV picture codes and" << clvCount << "CLV timecodes";

        // If the metadata has no picture numbers or time-codes (for example a
        // CVBS capture), the source carries no VBI; the caller aligns it
        // positionally instead of by disc frame number.
        if (cavCount == 0 && clvCount == 0) {
            chd::log::debug() << "MultiSourceAlignment::setMinAndMaxVbiFrames(): Source does not seem to contain valid CAV picture numbers or CLV time-codes - aligning positionally";
            sourceHasVbi[sourceNumber] = false;
            continue;
        }
        sourceHasVbi[sourceNumber] = true;

        // Determine disc type
        if (cavCount > clvCount) {
            sourceDiscTypeCav[sourceNumber] = true;
            chd::log::debug() << "MultiSourceAlignment::setMinAndMaxVbiFrames(): Got" << cavCount << "valid CAV picture numbers - source disc type is CAV";
            chd::log::info().nospace() << "Source #" << sourceNumber << " has a disc type of CAV (uses VBI frame numbers)";

            sourceMaximumVbiFrame[sourceNumber] = cavMax;
            sourceMinimumVbiFrame[sourceNumber] = cavMin;
        } else {
            sourceDiscTypeCav[sourceNumber] = false;
            chd::log::debug() << "MultiSourceAlignment::setMinAndMaxVbiFrames(): Got" << clvCount << "valid CLV picture numbers - source disc type is CLV";
            chd::log::info().nospace() << "Source #" << sourceNumber << " has a disc type of CLV (uses VBI time codes)";

            sourceMaximumVbiFrame[sourceNumber] = clvMax;
            sourceMinimumVbiFrame[sourceNumber] = clvMin;
        }

        chd::log::info().nospace() << "Source #" << sourceNumber << " has a VBI frame number range of " << sourceMinimumVbiFrame[sourceNumber] << " to " <<
            sourceMaximumVbiFrame[sourceNumber];
    }
}

// Method to convert the first source sequential frame number to a VBI frame number
int32_t MultiSourceAlignment::convertSequentialFrameNumberToVbi(int32_t sequentialFrameNumber, int32_t sourceNumber) const
{
    return (sourceMinimumVbiFrame[sourceNumber] - 1) + sequentialFrameNumber;
}

// Method to convert a VBI frame number to a sequential frame number
int32_t MultiSourceAlignment::convertVbiFrameNumberToSequential(int32_t vbiFrameNumber, int32_t sourceNumber) const
{
    // Offset the VBI frame number to get the sequential source frame number
    return vbiFrameNumber - sourceMinimumVbiFrame[sourceNumber] + 1;
}

// Method that returns a vector of the sources that contain data for the required VBI frame number
std::vector<int32_t> MultiSourceAlignment::getAvailableSourcesForFrame(int32_t vbiFrameNumber) const
{
    std::vector<int32_t> availableSourcesForFrame;
    for (int32_t sourceNo = 0; sourceNo < static_cast<int32_t>(sources.size()); sourceNo++) {
        if (vbiFrameNumber >= sourceMinimumVbiFrame[sourceNo] && vbiFrameNumber <= sourceMaximumVbiFrame[sourceNo]) {
            // Get the field numbers for the frame
            LdDecodeMetaData &meta = *sources[sourceNo].metadata;
            const bool firstFieldFirst = sources[sourceNo].firstFieldFirst;
            int32_t firstFieldNumber = meta.getFirstFieldNumber(convertVbiFrameNumberToSequential(vbiFrameNumber, sourceNo), firstFieldFirst);
            int32_t secondFieldNumber = meta.getSecondFieldNumber(convertVbiFrameNumberToSequential(vbiFrameNumber, sourceNo), firstFieldFirst);

            // Ensure the frame is not a padded field (i.e. missing)
            if (!(meta.getField(firstFieldNumber).pad &&
                  meta.getField(secondFieldNumber).pad)) {
                availableSourcesForFrame.push_back(sourceNo);
            }
        }
    }

    return availableSourcesForFrame;
}

}  // namespace chd::dropout
