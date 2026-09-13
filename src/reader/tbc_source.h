// SPDX-License-Identifier: GPL-3.0-or-later
/************************************************************************

    sourcevideo.h

    ld-decode-tools TBC library
    Copyright (C) 2018-2019 Simon Inns

    This file is part of ld-decode-tools.

    ld-decode-tools is free software: you can redistribute it and/or
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

#ifndef CHD_READER_TBC_SOURCE_H
#define CHD_READER_TBC_SOURCE_H

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "../format/sample_encoding.h"
#include "../format/signal_state.h"
#include "../metadata/core.h"
#include "field_cache.h"
#include "source.h"

namespace chd::reader {

class TbcSource : public ISource
{
public:
    // A vector of timebase-corrected video samples.
    // This is usually a complete field, but it may be a partial field if
    // you've requested fewer lines from getVideoField (or if you've sliced it
    // yourself).
    using Data = chd::reader::Data;

    TbcSource();
    ~TbcSource() override;

    // File handling methods
    bool open(std::string filename, int32_t _fieldLength, int32_t _fieldLineLength = -1);
    void close(void);

    // Bind a VideoParameters reference that outlives the source. Required
    // before parameters() may be called. The metadata is owned by the
    // chd_video handle in the C ABI layer; the source holds only a
    // non-owning pointer.
    void bindVideoParameters(const chd::metadata::LdDecodeMetaData::VideoParameters &vp);

    // ISource implementation -------------------------------------------------
    const chd::metadata::LdDecodeMetaData::VideoParameters &parameters() const override;
    chd::format::SignalState     signalState()    const override;
    chd::format::SampleEncoding  sampleEncoding() const override;
    chd::format::HorizontalAlignment horizontalAlignment() const override;
    chd::format::FrameLayout     frameLayout()    const override
    {
        return chd::format::FrameLayout::FIELD_RASTER;
    }

    bool isSourceValid() const override;
    int32_t getNumberOfAvailableFields() const override;
    int32_t getFieldLength() const override;

    Data getVideoField(int32_t fieldNumber,
                       int32_t startFieldLine = -1,
                       int32_t endFieldLine = -1) override;

private:
    // File handling globals
    std::ifstream inputFile;
    int64_t inputFilePos;
    bool isSourceVideoOpen;
    int32_t availableFields;
    int32_t fieldLength;
    int32_t fieldByteLength;
    int32_t fieldLineLength;

    Data outputFieldData;

    // Whole-field caching, bounded so a long capture is not held in RAM.
    FieldCache fieldCache;

    // Serialises I/O + cache access so concurrent getVideoField() calls from
    // different worker threads are race-free. The legacy
    // SourceVideo was single-threaded; this widens the contract.
    mutable std::mutex ioMutex;

    // Non-owning pointer to the VideoParameters bound via
    // bindVideoParameters(). The C ABI's chd_video handle owns the metadata.
    const chd::metadata::LdDecodeMetaData::VideoParameters *boundParameters = nullptr;
};

}  // namespace chd::reader

#endif // CHD_READER_TBC_SOURCE_H
