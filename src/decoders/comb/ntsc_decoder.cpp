// SPDX-License-Identifier: GPL-3.0-or-later
/************************************************************************

    ntscdecoder.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018 Chad Page
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2019-2021 Adam Sampson

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

#include "ntsc_decoder.h"

#include "../../common/log.h"

namespace chd::decoders::comb {

NtscDecoder::NtscDecoder(const Comb::Configuration &combConfig)
{
    config.combConfig = combConfig;
}

bool NtscDecoder::configure(const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters) {
    // Ensure the source video is NTSC
    if (videoParameters.system != chd::metadata::NTSC) {
        chd::log::fail() << "This decoder is for NTSC video sources only";
        return false;
    }

    config.videoParameters = videoParameters;

    // Configure the Comb instance now that we know the video parameters. It
    // rejects geometry its fixed-size frame buffers cannot hold, and reports
    // the reason itself.
    return comb.updateConfiguration(config.videoParameters, config.combConfig);
}

int32_t NtscDecoder::getLookBehind() const
{
    return config.combConfig.getLookBehind();
}

int32_t NtscDecoder::getLookAhead() const
{
    return config.combConfig.getLookAhead();
}

void NtscDecoder::decodeFrames(const std::vector<chd::decoders::SourceField> &inputFields,
                               int32_t startIndex, int32_t endIndex,
                               std::vector<chd::output::ComponentFrame> &componentFrames)
{
    // Decode fields to frames
    comb.decodeFrames(inputFields, startIndex, endIndex, componentFrames);
}

#if defined(CHD_WITH_NN)
void NtscDecoder::setNnModel(std::shared_ptr<chd::nn::InferenceEngine> engine)
{
    comb.setNnModel(std::move(engine));
}

void NtscDecoder::setTilePassCache(std::shared_ptr<Comb::TilePassCache> cache)
{
    comb.setTilePassCache(std::move(cache));
}
#endif

}  // namespace chd::decoders::comb
