// SPDX-License-Identifier: GPL-3.0-or-later
/************************************************************************

    ntscdecoder.h

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

#ifndef CHD_DECODERS_COMB_NTSC_DECODER_H
#define CHD_DECODERS_COMB_NTSC_DECODER_H

#include <cstdint>
#include <memory>
#include <vector>

#include "../../metadata/core.h"
#include "../../output/component_frame.h"
#include "../../reader/tbc_source.h"

#include "../decoder_base.h"
#include "../source_field.h"
#include "comb.h"

#if defined(CHD_WITH_NN)
namespace chd::nn { class InferenceEngine; }
#endif

namespace chd::decoders::comb {

// 2D/3D NTSC decoder using Comb
class NtscDecoder : public chd::decoders::Decoder {
public:
    NtscDecoder(const Comb::Configuration &combConfig);
    bool configure(const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters) override;
    int32_t getLookBehind() const override;
    int32_t getLookAhead() const override;

#if defined(CHD_WITH_NN)
    // Bind the nnTransform3D inference engine. Caller must also set
    // config.combConfig.nnTransform3D = true (and dimensions = 3) for the
    // engine to actually be used. Passing nullptr unbinds. This is the
    // public C++ entry point that the C ABI's chd_decoder_set_nn_model
    // will dispatch to later; this only exposes the C++ surface.
    void setNnModel(std::shared_ptr<chd::nn::InferenceEngine> engine);
    void setTilePassCache(std::shared_ptr<Comb::TilePassCache> cache);
#endif

    void decodeFrames(const std::vector<chd::decoders::SourceField> &inputFields,
                      int32_t startIndex, int32_t endIndex,
                      std::vector<chd::output::ComponentFrame> &componentFrames) override;

    // Parameters used by NtscDecoder
    struct Configuration : public chd::decoders::Decoder::Configuration {
        Comb::Configuration combConfig;
    };

private:
    Configuration config;
    Comb comb;
};

}  // namespace chd::decoders::comb

#endif  // CHD_DECODERS_COMB_NTSC_DECODER_H
