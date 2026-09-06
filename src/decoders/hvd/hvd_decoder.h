// SPDX-License-Identifier: GPL-3.0-or-later
//
// Holographic-variational (HVD) Y/C separation over the bundled hvd-core
// engine (vrunk11's hvd-cvbs-decoding): per-field burst lock-in, holographic
// sideband reconstruction of the chroma phasor, then Y/C arbitration by
// IRLS + conjugate gradient over the woven frame.
//
// The engine is called directly with IRE field planes built from the source
// metadata. hvd-core's own frame_bridge is deliberately bypassed: it assumes
// decode-orc buffer layouts, re-derives field order by measuring the signal,
// and mutates the process-global OpenMP thread count. The readers already
// know the levels, geometry, and field order.

#ifndef CHD_DECODERS_HVD_HVD_DECODER_H
#define CHD_DECODERS_HVD_HVD_DECODER_H

#include <cstdint>
#include <memory>

#include "engine/hvd_config.h"
#include "engine/ntsc_geometry.h"

#include "../decoder_base.h"

namespace hvd { class HvdEngine; }

namespace chd::decoders::hvd {

class HvdDecoder : public Decoder {
public:
    struct HvdConfiguration : public Decoder::Configuration {
        double chromaGain = 1.0;
        double chromaPhase = 0.0;
        // CG budget across the IRLS passes. 0 stops at the holographic
        // init; -1 keeps the engine's default.
        int32_t cgIterations = -1;
        // CHD_DEC_HVD_3D: run the field-granularity sequence pipeline
        // instead of the woven-frame path, with chunk_overlap frames of
        // temporal context each side.
        bool temporal3d = false;
        // Cross-field data-term weight; 0 adapts to the window's measured
        // Y/C ambiguity.
        double temporalStrength = 0.0;
    };

    explicit HvdDecoder(const HvdConfiguration &hvdConfig);
    ~HvdDecoder() override;

    HvdDecoder(const HvdDecoder &) = delete;
    HvdDecoder &operator=(const HvdDecoder &) = delete;

    bool configure(const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters) override;

    // The 3D pipeline needs its temporal context in the window.
    int32_t getLookBehind() const override;
    int32_t getLookAhead() const override;

    void decodeFrames(const std::vector<SourceField> &inputFields,
                      int32_t startIndex, int32_t endIndex,
                      std::vector<chd::output::ComponentFrame> &componentFrames) override;

private:
    void decodeFrame(const SourceField &firstField, const SourceField &secondField,
                     chd::output::ComponentFrame &componentFrame);
    void decodeFrames3d(const std::vector<SourceField> &inputFields,
                        int32_t startIndex, int32_t endIndex,
                        std::vector<chd::output::ComponentFrame> &componentFrames);

    HvdConfiguration config;
    ::hvd::HvdConfig engineConfig;
    ::hvd::FieldGeometry geometry;
    std::unique_ptr<::hvd::HvdEngine> engine;
};

}  // namespace chd::decoders::hvd

#endif  // CHD_DECODERS_HVD_HVD_DECODER_H
