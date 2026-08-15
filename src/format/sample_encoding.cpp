// SPDX-License-Identifier: GPL-3.0-or-later

#include "sample_encoding.h"

#include <cstring>

namespace chd::format {

static constexpr SampleEncodingPreset PRESETS[] = {
    { SampleEncoding::CVBS_U10_4FSC,    "CVBS_U10_4FSC",    2, true,  true  },
    { SampleEncoding::CVBS_U16_4FSC,    "CVBS_U16_4FSC",    2, false, true  },
    { SampleEncoding::CVBS_TPG21_4FSC,  "CVBS_TPG21_4FSC",  2, true,  true  },
    { SampleEncoding::CVBS_S16_4FSC,    "CVBS_S16_4FSC",    2, true,  true  },
    { SampleEncoding::RAW_S16_28M,      "RAW_S16_28M",      2, true,  false },
    { SampleEncoding::RAW_S16_40M,      "RAW_S16_40M",      2, true,  false },
};

// TPG21 hardware offset (spec sample-encoding-presets.md): int16 = (val10 - 508) × 64.
static constexpr int32_t TPG21_OFFSET_10B = 508;

// Preset names that changed spelling in the spec, mapped to their current
// name. The rename was purely editorial: word format, offset and scale, and
// the YC chroma centring are all identical either side of it.
static constexpr struct { const char *legacy; SampleEncoding encoding; } LEGACY_NAMES[] = {
    // Spelled without the `4` until spec v1.4.0, which renamed it for
    // consistency with the other 4fsc presets without bumping user_version.
    { "CVBS_S16_FSC", SampleEncoding::CVBS_S16_4FSC },
};

const SampleEncodingPreset *findSampleEncodingByName(const std::string &name)
{
    for (const auto &preset : PRESETS) {
        if (name == preset.name) return &preset;
    }
    for (const auto &alias : LEGACY_NAMES) {
        if (name == alias.legacy) return &getSampleEncoding(alias.encoding);
    }
    return nullptr;
}

const SampleEncodingPreset &getSampleEncoding(SampleEncoding encoding)
{
    return PRESETS[static_cast<size_t>(encoding)];
}

uint16_t convertCompositeSampleToCanonical(SampleEncoding encoding, int16_t raw,
                                           int32_t blanking10)
{
    switch (encoding) {
        case SampleEncoding::CVBS_U10_4FSC: {
            // raw is the 10-bit value in [0..1023] stored in a signed int16
            // (with optional sub-zero / above-peak headroom). Scale to ×64
            // and clamp to uint16_t range.
            const int32_t scaled = static_cast<int32_t>(raw) * 64;
            if (scaled < 0) return 0;
            if (scaled > 65535) return 65535;
            return static_cast<uint16_t>(scaled);
        }
        case SampleEncoding::CVBS_U16_4FSC: {
            // raw is already the 10-bit value × 64 (per spec) in an unsigned
            // 16-bit container. Reinterpret as uint16_t.
            return static_cast<uint16_t>(raw);
        }
        case SampleEncoding::CVBS_TPG21_4FSC: {
            // int16 = (val10 - 508) × 64; recover val10 × 64 = int16 + 508×64.
            const int32_t centered = static_cast<int32_t>(raw) + TPG21_OFFSET_10B * 64;
            if (centered < 0) return 0;
            if (centered > 65535) return 65535;
            return static_cast<uint16_t>(centered);
        }
        case SampleEncoding::CVBS_S16_4FSC: {
            // int16 = (val10 - blanking10) × 32; recover val10 × 64 =
            // int16 × 2 + blanking10 × 64.
            const int32_t scaled = static_cast<int32_t>(raw) * 2 + blanking10 * 64;
            if (scaled < 0) return 0;
            if (scaled > 65535) return 65535;
            return static_cast<uint16_t>(scaled);
        }
        case SampleEncoding::RAW_S16_28M:
        case SampleEncoding::RAW_S16_40M: {
            // Raw captures have no defined amplitude mapping. Pass through as
            // an unsigned 16-bit interpretation; decoders will reject the
            // accompanying RAW signal state.
            return static_cast<uint16_t>(raw);
        }
    }
    return static_cast<uint16_t>(raw);
}

int16_t convertChromaSampleToCenteredCanonical(SampleEncoding encoding, int16_t raw,
                                               int32_t blanking10)
{
    // Per the spec, chroma is centred at 10-bit value 512. We return the
    // centred 10-bit value × 64 as a signed int16.
    switch (encoding) {
        case SampleEncoding::CVBS_U10_4FSC: {
            // raw is 10-bit value (signed int16 container). Subtract 512, then
            // × 64. The result fits comfortably in int16 (range roughly
            // ±32768).
            const int32_t centered = (static_cast<int32_t>(raw) - 512) * 64;
            if (centered < INT16_MIN) return INT16_MIN;
            if (centered > INT16_MAX) return INT16_MAX;
            return static_cast<int16_t>(centered);
        }
        case SampleEncoding::CVBS_U16_4FSC: {
            // raw is 10-bit × 64 in uint16 (chroma zero at 512·64 = 32768).
            // Reinterpret as uint and subtract 32768 to get signed excursion.
            const int32_t asU = static_cast<int32_t>(static_cast<uint16_t>(raw));
            const int32_t centered = asU - 32768;
            if (centered < INT16_MIN) return INT16_MIN;
            if (centered > INT16_MAX) return INT16_MAX;
            return static_cast<int16_t>(centered);
        }
        case SampleEncoding::CVBS_TPG21_4FSC: {
            // int16 = (val10 - 508) × 64. To get centred-at-512 excursion ×
            // 64, subtract (512 - 508) × 64 = 256.
            const int32_t centered = static_cast<int32_t>(raw) - (512 - TPG21_OFFSET_10B) * 64;
            if (centered < INT16_MIN) return INT16_MIN;
            if (centered > INT16_MAX) return INT16_MAX;
            return static_cast<int16_t>(centered);
        }
        case SampleEncoding::CVBS_S16_4FSC: {
            // int16 = (val10 - blanking10) × 32 with chroma centred at
            // val10 = 512; excursion × 64 = int16 × 2 + (blanking10 - 512) × 64.
            const int32_t centered =
                static_cast<int32_t>(raw) * 2 + (blanking10 - 512) * 64;
            if (centered < INT16_MIN) return INT16_MIN;
            if (centered > INT16_MAX) return INT16_MAX;
            return static_cast<int16_t>(centered);
        }
        case SampleEncoding::RAW_S16_28M:
        case SampleEncoding::RAW_S16_40M: {
            // Raw captures don't have a separate chroma file in practice;
            // pass through unchanged.
            return raw;
        }
    }
    return raw;
}

}  // namespace chd::format
