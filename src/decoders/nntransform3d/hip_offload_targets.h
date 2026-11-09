// SPDX-License-Identifier: GPL-3.0-or-later
//
// Matches the offload targets a HIP build embedded against the GPU it is about
// to run on. Kept free of HIP so the matching rules can be tested on any host.

#ifndef CHD_DECODERS_NNTRANSFORM3D_HIP_OFFLOAD_TARGETS_H
#define CHD_DECODERS_NNTRANSFORM3D_HIP_OFFLOAD_TARGETS_H

#include <cstddef>
#include <string_view>

namespace chd::decoders::nntransform3d {

enum class HipTargetMatch {
    Native,          // a code object was built for exactly this GPU
    RuntimeDecides,  // only SPIR-V or generic entries could serve it
    None,            // nothing built can run on this GPU
};

namespace detail {

// Every ":feature" named by `target` (e.g. xnack+) must also be in `device`.
// A target that names no feature runs in either mode.
inline bool targetFeaturesSatisfied(std::string_view target, std::string_view device) {
    while (true) {
        const std::size_t colon = target.find(':');
        if (colon == std::string_view::npos) return true;
        target.remove_prefix(colon + 1);
        const std::string_view feature = target.substr(0, target.find(':'));

        bool found = false;
        std::string_view rest = device;
        while (!found) {
            const std::size_t deviceColon = rest.find(':');
            if (deviceColon == std::string_view::npos) break;
            rest.remove_prefix(deviceColon + 1);
            found = rest.substr(0, rest.find(':')) == feature;
        }
        if (!found) return false;
    }
}

}  // namespace detail

// `targets` is the comma-separated -Drocm_offload_targets list; `deviceArch` is
// the GPU's gcnArchName, such as "gfx906:sramecc+:xnack-".
//
// A generic target can't be decided from names: gfx9-generic loads on a gfx900
// but the runtime rejects it on a gfx906 with sramecc on.
inline HipTargetMatch matchHipOffloadTargets(std::string_view targets,
                                             std::string_view deviceArch) {
    const std::string_view deviceBase = deviceArch.substr(0, deviceArch.find(':'));
    bool runtimeMayServe = false;

    while (!targets.empty()) {
        const std::size_t comma = targets.find(',');
        const std::string_view target = targets.substr(0, comma);
        targets = (comma == std::string_view::npos) ? std::string_view{}
                                                    : targets.substr(comma + 1);

        const std::string_view base = target.substr(0, target.find(':'));
        if (base == "amdgcnspirv" || base.find("-generic") != std::string_view::npos) {
            runtimeMayServe = true;
        } else if (base == deviceBase && detail::targetFeaturesSatisfied(target, deviceArch)) {
            return HipTargetMatch::Native;
        }
    }
    return runtimeMayServe ? HipTargetMatch::RuntimeDecides : HipTargetMatch::None;
}

}  // namespace chd::decoders::nntransform3d

#endif  // CHD_DECODERS_NNTRANSFORM3D_HIP_OFFLOAD_TARGETS_H
