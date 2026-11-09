// SPDX-License-Identifier: GPL-3.0-or-later
//
// Matching of built HIP offload targets against a GPU's gcnArchName. A wrong
// answer of "Native" lets the first kernel launch crash the process, so the
// cases that must not match matter as much as the ones that must.

#include "../../src/decoders/nntransform3d/hip_offload_targets.h"

#include <iostream>

namespace {

using chd::decoders::nntransform3d::HipTargetMatch;
using chd::decoders::nntransform3d::matchHipOffloadTargets;

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        return 1; \
    } \
} while (0)

const char *const kVega20 = "gfx906:sramecc+:xnack-";
const char *const kVega10 = "gfx900:xnack-";
const char *const kMi350x = "gfx950:sramecc+:xnack-";

}  // namespace

int main() {
    // The issue's failing build: only the clang default target embedded.
    REQUIRE(matchHipOffloadTargets("gfx906", kMi350x) == HipTargetMatch::None);
    REQUIRE(matchHipOffloadTargets("gfx906", kVega20) == HipTargetMatch::Native);

    // Any entry in a list can match, and a name must match in full.
    REQUIRE(matchHipOffloadTargets("gfx906,gfx950", kMi350x) == HipTargetMatch::Native);
    REQUIRE(matchHipOffloadTargets("gfx95", kMi350x) == HipTargetMatch::None);
    REQUIRE(matchHipOffloadTargets("gfx9500", kMi350x) == HipTargetMatch::None);

    // Features a target names have to agree with the device.
    REQUIRE(matchHipOffloadTargets("gfx906:xnack-", kVega20) == HipTargetMatch::Native);
    REQUIRE(matchHipOffloadTargets("gfx906:sramecc+:xnack-", kVega20) == HipTargetMatch::Native);
    REQUIRE(matchHipOffloadTargets("gfx906:xnack+", kVega20) == HipTargetMatch::None);
    REQUIRE(matchHipOffloadTargets("gfx900:sramecc+", kVega10) == HipTargetMatch::None);
    REQUIRE(matchHipOffloadTargets("gfx906:xnack-:sramecc+", kVega20) == HipTargetMatch::Native);
    REQUIRE(matchHipOffloadTargets("gfx906:xnack-", "gfx906") == HipTargetMatch::None);

    // SPIR-V and generic targets can't be judged by name.
    REQUIRE(matchHipOffloadTargets("amdgcnspirv", kVega20) == HipTargetMatch::RuntimeDecides);
    REQUIRE(matchHipOffloadTargets("gfx9-generic", kVega10) == HipTargetMatch::RuntimeDecides);
    REQUIRE(matchHipOffloadTargets("gfx950,gfx9-generic", kVega20) == HipTargetMatch::RuntimeDecides);
    REQUIRE(matchHipOffloadTargets("gfx10-3-generic", kVega20) == HipTargetMatch::RuntimeDecides);
    REQUIRE(matchHipOffloadTargets("gfx906:xnack+,gfx9-generic", kVega20) == HipTargetMatch::RuntimeDecides);

    // A native match wins over an entry the runtime would have to judge.
    REQUIRE(matchHipOffloadTargets("amdgcnspirv,gfx906", kVega20) == HipTargetMatch::Native);

    REQUIRE(matchHipOffloadTargets("", kVega20) == HipTargetMatch::None);
    REQUIRE(matchHipOffloadTargets(",gfx906,", kVega20) == HipTargetMatch::Native);
    return 0;
}
