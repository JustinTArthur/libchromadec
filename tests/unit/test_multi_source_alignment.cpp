// SPDX-License-Identifier: GPL-3.0-or-later
//
// MultiSourceAlignment unit test. Builds in-memory LdDecodeMetaData for a
// primary capture and two extra captures of the same disc — one a CAV
// LaserDisc whose VBI picture numbers start at a different disc frame, and one
// with no VBI codes at all (e.g. a CVBS capture) — then asserts that
// sourceFrameForPrimaryFrame registers them by disc frame number where VBI is
// present and falls back to positional alignment where it is not.

#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "../../src/dropout/multi_source_alignment.h"
#include "../../src/metadata/core.h"

namespace {

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        return 1; \
    } \
} while (0)

using chd::metadata::LdDecodeMetaData;

// Encode a decimal value as binary-coded decimal (one nibble per digit), the
// layout VbiDecoder::decodeBCD expects when reading a CAV picture number.
uint32_t toBcd(int32_t value) {
    uint32_t bcd = 0;
    int32_t shift = 0;
    while (value > 0) {
        bcd |= static_cast<uint32_t>(value % 10) << shift;
        value /= 10;
        shift += 4;
    }
    return bcd;
}

// Build metadata for `frameCount` still-frames (two fields each). If
// `cavStart >= 0` every frame carries a CAV picture number on VBI line 17
// (the 0xF8.... lead nibble per IEC 60857-1986 10.1.3), counting up from
// `cavStart`; otherwise the fields carry no VBI codes.
std::unique_ptr<LdDecodeMetaData> makeSource(int32_t frameCount, int32_t cavStart) {
    auto meta = std::make_unique<LdDecodeMetaData>();
    meta->setIsFirstFieldFirst(true);

    LdDecodeMetaData::VideoParameters vp;
    vp.numberOfSequentialFields = frameCount * 2;
    meta->setVideoParameters(vp);

    for (int32_t f = 0; f < frameCount; ++f) {
        LdDecodeMetaData::Field first;
        first.isFirstField = true;
        LdDecodeMetaData::Field second;
        second.isFirstField = false;

        if (cavStart >= 0) {
            const int32_t picNo = cavStart + f;
            const int32_t vbi17 = 0xF00000 | static_cast<int32_t>(toBcd(picNo));
            first.vbi.vbiData = {0, vbi17, 0};
            second.vbi.vbiData = {0, vbi17, 0};
        }

        meta->appendField(first);
        meta->appendField(second);
    }
    return meta;
}

}  // namespace

int main() {
    // Primary capture covers disc picture numbers 100..109. Extra A is a CAV
    // capture of the same disc that started five frames later (105..114), so
    // its sequential frame 1 holds disc frame 105. Extra B carries no VBI.
    auto primary = makeSource(10, 100);
    auto extraA  = makeSource(10, 105);
    auto extraB  = makeSource(10, -1);

    using Source = chd::dropout::MultiSourceAlignment::Source;
    std::vector<Source> sources{
        {primary.get(), true}, {extraA.get(), true}, {extraB.get(), true}};
    chd::dropout::MultiSourceAlignment align(std::move(sources));

    // hasVbi reflects which captures carry usable codes.
    REQUIRE(align.hasVbi(0));
    REQUIRE(align.hasVbi(1));
    REQUIRE(!align.hasVbi(2));

    // The primary's sequential frame 6 is disc picture number 105.
    REQUIRE(align.convertSequentialFrameNumberToVbi(6, 0) == 105);

    // Disc frame 105 is extra A's sequential frame 1 — VBI registration must
    // pick that up rather than naively returning frame 6.
    REQUIRE(align.sourceFrameForPrimaryFrame(6, 1) == 1);

    // The primary's frame 1 is disc 100, before extra A's range, so extra A
    // does not cover it.
    REQUIRE(align.sourceFrameForPrimaryFrame(1, 1) == -1);

    // Source 0 (the primary) always maps to itself.
    REQUIRE(align.sourceFrameForPrimaryFrame(6, 0) == 6);

    // Extra B has no VBI codes, so it falls back to positional alignment
    // (the same sequential frame number).
    REQUIRE(align.sourceFrameForPrimaryFrame(6, 2) == 6);

    // VBI <-> sequential conversion round-trips across extra A's range.
    REQUIRE(align.convertVbiFrameNumberToSequential(105, 1) == 1);
    REQUIRE(align.convertVbiFrameNumberToSequential(114, 1) == 10);

    // Disc frame 105 is covered by the primary and extra A, but not the
    // VBI-less extra B.
    const std::vector<int32_t> avail = align.getAvailableSourcesForFrame(105);
    REQUIRE(avail.size() == 2);
    REQUIRE(avail[0] == 0);
    REQUIRE(avail[1] == 1);

    // Pairing order is the caller's, not the metadata's. Reversed (second
    // field opens each frame), field 1 pairs with nothing, so sequential frame
    // 1 is fields (3, 2) and reads its picture number from first field 3:
    // disc frame 101. The metadata's own flag stays first-field-first.
    {
        std::vector<Source> reversed{{primary.get(), false}, {extraA.get(), false}};
        chd::dropout::MultiSourceAlignment rev(std::move(reversed));
        REQUIRE(primary->getIsFirstFieldFirst());
        REQUIRE(primary->getNumberOfFrames(false) == 9);
        REQUIRE(primary->getFirstFieldNumber(1, false) == 3);
        REQUIRE(primary->getSecondFieldNumber(1, false) == 2);
        REQUIRE(rev.convertSequentialFrameNumberToVbi(1, 0) == 101);
        REQUIRE(rev.convertSequentialFrameNumberToVbi(1, 1) == 106);
        // Disc 106 is the reversed primary's frame 6 and extra A's frame 1;
        // disc 105 (the reversed primary's frame 5) is before extra A's range.
        REQUIRE(rev.sourceFrameForPrimaryFrame(6, 1) == 1);
        REQUIRE(rev.sourceFrameForPrimaryFrame(5, 1) == -1);
    }

    std::cout << "test_multi_source_alignment: all assertions passed\n";
    return 0;
}