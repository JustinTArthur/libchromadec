// SPDX-License-Identifier: GPL-3.0-or-later

#include "nntransform3d_window.h"

#include <algorithm>
#include <cmath>
#include <mutex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace chd::decoders::nntransform3d {

namespace {
SineWindows  g_windows;
std::once_flag g_initOnce;
}  // namespace

const SineWindows &getSineWindows()
{
    std::call_once(g_initOnce, []() {
        for (int i = 0; i < kNx; i++) {
            g_windows.x[i] = std::sin(M_PI * (static_cast<double>(i) + 0.5) /
                                      static_cast<double>(kNx));
        }
        for (int i = 0; i < kNy; i++) {
            g_windows.y[i] = std::sin(M_PI * (static_cast<double>(i) + 0.5) /
                                      static_cast<double>(kNy));
        }
        for (int i = 0; i < kNt; i++) {
            g_windows.t[i] = std::sin(M_PI * (static_cast<double>(i) + 0.5) /
                                      static_cast<double>(kNt));
        }
    });
    return g_windows;
}

namespace {

// Smallest lattice position whose tile still reaches `first`.
int32_t firstTile(int32_t anchor, int32_t step, int32_t size, int32_t first)
{
    const int32_t distance = (first - (size - 1)) - anchor;
    const int32_t steps = (distance >= 0) ? (distance + step - 1) / step : -((-distance) / step);
    return anchor + (steps * step);
}

}  // namespace

TileGrid makeTileGrid(const chd::metadata::LdDecodeMetaData::VideoParameters &vp)
{
    // asdfqazsnbb's harnesses tile NTSC frame lines 40..524 from line 32.
    constexpr int32_t kAreaFirstLine = 40;
    constexpr int32_t kAreaLastLine  = 524;
    constexpr int32_t kLineAnchor    = 32;

    const int32_t lastStoredLine = (vp.fieldHeight * 2) - 2;
    const int32_t pictureStart = (vp.pictureStart >= 0) ? vp.pictureStart : vp.activeVideoStart;

    TileGrid grid;
    grid.firstLine   = std::min(kAreaFirstLine, vp.firstActiveFrameLine);
    grid.lastLine    = std::min(std::max(kAreaLastLine, vp.lastActiveFrameLine), lastStoredLine);
    grid.firstSample = (vp.tileReadStart >= 0) ? vp.tileReadStart : vp.activeVideoStart;
    grid.endSample   = (vp.tileReadEnd >= 0) ? vp.tileReadEnd : vp.activeVideoEnd;
    grid.startY = firstTile(kLineAnchor, kStepY, kNy, grid.firstLine);
    grid.startX = firstTile(pictureStart - (kNx / 2), kStepX, kNx, grid.firstSample);
    return grid;
}

}  // namespace chd::decoders::nntransform3d
