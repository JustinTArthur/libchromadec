// SPDX-License-Identifier: GPL-3.0-or-later
//
// FFTW's planner state is process-global and only fftw_execute* is thread-safe,
// so every fftw_plan_* and fftw_destroy_plan in the library must hold this
// mutex. It can't protect against other code in the process calling the same
// FFTW library's planner.

#ifndef CHD_COMMON_FFTW_PLANNER_H
#define CHD_COMMON_FFTW_PLANNER_H

#include <fftw3.h>

#include <mutex>

namespace chd {

// Never destroyed, so a decoder freed during static destruction can still lock
// it.
inline std::mutex &fftwPlannerMutex()
{
    static std::mutex &mutex = *new std::mutex;
    return mutex;
}

// Either plan may be null.
inline void destroyFftwPlans(fftw_plan forward, fftw_plan inverse)
{
    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
    if (forward != nullptr) fftw_destroy_plan(forward);
    if (inverse != nullptr) fftw_destroy_plan(inverse);
}

}  // namespace chd

#endif  // CHD_COMMON_FFTW_PLANNER_H
