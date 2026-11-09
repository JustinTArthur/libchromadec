// SPDX-License-Identifier: GPL-3.0-or-later

#include "nntransform3d_fft_cpu.h"

#include <mutex>

#include "../../common/fftw_planner.h"

namespace chd::decoders::nntransform3d {

const CpuPlans &getCpuPlans()
{
    static const CpuPlans plans = [] {
        // Allocate scratch buffers just to construct the plans, then free
        // them. The plans don't keep these buffers, but they do require the
        // same alignment from the ones they later execute on.
        auto *planIn  = reinterpret_cast<fftw_complex *>(
            fftw_malloc(sizeof(fftw_complex) * kNt * kNy * kNx));
        auto *planOut = reinterpret_cast<fftw_complex *>(
            fftw_malloc(sizeof(fftw_complex) * kNt * kNy * kNx));
        CpuPlans created { nullptr, nullptr };
        {
            std::lock_guard<std::mutex> lock(chd::fftwPlannerMutex());
            created.forward = fftw_plan_dft_3d(kNt, kNy, kNx, planIn, planOut,
                                               FFTW_FORWARD, FFTW_ESTIMATE);
            created.inverse = fftw_plan_dft_3d(kNt, kNy, kNx, planOut, planIn,
                                               FFTW_BACKWARD, FFTW_ESTIMATE);
        }
        fftw_free(planIn);
        fftw_free(planOut);
        return created;
    }();
    return plans;
}

}  // namespace chd::decoders::nntransform3d
