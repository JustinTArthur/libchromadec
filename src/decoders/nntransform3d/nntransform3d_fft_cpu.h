// SPDX-License-Identifier: GPL-3.0-or-later
//
// Shared FFTW plans for the nnTransform3D CPU path. The plans operate on
// Nt=4, Ny=16, Nx=16 complex tiles, matching the chroma_net model's expected
// input shape. Worker threads execute the same plans in parallel through
// fftw_execute_dft on their own buffers, which FFTW documents as thread-safe.
//
// This is the FFT primitive for the CPU path only. The CUDA and HIP paths
// carry their own device-resident cuFFT/hipFFT inside their pipelines.

#ifndef CHD_DECODERS_NNTRANSFORM3D_FFT_CPU_H
#define CHD_DECODERS_NNTRANSFORM3D_FFT_CPU_H

#include <fftw3.h>

#include <cstddef>
#include <new>
#include <vector>

namespace chd::decoders::nntransform3d {

// Block dimensions matching the chroma_net model trained by asdfqazsnbb.
// Changing these requires a different model.
inline constexpr int kNt = 4;   // temporal block
inline constexpr int kNy = 16;  // vertical block
inline constexpr int kNx = 16;  // horizontal block

// Overlap-add step follows asdfqazsnbb's harness: half the block size, so each output
// sample is touched by ~4 tiles on the (X,Y) plane.
inline constexpr int kStepX = 8;
inline constexpr int kStepY = 8;

// The first call in the process creates both plans (forward + inverse) sized
// for Nt*Ny*Nx complex points; later calls return the same ones. Plans are
// intentionally never destroyed; they live until process exit.
struct CpuPlans {
    fftw_plan forward;
    fftw_plan inverse;
};
const CpuPlans &getCpuPlans();

// fftw_execute_dft needs buffers aligned like the ones the plans were made on,
// which came from fftw_malloc. Tile buffers passed to it must use this storage.
template <typename T>
struct FftwAllocator {
    using value_type = T;

    FftwAllocator() = default;
    template <typename U>
    FftwAllocator(const FftwAllocator<U> &) {}

    T *allocate(std::size_t n)
    {
        void *p = fftw_malloc(n * sizeof(T));
        if (p == nullptr) throw std::bad_alloc();
        return static_cast<T *>(p);
    }
    void deallocate(T *p, std::size_t) { fftw_free(p); }

    template <typename U>
    bool operator==(const FftwAllocator<U> &) const { return true; }
    template <typename U>
    bool operator!=(const FftwAllocator<U> &) const { return false; }
};

// Interleaved real/imaginary doubles, reinterpreted as fftw_complex by callers.
using FftwAlignedDoubles = std::vector<double, FftwAllocator<double>>;

}  // namespace chd::decoders::nntransform3d

#endif  // CHD_DECODERS_NNTRANSFORM3D_FFT_CPU_H
