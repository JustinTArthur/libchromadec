// SPDX-License-Identifier: GPL-3.0-or-later
//
// FFTW's planner is process-global and only fftw_execute* may run
// concurrently, so every plan and destroy in the library has to be serialised.
// This drives those sites from many threads at once, forgetting wisdom between
// rounds so each round really plans. A planner race shows up here as a crash or
// a null plan.

#include <fftw3.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "../../src/decoders/palcolour/transform_pal_2d.h"
#include "../../src/decoders/palcolour/transform_pal_3d.h"
#include "../../src/decoders/secam/secam_decoder.h"
#include "../../src/metadata/core.h"
#if defined(CHD_WITH_NN)
#include "../../src/decoders/nntransform3d/nntransform3d_fft_cpu.h"
#endif

namespace {

#define REQUIRE(cond)                                                            \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #cond \
                      << "\n";                                                   \
            return 1;                                                            \
        }                                                                        \
    } while (0)

constexpr int32_t kRounds = 40;
constexpr int32_t kThreadsPerRound = 16;

chd::metadata::LdDecodeMetaData::VideoParameters secamParams() {
    chd::metadata::LdDecodeMetaData::VideoParameters vp;
    vp.system = chd::metadata::SECAM;
    vp.fieldWidth = 1135;
    vp.fieldHeight = 313;
    vp.sampleRate = 17734475.0;
    vp.isValid = true;
    return vp;
}

// Runs body on kThreadsPerRound new threads released together.
template <typename Body>
void runTogether(Body body) {
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreadsPerRound);
    for (int32_t t = 0; t < kThreadsPerRound; t++) {
        threads.emplace_back([&go, &body, t]() {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            body(t);
        });
    }
    go.store(true, std::memory_order_release);
    for (auto &thread : threads) thread.join();
}

int testCommitTimeSites() {
    using chd::decoders::secam::SecamDecoder;
    const auto vp = secamParams();
    std::atomic<int32_t> failures{0};
    for (int32_t round = 0; round < kRounds; round++) {
        runTogether([&](int32_t t) {
            if (t % 3 == 0) {
                auto pal2d = std::make_unique<chd::decoders::palcolour::TransformPal2D>();
            } else if (t % 3 == 1) {
                auto pal3d = std::make_unique<chd::decoders::palcolour::TransformPal3D>();
            } else {
                SecamDecoder::SecamConfiguration config;
                SecamDecoder secam(config);
                if (!secam.configure(vp)) failures.fetch_add(1);
            }
        });
        fftw_forget_wisdom();
    }
    REQUIRE(failures.load() == 0);
    return 0;
}

#if defined(CHD_WITH_NN)
int testNnTransform3DPlans() {
    using namespace chd::decoders::nntransform3d;
    constexpr int32_t kPoints = kNt * kNy * kNx;
    FftwAlignedDoubles tileStorage(2 * kPoints);
    REQUIRE(fftw_alignment_of(tileStorage.data()) == 0);
    std::atomic<int32_t> failures{0};
    const fftw_plan sharedForward = getCpuPlans().forward;
    for (int32_t round = 0; round < kRounds; round++) {
        runTogether([&](int32_t t) {
            const auto &plans = getCpuPlans();
            // A plan per thread would be lost when the thread exits.
            if (plans.forward != sharedForward || plans.inverse == nullptr) {
                failures.fetch_add(1);
                return;
            }
            // Each thread round-trips its own data through the plans.
            fftw_complex *in = fftw_alloc_complex(kPoints);
            fftw_complex *spectrum = fftw_alloc_complex(kPoints);
            std::vector<double> expected(kPoints);
            for (int32_t i = 0; i < kPoints; i++) {
                expected[i] = t + std::sin(i * 0.1);
                in[i][0] = expected[i];
                in[i][1] = 0.0;
            }
            fftw_execute_dft(plans.forward, in, spectrum);
            fftw_execute_dft(plans.inverse, spectrum, in);
            for (int32_t i = 0; i < kPoints; i++) {
                if (std::abs(in[i][0] / kPoints - expected[i]) > 1e-9) {
                    failures.fetch_add(1);
                    break;
                }
            }
            fftw_free(in);
            fftw_free(spectrum);
        });
    }
    REQUIRE(failures.load() == 0);
    return 0;
}
#endif

}  // namespace

int main() {
    if (testCommitTimeSites() != 0) return 1;
#if defined(CHD_WITH_NN)
    if (testNnTransform3DPlans() != 0) return 1;
#endif
    return 0;
}
