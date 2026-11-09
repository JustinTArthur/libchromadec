// SPDX-License-Identifier: GPL-3.0-or-later
//
// HIP + hipFFT + ORT-IoBinding pipeline for nnTransform3D. AMD ROCm /
// MIGraphX counterpart of the CUDA cuFFT pipeline; the kernel bodies
// and host orchestration are a deliberate near-textual mirror of
// nntransform3d_pipeline_cuda.cu because hipFFT tracks the cuFFT API
// (literal s/cufft/hipfft/g) and HIP kernel syntax matches CUDA's
// `__global__` + `<<<>>>` form.
//
// Algorithm authorship: asdfqazsnbb (originally Discord-distributed
// nnTransform3D, v2 later open-sourced as a standalone C++/CUDA harness).
// HIP port: by inspection from that harness; the original author did
// not publish a HIP variant.
//
// Build note: this file uses HIP-specific syntax (`__global__`,
// `<<<grid, block>>>`) and must be compiled by hipcc, not the host
// C++ compiler. Meson's `custom_target` in src/meson.build pipes this
// file to hipcc when `with_rocm` is enabled.

#include "nntransform3d_pipeline_hip.h"

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <string>
#include <vector>

#include "hip_offload_targets.h"
#include "nntransform3d_fft_cpu.h"
#include "nntransform3d_window.h"
#include "../../common/log.h"
#include "../../nn/ort_session.h"

namespace chd::decoders::nntransform3d {

namespace {

#define CHD_HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        chd::log::warn() << "nnTransform3D[HIP]: " #expr " failed:" \
                         << hipGetErrorString(_err); \
        return false; \
    } \
} while (0)

#define CHD_HIPFFT_CHECK(expr) do { \
    hipfftResult _err = (expr); \
    if (_err != HIPFFT_SUCCESS) { \
        chd::log::warn() << "nnTransform3D[HIP]: " #expr " failed with hipFFT code" \
                         << static_cast<int>(_err); \
        return false; \
    } \
} while (0)

// src/meson.build passes the list unquoted; see the note there.
#define CHD_HIP_QUOTE_ARGS(...) #__VA_ARGS__
#define CHD_HIP_QUOTE(...) CHD_HIP_QUOTE_ARGS(__VA_ARGS__)
constexpr const char *kOffloadTargets = CHD_HIP_QUOTE(CHD_HIP_OFFLOAD_TARGETS);

void freeOrWarn(void *deviceMemory) {
    if (deviceMemory == nullptr) return;
    const hipError_t err = hipFree(deviceMemory);
    if (err != hipSuccess) {
        chd::log::warn() << "nnTransform3D[HIP]: hipFree failed:" << hipGetErrorString(err);
    }
}

// ─── Device kernels ────────────────────────────────────────────────────────
// Same algorithm as the CUDA path; only the namespace prefix on the FFT
// scalar type changes.

__global__ void calcDCKernel(const uint16_t *dCvbsF0, const uint16_t *dCvbsF1,
                             bool padF0, bool padF1,
                             const int *dLedgerY, const int *dLedgerX,
                             double *dLedgerDc,
                             int numBlocks,
                             int activeStartX, int activeEndX,
                             int firstActiveY, int lastActiveY,
                             int frameStride)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= numBlocks) return;

    const int y = dLedgerY[b];
    const int x = dLedgerX[b];
    double blockDc = 0.0;
    int pixelCount = 0;

    for (int t = 0; t < 4; ++t) {
        if ((t < 2) ? padF0 : padF1) continue;
        const uint16_t *cvbs = (t < 2) ? dCvbsF0 : dCvbsF1;
        const bool isOddField = (t % 2 != 0);

        for (int dy = 0; dy < 16; ++dy) {
            const int absY = y + dy;
            if (absY < firstActiveY || absY > lastActiveY) continue;
            if ((absY % 2 != 0) != isOddField) continue;
            for (int dx = 0; dx < 16; ++dx) {
                const int absX = x + dx;
                if (absX >= activeStartX && absX < activeEndX) {
                    blockDc += static_cast<double>(cvbs[absY * frameStride + absX]);
                    ++pixelCount;
                }
            }
        }
    }
    dLedgerDc[b] = (pixelCount > 0) ? (blockDc / pixelCount) : 0.0;
}

__global__ void packAndWindowKernel(const uint16_t *dCvbsF0, const uint16_t *dCvbsF1,
                                    bool padF0, bool padF1,
                                    hipfftDoubleComplex *dInBatch,
                                    const int *dLedgerY, const int *dLedgerX,
                                    const double *dLedgerDc,
                                    const double *dWinT, const double *dWinY, const double *dWinX,
                                    int numBlocks, int blockSize,
                                    int activeStartX, int activeEndX,
                                    int firstActiveY, int lastActiveY,
                                    int frameStride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBlocks * blockSize) return;

    const int b = idx / blockSize;
    const int i = idx % blockSize;
    const int t  = i / 256;
    const int rem = i % 256;
    const int dy = rem / 16;
    const int dx = rem % 16;

    const int absY = dLedgerY[b] + dy;
    const int absX = dLedgerX[b] + dx;
    const bool isOddField = (t % 2 != 0);
    const bool isPad = (t < 2) ? padF0 : padF1;

    if (isPad
        || absY < firstActiveY || absY > lastActiveY
        || absX < activeStartX || absX >= activeEndX
        || ((absY % 2 != 0) != isOddField)) {
        dInBatch[idx].x = 0.0;
    } else {
        const uint16_t *cvbs = (t < 2) ? dCvbsF0 : dCvbsF1;
        const double val = static_cast<double>(cvbs[absY * frameStride + absX]);
        dInBatch[idx].x = (val - dLedgerDc[b]) * dWinT[t] * dWinY[dy] * dWinX[dx];
    }
    dInBatch[idx].y = 0.0;
}

__global__ void calcMagnitudeKernel(const hipfftDoubleComplex *dOutBatch,
                                    float *dTrtInput,
                                    int numBlocks, int blockSize,
                                    double inputMagnitudeScale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBlocks * blockSize) return;

    const int b = idx / blockSize;
    const int i = idx % blockSize;

    const int Nx = 16, Ny = 16;
    const int t = i / (Ny * Nx);
    const int rem = i % (Ny * Nx);
    const int y = rem / Nx;
    const int x = rem % Nx;

    const double re = dOutBatch[idx].x;
    const double im = dOutBatch[idx].y;
    const double mag = sqrt(re * re + im * im);

    int refT = (2 - t) % 4;       if (refT < 0) refT += 4;
    int refY = (16 - y) % 16;
    int refX = (8 - x) % 16;      if (refX < 0) refX += 16;
    const int idxRef = b * blockSize + (refT * Ny * Nx + refY * Nx + refX);
    const double reRef = dOutBatch[idxRef].x;
    const double imRef = dOutBatch[idxRef].y;
    const double magRef = sqrt(reRef * reRef + imRef * imRef);

    const int out0 = b * (2 * blockSize) + 0 * blockSize + i;
    const int out1 = b * (2 * blockSize) + 1 * blockSize + i;
    const double invScale = 1.0 / inputMagnitudeScale;
    dTrtInput[out0] = static_cast<float>(mag    * invScale);
    dTrtInput[out1] = static_cast<float>(magRef * invScale);
}

__global__ void applyMaskKernel(hipfftDoubleComplex *dOutBatch,
                                const float *dMask,
                                int totalElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;
    const float gain = dMask[idx];
    dOutBatch[idx].x *= gain;
    dOutBatch[idx].y *= gain;
}

__global__ void olaKernel(const hipfftDoubleComplex *dInBatch,
                          double *dAccChromaF0, double *dAccChromaF1,
                          double *dWeightSumF0, double *dWeightSumF1,
                          bool padF0, bool padF1,
                          const int *dLedgerY, const int *dLedgerX,
                          const double *dWinT, const double *dWinY, const double *dWinX,
                          int numBlocks, int blockSize,
                          int activeStartX, int activeEndX,
                          int firstActiveY, int lastActiveY,
                          int frameStride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBlocks * blockSize) return;

    const int b = idx / blockSize;
    const int i = idx % blockSize;
    const int t  = i / 256;
    const int rem = i % 256;
    const int dy = rem / 16;
    const int dx = rem % 16;

    if ((t < 2) ? padF0 : padF1) return;

    const int absY = dLedgerY[b] + dy;
    const int absX = dLedgerX[b] + dx;
    if (absY < firstActiveY || absY > lastActiveY) return;
    if ((absY % 2 != 0) != (t % 2 != 0)) return;
    if (absX < activeStartX || absX >= activeEndX) return;

    const double val = dInBatch[idx].x / static_cast<double>(blockSize);
    const double w   = dWinT[t] * dWinY[dy] * dWinX[dx];
    const int    frameIdx = absY * frameStride + absX;

    double *acc    = (t < 2) ? dAccChromaF0  : dAccChromaF1;
    double *weight = (t < 2) ? dWeightSumF0  : dWeightSumF1;
    atomicAdd(&acc[frameIdx],    val * w);
    atomicAdd(&weight[frameIdx], w   * w);
}

// ─── Thread-local persistent state ─────────────────────────────────────────

struct HipThreadCache {
    hipfftHandle pFwd = 0;
    hipfftHandle pInv = 0;
    int cachedNumBlocks = 0;

    hipfftDoubleComplex *dInBatch  = nullptr;
    hipfftDoubleComplex *dOutBatch = nullptr;
    float               *dTrtInput = nullptr;
    float               *dMask     = nullptr;

    int    *dLedgerY  = nullptr;
    int    *dLedgerX  = nullptr;
    double *dLedgerDc = nullptr;

    double *dWinT = nullptr;
    double *dWinY = nullptr;
    double *dWinX = nullptr;

    // HIP device the plans and buffers were created on; -1 before first use.
    int device = -1;
    Ort::MemoryInfo memInfo{"Hip", OrtDeviceAllocator, 0, OrtMemTypeDefault};

    void releaseBatch() {
        if (pFwd)      { hipfftDestroy(pFwd);  pFwd = 0; }
        if (pInv)      { hipfftDestroy(pInv);  pInv = 0; }
        freeOrWarn(dInBatch);  dInBatch  = nullptr;
        freeOrWarn(dOutBatch); dOutBatch = nullptr;
        freeOrWarn(dTrtInput); dTrtInput = nullptr;
        freeOrWarn(dMask);     dMask     = nullptr;
        freeOrWarn(dLedgerY);  dLedgerY  = nullptr;
        freeOrWarn(dLedgerX);  dLedgerX  = nullptr;
        freeOrWarn(dLedgerDc); dLedgerDc = nullptr;
        cachedNumBlocks = 0;
    }

    void releaseAll() {
        releaseBatch();
        freeOrWarn(dWinT); dWinT = nullptr;
        freeOrWarn(dWinY); dWinY = nullptr;
        freeOrWarn(dWinX); dWinX = nullptr;
    }

    ~HipThreadCache() { releaseAll(); }
};

HipThreadCache &getCache() {
    static thread_local HipThreadCache cache;
    return cache;
}

// Every HIP call below goes to the calling thread's current device, and the
// cached plans and buffers belong to the device they were created on. Select
// the session's device, releasing the cache on its old device if it moved.
bool ensureDevice(HipThreadCache &cache, int deviceId) {
    if (cache.device == deviceId) return true;
    if (cache.device >= 0) {
        CHD_HIP_CHECK(hipSetDevice(cache.device));
        cache.releaseAll();
    }
    CHD_HIP_CHECK(hipSetDevice(deviceId));
    cache.memInfo = Ort::MemoryInfo("Hip", OrtDeviceAllocator, deviceId, OrtMemTypeDefault);
    cache.device  = deviceId;
    return true;
}

bool ensureBatchCapacity(HipThreadCache &cache, int numBlocks) {
    if (cache.cachedNumBlocks == numBlocks && cache.pFwd && cache.pInv) {
        return true;
    }

    cache.releaseBatch();

    const size_t blockSize = static_cast<size_t>(kNt) * kNy * kNx;
    const size_t batchBytes = sizeof(hipfftDoubleComplex) * numBlocks * blockSize;

    CHD_HIP_CHECK(hipMalloc(&cache.dInBatch,  batchBytes));
    CHD_HIP_CHECK(hipMalloc(&cache.dOutBatch, batchBytes));
    CHD_HIP_CHECK(hipMalloc(&cache.dTrtInput, sizeof(float) * numBlocks * 2 * blockSize));
    CHD_HIP_CHECK(hipMalloc(&cache.dMask,     sizeof(float) * numBlocks * blockSize));
    CHD_HIP_CHECK(hipMalloc(&cache.dLedgerY,  sizeof(int)    * numBlocks));
    CHD_HIP_CHECK(hipMalloc(&cache.dLedgerX,  sizeof(int)    * numBlocks));
    CHD_HIP_CHECK(hipMalloc(&cache.dLedgerDc, sizeof(double) * numBlocks));

    int n[] = { kNt, kNy, kNx };
    CHD_HIPFFT_CHECK(hipfftPlanMany(&cache.pFwd, 3, n,
                                    nullptr, 1, static_cast<int>(blockSize),
                                    nullptr, 1, static_cast<int>(blockSize),
                                    HIPFFT_Z2Z, numBlocks));
    CHD_HIPFFT_CHECK(hipfftPlanMany(&cache.pInv, 3, n,
                                    nullptr, 1, static_cast<int>(blockSize),
                                    nullptr, 1, static_cast<int>(blockSize),
                                    HIPFFT_Z2Z, numBlocks));

    cache.cachedNumBlocks = numBlocks;
    return true;
}

bool ensureWindows(HipThreadCache &cache) {
    if (cache.dWinT && cache.dWinY && cache.dWinX) return true;

    const auto &windows = getSineWindows();
    if (!cache.dWinT) CHD_HIP_CHECK(hipMalloc(&cache.dWinT, sizeof(double) * kNt));
    if (!cache.dWinY) CHD_HIP_CHECK(hipMalloc(&cache.dWinY, sizeof(double) * kNy));
    if (!cache.dWinX) CHD_HIP_CHECK(hipMalloc(&cache.dWinX, sizeof(double) * kNx));
    CHD_HIP_CHECK(hipMemcpy(cache.dWinT, windows.t, sizeof(double) * kNt, hipMemcpyHostToDevice));
    CHD_HIP_CHECK(hipMemcpy(cache.dWinY, windows.y, sizeof(double) * kNy, hipMemcpyHostToDevice));
    CHD_HIP_CHECK(hipMemcpy(cache.dWinX, windows.x, sizeof(double) * kNx, hipMemcpyHostToDevice));
    return true;
}

void flattenInto(double *dst, const std::vector<std::vector<double>> &src,
                 size_t fieldWidth, size_t frameHeight)
{
    for (size_t y = 0; y < frameHeight; ++y) {
        std::memcpy(dst + y * fieldWidth, src[y].data(), sizeof(double) * fieldWidth);
    }
}

void unflattenFrom(std::vector<std::vector<double>> &dst, const double *src,
                   size_t fieldWidth, size_t frameHeight)
{
    for (size_t y = 0; y < frameHeight; ++y) {
        std::memcpy(dst[y].data(), src + y * fieldWidth, sizeof(double) * fieldWidth);
    }
}

}  // namespace

bool hipKernelsRunOnDevice(int deviceId, std::string &whyNot) {
    hipDeviceProp_t props{};
    hipError_t err = hipGetDeviceProperties(&props, deviceId);
    if (err != hipSuccess) {
        whyNot = "couldn't query HIP device " + std::to_string(deviceId) + ": " +
                 hipGetErrorString(err);
        return false;
    }

    const std::string noKernels = "nnTransform3D's HIP kernels can't run on HIP device " +
                                  std::to_string(deviceId) + " (" + props.gcnArchName +
                                  "); this library was built for: " + kOffloadTargets;

    switch (matchHipOffloadTargets(kOffloadTargets, props.gcnArchName)) {
        case HipTargetMatch::Native:
            return true;
        case HipTargetMatch::None:
            whyNot = noKernels;
            return false;
        case HipTargetMatch::RuntimeDecides:
            break;
    }

    // SPIR-V or a generic target might serve this GPU, and only the runtime can
    // tell. Asking resolves the code object, compiling SPIR-V if it has to.
    // Runtimes built with assertions abort here when there's no code object,
    // where a kernel launch would have crashed anyway; AMD's packaged runtimes
    // return an error. Every kernel lives in this one file, so one is enough.
    // The caller's thread may have its own use for HIP, so restore its device.
    int callersDevice = -1;
    if (hipGetDevice(&callersDevice) != hipSuccess) callersDevice = -1;
    err = hipSetDevice(deviceId);
    if (err != hipSuccess) {
        whyNot = "couldn't select HIP device " + std::to_string(deviceId) + ": " +
                 hipGetErrorString(err);
        return false;
    }
    hipFuncAttributes attributes{};
    err = hipFuncGetAttributes(&attributes, reinterpret_cast<const void *>(calcDCKernel));
    if (callersDevice >= 0) (void)hipSetDevice(callersDevice);

    if (err != hipSuccess) {
        whyNot = noKernels + " (" + hipGetErrorString(err) + ")";
        return false;
    }
    return true;
}

// ─── runHipPipeline ────────────────────────────────────────────────────────

bool runHipPipeline(
    const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters,
    const TileGrid &grid,
    const uint16_t *currentRaw,
    const uint16_t *nextRaw,
    bool padF0,
    bool padF1,
    std::vector<std::vector<double>> &currentAccChroma,
    std::vector<std::vector<double>> &currentWeightSum,
    std::vector<std::vector<double>> &nextAccChroma,
    std::vector<std::vector<double>> &nextWeightSum,
    chd::nn::OrtSession &session,
    std::mutex &runMutex,
    double inputMagnitudeScale,
    int deviceId)
{
    const int32_t firstActiveY = grid.firstLine;
    const int32_t lastActiveY  = grid.lastLine;
    const int32_t activeStartX = grid.firstSample;
    const int32_t activeEndX   = grid.endSample;
    const int32_t fieldWidth   = videoParameters.fieldWidth;
    const int32_t frameHeight  = (videoParameters.fieldHeight * 2) - 1;

    const int32_t startY = grid.startY;
    const int32_t startX = grid.startX;

    std::vector<int> hLedgerY;
    std::vector<int> hLedgerX;
    hLedgerY.reserve(64);
    hLedgerX.reserve(64);
    for (int32_t y = startY; y <= lastActiveY; y += kStepY) {
        for (int32_t x = startX; x < activeEndX; x += kStepX) {
            hLedgerY.push_back(y);
            hLedgerX.push_back(x);
        }
    }
    const int numBlocks = static_cast<int>(hLedgerY.size());
    if (numBlocks == 0) return true;

    std::lock_guard<std::mutex> runLock(runMutex);

    auto &cache = getCache();
    if (!ensureDevice(cache, deviceId))         return false;
    if (!ensureBatchCapacity(cache, numBlocks)) return false;
    if (!ensureWindows(cache))                  return false;

    CHD_HIP_CHECK(hipMemcpy(cache.dLedgerY, hLedgerY.data(),
                            sizeof(int) * numBlocks, hipMemcpyHostToDevice));
    CHD_HIP_CHECK(hipMemcpy(cache.dLedgerX, hLedgerX.data(),
                            sizeof(int) * numBlocks, hipMemcpyHostToDevice));

    const size_t cvbsBytes = sizeof(uint16_t) * frameHeight * fieldWidth;
    const size_t accBytes  = sizeof(double)   * frameHeight * fieldWidth;

    uint16_t *dCvbsF0 = nullptr, *dCvbsF1 = nullptr;
    double   *dAccF0  = nullptr, *dAccF1  = nullptr;
    double   *dWtF0   = nullptr, *dWtF1   = nullptr;

    auto cleanupFrameBuffers = [&]() {
        freeOrWarn(dCvbsF0);
        freeOrWarn(dCvbsF1);
        freeOrWarn(dAccF0);
        freeOrWarn(dAccF1);
        freeOrWarn(dWtF0);
        freeOrWarn(dWtF1);
    };

#define CHD_HIP_CHECK_OR_CLEANUP(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        chd::log::warn() << "nnTransform3D[HIP]: " #expr " failed:" \
                         << hipGetErrorString(_err); \
        cleanupFrameBuffers(); \
        return false; \
    } \
} while (0)

#define CHD_HIPFFT_CHECK_OR_CLEANUP(expr) do { \
    hipfftResult _err = (expr); \
    if (_err != HIPFFT_SUCCESS) { \
        chd::log::warn() << "nnTransform3D[HIP]: " #expr " failed with hipFFT code" \
                         << static_cast<int>(_err); \
        cleanupFrameBuffers(); \
        return false; \
    } \
} while (0)

    CHD_HIP_CHECK_OR_CLEANUP(hipMalloc(&dCvbsF0, cvbsBytes));
    CHD_HIP_CHECK_OR_CLEANUP(hipMalloc(&dCvbsF1, cvbsBytes));
    CHD_HIP_CHECK_OR_CLEANUP(hipMalloc(&dAccF0,  accBytes));
    CHD_HIP_CHECK_OR_CLEANUP(hipMalloc(&dAccF1,  accBytes));
    CHD_HIP_CHECK_OR_CLEANUP(hipMalloc(&dWtF0,   accBytes));
    CHD_HIP_CHECK_OR_CLEANUP(hipMalloc(&dWtF1,   accBytes));

    CHD_HIP_CHECK_OR_CLEANUP(hipMemcpy(dCvbsF0, currentRaw, cvbsBytes, hipMemcpyHostToDevice));
    CHD_HIP_CHECK_OR_CLEANUP(hipMemcpy(dCvbsF1, nextRaw,    cvbsBytes, hipMemcpyHostToDevice));

    std::vector<double> flat(static_cast<size_t>(frameHeight) * fieldWidth);

    flattenInto(flat.data(), currentAccChroma, fieldWidth, frameHeight);
    CHD_HIP_CHECK_OR_CLEANUP(hipMemcpy(dAccF0, flat.data(), accBytes, hipMemcpyHostToDevice));
    flattenInto(flat.data(), currentWeightSum, fieldWidth, frameHeight);
    CHD_HIP_CHECK_OR_CLEANUP(hipMemcpy(dWtF0, flat.data(), accBytes, hipMemcpyHostToDevice));
    flattenInto(flat.data(), nextAccChroma,    fieldWidth, frameHeight);
    CHD_HIP_CHECK_OR_CLEANUP(hipMemcpy(dAccF1, flat.data(), accBytes, hipMemcpyHostToDevice));
    flattenInto(flat.data(), nextWeightSum,    fieldWidth, frameHeight);
    CHD_HIP_CHECK_OR_CLEANUP(hipMemcpy(dWtF1, flat.data(), accBytes, hipMemcpyHostToDevice));

    constexpr int kThreadsPerBlock = 256;
    const int blocksForLedger = (numBlocks + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int totalElements   = numBlocks * static_cast<int>(kNt * kNy * kNx);
    const int blocksForAll    = (totalElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int blockSize       = static_cast<int>(kNt * kNy * kNx);

    // The launch checks below read HIP's last error, which an earlier call on
    // this thread may have left behind. Don't blame the kernels for it.
    (void)hipGetLastError();

    calcDCKernel<<<blocksForLedger, kThreadsPerBlock>>>(
        dCvbsF0, dCvbsF1, padF0, padF1,
        cache.dLedgerY, cache.dLedgerX, cache.dLedgerDc,
        numBlocks, activeStartX, activeEndX,
        firstActiveY, lastActiveY, fieldWidth);
    CHD_HIP_CHECK_OR_CLEANUP(hipGetLastError());

    packAndWindowKernel<<<blocksForAll, kThreadsPerBlock>>>(
        dCvbsF0, dCvbsF1, padF0, padF1,
        cache.dInBatch,
        cache.dLedgerY, cache.dLedgerX, cache.dLedgerDc,
        cache.dWinT, cache.dWinY, cache.dWinX,
        numBlocks, blockSize, activeStartX, activeEndX,
        firstActiveY, lastActiveY, fieldWidth);
    CHD_HIP_CHECK_OR_CLEANUP(hipGetLastError());
    CHD_HIP_CHECK_OR_CLEANUP(hipDeviceSynchronize());

    CHD_HIPFFT_CHECK_OR_CLEANUP(hipfftExecZ2Z(cache.pFwd, cache.dInBatch, cache.dOutBatch, HIPFFT_FORWARD));

    calcMagnitudeKernel<<<blocksForAll, kThreadsPerBlock>>>(
        cache.dOutBatch, cache.dTrtInput,
        numBlocks, blockSize, inputMagnitudeScale);
    CHD_HIP_CHECK_OR_CLEANUP(hipGetLastError());
    CHD_HIP_CHECK_OR_CLEANUP(hipDeviceSynchronize());

    try {
        Ort::IoBinding iobinding(session.session());

        const int64_t inputShape[]  = { numBlocks, 2, kNt, kNy, kNx };
        const int64_t outputShape[] = { numBlocks, 1, kNt, kNy, kNx };

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            cache.memInfo, cache.dTrtInput,
            static_cast<size_t>(numBlocks) * 2 * blockSize,
            inputShape, 5);
        Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
            cache.memInfo, cache.dMask,
            static_cast<size_t>(numBlocks) * blockSize,
            outputShape, 5);

        iobinding.BindInput ("input",  inputTensor);
        iobinding.BindOutput("output", outputTensor);

        session.session().Run(Ort::RunOptions{ nullptr }, iobinding);
    } catch (const Ort::Exception &e) {
        chd::log::warn() << "nnTransform3D[HIP] ORT IoBinding Run failed:" << e.what();
        cleanupFrameBuffers();
        return false;
    }

    // Run() may have left a HIP error it handled itself.
    (void)hipGetLastError();

    applyMaskKernel<<<blocksForAll, kThreadsPerBlock>>>(
        cache.dOutBatch, cache.dMask, totalElements);
    CHD_HIP_CHECK_OR_CLEANUP(hipGetLastError());
    CHD_HIP_CHECK_OR_CLEANUP(hipDeviceSynchronize());

    CHD_HIPFFT_CHECK_OR_CLEANUP(hipfftExecZ2Z(cache.pInv, cache.dOutBatch, cache.dInBatch, HIPFFT_BACKWARD));

    olaKernel<<<blocksForAll, kThreadsPerBlock>>>(
        cache.dInBatch,
        dAccF0, dAccF1, dWtF0, dWtF1,
        padF0, padF1,
        cache.dLedgerY, cache.dLedgerX,
        cache.dWinT, cache.dWinY, cache.dWinX,
        numBlocks, blockSize,
        activeStartX, activeEndX,
        firstActiveY, lastActiveY, fieldWidth);
    CHD_HIP_CHECK_OR_CLEANUP(hipGetLastError());
    CHD_HIP_CHECK_OR_CLEANUP(hipDeviceSynchronize());

    CHD_HIP_CHECK_OR_CLEANUP(hipMemcpy(flat.data(), dAccF0, accBytes, hipMemcpyDeviceToHost));
    unflattenFrom(currentAccChroma, flat.data(), fieldWidth, frameHeight);
    CHD_HIP_CHECK_OR_CLEANUP(hipMemcpy(flat.data(), dWtF0, accBytes, hipMemcpyDeviceToHost));
    unflattenFrom(currentWeightSum, flat.data(), fieldWidth, frameHeight);
    CHD_HIP_CHECK_OR_CLEANUP(hipMemcpy(flat.data(), dAccF1, accBytes, hipMemcpyDeviceToHost));
    unflattenFrom(nextAccChroma,    flat.data(), fieldWidth, frameHeight);
    CHD_HIP_CHECK_OR_CLEANUP(hipMemcpy(flat.data(), dWtF1, accBytes, hipMemcpyDeviceToHost));
    unflattenFrom(nextWeightSum,    flat.data(), fieldWidth, frameHeight);

    cleanupFrameBuffers();
    return true;

#undef CHD_HIPFFT_CHECK_OR_CLEANUP
#undef CHD_HIP_CHECK_OR_CLEANUP
}

}  // namespace chd::decoders::nntransform3d
