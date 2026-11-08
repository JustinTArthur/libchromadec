// SPDX-License-Identifier: GPL-3.0-or-later
//
// CUDA cuFFT + ORT-IoBinding pipeline for nnTransform3D, ported by
// inspection from asdfqazsnbb's standalone main.cu reference
// (https://github.com/asdfqazsnbb/nnTransform3D)
//
// Key departures from the reference:
//
//   - Kernels parameterised by frame dimensions (fieldWidth,
//     firstActiveFrameLine, lastActiveFrameLine, activeVideoStart,
//     activeVideoEnd, inputMagnitudeScale) so PAL works alongside NTSC.
//     The reference hardcodes 910×525 + active range 40-525 + scale 128.0.
//
//   - Per-worker thread_local plan + persistent device buffer cache
//     (cuFFT plans, ledger arrays, FFT batch buffers, model I/O tensor
//     memory) instead of the reference's process-wide statics. Per-call we
//     still allocate the per-frame CVBS + accumulator device memory,
//     which simplifies lifecycle at the cost of ~24 MB of cudaMalloc/
//     cudaFree per frame. A future optimisation can persist per-frame
//     device buffers keyed by FrameBuffer pointer (matching the reference's
//     pattern) once the structural port is validated.
//
//   - Single-stream model (default stream + the caller's runMutex held
//     across the entire pipeline) so multiple worker threads serialise
//     on the GPU. Multi-stream parallelism via ORT's
//     OrtCUDAProviderOptionsV2 user_compute_stream is a future
//     optimisation.
//
//   - Error handling: any CUDA / cuFFT / ORT failure returns false so
//     the caller can fall back to the 2D chroma path, matching the CPU
//     body's behaviour on ORT exceptions.

#include "nntransform3d_pipeline_cuda.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <string>

#include "nntransform3d_fft_cpu.h"  // for kNt / kNy / kNx / kStepX / kStepY
#include "nntransform3d_window.h"
#include "../../common/log.h"
#include "../../nn/ort_session.h"

namespace chd::decoders::nntransform3d {

namespace {

// ─── CUDA error helpers ────────────────────────────────────────────────────

#define CHD_CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        chd::log::warn() << "nnTransform3D[CUDA]: " #expr " failed:" \
                         << cudaGetErrorString(_err); \
        return false; \
    } \
} while (0)

#define CHD_CUFFT_CHECK(expr) do { \
    cufftResult _err = (expr); \
    if (_err != CUFFT_SUCCESS) { \
        chd::log::warn() << "nnTransform3D[CUDA]: " #expr " failed with cuFFT code" \
                         << static_cast<int>(_err); \
        return false; \
    } \
} while (0)

// ─── Device kernels ────────────────────────────────────────────────────────
//
// Lifted from asdfqazsnbb's main.cu; the only mechanical edits are (a) all
// hardcoded 910 / 40 / 525 are now kernel parameters (frameStride =
// fieldWidth; firstActiveY / lastActiveY = firstActiveFrameLine /
// lastActiveFrameLine) and (b) the magnitude normalisation constant
// (128.0 in the reference) is passed as a parameter so chroma_net v1
// (scale 1.0) also works.

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
                                    cufftDoubleComplex *dInBatch,
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

// Channel 0 = |F|; channel 1 = |F'| where F' is the centro-symmetric
// reflection (matches the CPU body's reflectedIndex loop in comb.cpp).
__global__ void calcMagnitudeKernel(const cufftDoubleComplex *dOutBatch,
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

__global__ void applyMaskKernel(cufftDoubleComplex *dOutBatch,
                                const float *dMask,
                                int totalElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;
    const float gain = dMask[idx];
    dOutBatch[idx].x *= gain;
    dOutBatch[idx].y *= gain;
}

__global__ void olaKernel(const cufftDoubleComplex *dInBatch,
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
//
// cuFFT plans + batch buffers + ledger/window arrays survive across calls
// on the same worker thread. They're sized by numBlocks (constant per
// (video standard, active region) combination), so the cached_* fields
// detect a change and rebuild. The per-frame CVBS + accumulator
// allocations are NOT in here — they're per-call (see runCudaPipeline).
struct CudaThreadCache {
    cufftHandle pFwd = 0;
    cufftHandle pInv = 0;
    int cachedNumBlocks = 0;

    cufftDoubleComplex *dInBatch  = nullptr;
    cufftDoubleComplex *dOutBatch = nullptr;
    float              *dTrtInput = nullptr;
    float              *dMask     = nullptr;

    int    *dLedgerY  = nullptr;
    int    *dLedgerX  = nullptr;
    double *dLedgerDc = nullptr;

    double *dWinT = nullptr;
    double *dWinY = nullptr;
    double *dWinX = nullptr;

    // CUDA device the plans and buffers were created on; -1 before first use.
    int device = -1;
    // CUDA-pointer MemoryInfo reused for every Ort::Value::CreateTensor
    // call. Brace-init avoids a copy of a move-only ORT C++ type.
    Ort::MemoryInfo memInfo{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

    // Best-effort cleanup; return values go unchecked because the destructor
    // may run after CUDA has shut down through atexit sequencing.
    void releaseBatch() {
        if (pFwd)      { cufftDestroy(pFwd);  pFwd = 0; }
        if (pInv)      { cufftDestroy(pInv);  pInv = 0; }
        if (dInBatch)  { cudaFree(dInBatch);  dInBatch  = nullptr; }
        if (dOutBatch) { cudaFree(dOutBatch); dOutBatch = nullptr; }
        if (dTrtInput) { cudaFree(dTrtInput); dTrtInput = nullptr; }
        if (dMask)     { cudaFree(dMask);     dMask     = nullptr; }
        if (dLedgerY)  { cudaFree(dLedgerY);  dLedgerY  = nullptr; }
        if (dLedgerX)  { cudaFree(dLedgerX);  dLedgerX  = nullptr; }
        if (dLedgerDc) { cudaFree(dLedgerDc); dLedgerDc = nullptr; }
        cachedNumBlocks = 0;
    }

    void releaseAll() {
        releaseBatch();
        if (dWinT) { cudaFree(dWinT); dWinT = nullptr; }
        if (dWinY) { cudaFree(dWinY); dWinY = nullptr; }
        if (dWinX) { cudaFree(dWinX); dWinX = nullptr; }
    }

    ~CudaThreadCache() { releaseAll(); }
};

CudaThreadCache &getCache() {
    static thread_local CudaThreadCache cache;
    return cache;
}

// Every CUDA call below goes to the calling thread's current device, and the
// cached plans and buffers belong to the device they were created on. Select
// the session's device, releasing the cache on its old device if it moved.
bool ensureDevice(CudaThreadCache &cache, int deviceId) {
    if (cache.device == deviceId) return true;
    if (cache.device >= 0) {
        CHD_CUDA_CHECK(cudaSetDevice(cache.device));
        cache.releaseAll();
    }
    CHD_CUDA_CHECK(cudaSetDevice(deviceId));
    cache.memInfo = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, deviceId, OrtMemTypeDefault);
    cache.device  = deviceId;
    return true;
}

bool ensureBatchCapacity(CudaThreadCache &cache, int numBlocks) {
    if (cache.cachedNumBlocks == numBlocks && cache.pFwd && cache.pInv) {
        return true;
    }

    cache.releaseBatch();

    const size_t blockSize = static_cast<size_t>(kNt) * kNy * kNx;
    const size_t batchBytes = sizeof(cufftDoubleComplex) * numBlocks * blockSize;

    CHD_CUDA_CHECK(cudaMalloc(&cache.dInBatch,  batchBytes));
    CHD_CUDA_CHECK(cudaMalloc(&cache.dOutBatch, batchBytes));
    CHD_CUDA_CHECK(cudaMalloc(&cache.dTrtInput, sizeof(float) * numBlocks * 2 * blockSize));
    CHD_CUDA_CHECK(cudaMalloc(&cache.dMask,     sizeof(float) * numBlocks * blockSize));
    CHD_CUDA_CHECK(cudaMalloc(&cache.dLedgerY,  sizeof(int)    * numBlocks));
    CHD_CUDA_CHECK(cudaMalloc(&cache.dLedgerX,  sizeof(int)    * numBlocks));
    CHD_CUDA_CHECK(cudaMalloc(&cache.dLedgerDc, sizeof(double) * numBlocks));

    int n[] = { kNt, kNy, kNx };
    CHD_CUFFT_CHECK(cufftPlanMany(&cache.pFwd, 3, n,
                                  nullptr, 1, static_cast<int>(blockSize),
                                  nullptr, 1, static_cast<int>(blockSize),
                                  CUFFT_Z2Z, numBlocks));
    CHD_CUFFT_CHECK(cufftPlanMany(&cache.pInv, 3, n,
                                  nullptr, 1, static_cast<int>(blockSize),
                                  nullptr, 1, static_cast<int>(blockSize),
                                  CUFFT_Z2Z, numBlocks));

    cache.cachedNumBlocks = numBlocks;
    return true;
}

bool ensureWindows(CudaThreadCache &cache) {
    if (cache.dWinT && cache.dWinY && cache.dWinX) return true;

    const auto &windows = getSineWindows();
    if (!cache.dWinT) CHD_CUDA_CHECK(cudaMalloc(&cache.dWinT, sizeof(double) * kNt));
    if (!cache.dWinY) CHD_CUDA_CHECK(cudaMalloc(&cache.dWinY, sizeof(double) * kNy));
    if (!cache.dWinX) CHD_CUDA_CHECK(cudaMalloc(&cache.dWinX, sizeof(double) * kNx));
    CHD_CUDA_CHECK(cudaMemcpy(cache.dWinT, windows.t, sizeof(double) * kNt, cudaMemcpyHostToDevice));
    CHD_CUDA_CHECK(cudaMemcpy(cache.dWinY, windows.y, sizeof(double) * kNy, cudaMemcpyHostToDevice));
    CHD_CUDA_CHECK(cudaMemcpy(cache.dWinX, windows.x, sizeof(double) * kNx, cudaMemcpyHostToDevice));
    return true;
}

// Flatten the std::vector<std::vector<double>> accumulator into a contiguous
// host buffer for one bulk H2D / D2H copy. The CPU body keeps these as
// vector-of-vectors so we have to flatten at the GPU boundary.
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

// ─── runCudaPipeline ───────────────────────────────────────────────────────

bool runCudaPipeline(
    const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters,
    const uint16_t *currentRaw,
    const uint16_t *nextRaw,
    std::vector<std::vector<double>> &currentAccChroma,
    std::vector<std::vector<double>> &currentWeightSum,
    std::vector<std::vector<double>> &nextAccChroma,
    std::vector<std::vector<double>> &nextWeightSum,
    chd::nn::OrtSession &session,
    std::mutex &runMutex,
    double inputMagnitudeScale,
    int deviceId)
{
    const int32_t firstActiveY = videoParameters.firstActiveFrameLine;
    const int32_t lastActiveY  = videoParameters.lastActiveFrameLine;
    const int32_t activeStartX = videoParameters.activeVideoStart;
    const int32_t activeEndX   = videoParameters.activeVideoEnd;
    const int32_t fieldWidth   = videoParameters.fieldWidth;
    const int32_t frameHeight  = (videoParameters.fieldHeight * 2) - 1;

    const int32_t startY = firstActiveY - (kNy / 2);
    const int32_t startX = activeStartX - (kNx / 2);

    // Build the ledger (block top-left coordinates) on the host first;
    // it determines numBlocks which sizes all the device buffers.
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

    // Hold runMutex across the whole pipeline (kernel launches + cuFFT +
    // Run + D2H) so worker threads don't interleave commands on the CUDA
    // default stream.
    std::lock_guard<std::mutex> runLock(runMutex);

    auto &cache = getCache();
    if (!ensureDevice(cache, deviceId))         return false;
    if (!ensureBatchCapacity(cache, numBlocks)) return false;
    if (!ensureWindows(cache))                  return false;

    CHD_CUDA_CHECK(cudaMemcpy(cache.dLedgerY, hLedgerY.data(),
                              sizeof(int) * numBlocks, cudaMemcpyHostToDevice));
    CHD_CUDA_CHECK(cudaMemcpy(cache.dLedgerX, hLedgerX.data(),
                              sizeof(int) * numBlocks, cudaMemcpyHostToDevice));

    // ── Per-call frame buffers ──────────────────────────────────────────
    // d_cvbs[2], d_accChroma[2], d_weightSum[2]. We upload + download
    // these every call rather than persisting across split3DnnTransform
    // calls; ~16 MB extra H2D/D2H traffic per call vs the reference's
    // persistent-keyed-by-frame design, which is negligible compared to
    // the FFT + inference cost. A future optimisation can persist these
    // by FrameBuffer pointer (the reference pattern).

    const size_t cvbsBytes = sizeof(uint16_t) * frameHeight * fieldWidth;
    const size_t accBytes  = sizeof(double)   * frameHeight * fieldWidth;

    uint16_t *dCvbsF0 = nullptr, *dCvbsF1 = nullptr;
    double   *dAccF0  = nullptr, *dAccF1  = nullptr;
    double   *dWtF0   = nullptr, *dWtF1   = nullptr;

    auto cleanupFrameBuffers = [&]() {
        if (dCvbsF0) cudaFree(dCvbsF0);
        if (dCvbsF1) cudaFree(dCvbsF1);
        if (dAccF0)  cudaFree(dAccF0);
        if (dAccF1)  cudaFree(dAccF1);
        if (dWtF0)   cudaFree(dWtF0);
        if (dWtF1)   cudaFree(dWtF1);
    };

#define CHD_CUDA_CHECK_OR_CLEANUP(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        chd::log::warn() << "nnTransform3D[CUDA]: " #expr " failed:" \
                         << cudaGetErrorString(_err); \
        cleanupFrameBuffers(); \
        return false; \
    } \
} while (0)

    CHD_CUDA_CHECK_OR_CLEANUP(cudaMalloc(&dCvbsF0, cvbsBytes));
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMalloc(&dCvbsF1, cvbsBytes));
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMalloc(&dAccF0,  accBytes));
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMalloc(&dAccF1,  accBytes));
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMalloc(&dWtF0,   accBytes));
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMalloc(&dWtF1,   accBytes));

    CHD_CUDA_CHECK_OR_CLEANUP(cudaMemcpy(dCvbsF0, currentRaw, cvbsBytes, cudaMemcpyHostToDevice));
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMemcpy(dCvbsF1, nextRaw,    cvbsBytes, cudaMemcpyHostToDevice));

    // Flatten + upload accumulators (may carry partial contributions from
    // a previous split3DnnTransform call).
    std::vector<double> flat(static_cast<size_t>(frameHeight) * fieldWidth);

    flattenInto(flat.data(), currentAccChroma, fieldWidth, frameHeight);
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMemcpy(dAccF0, flat.data(), accBytes, cudaMemcpyHostToDevice));
    flattenInto(flat.data(), currentWeightSum, fieldWidth, frameHeight);
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMemcpy(dWtF0, flat.data(), accBytes, cudaMemcpyHostToDevice));
    flattenInto(flat.data(), nextAccChroma,    fieldWidth, frameHeight);
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMemcpy(dAccF1, flat.data(), accBytes, cudaMemcpyHostToDevice));
    flattenInto(flat.data(), nextWeightSum,    fieldWidth, frameHeight);
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMemcpy(dWtF1, flat.data(), accBytes, cudaMemcpyHostToDevice));

    // ── Pipeline ────────────────────────────────────────────────────────
    constexpr int kThreadsPerBlock = 256;
    const int blocksForLedger = (numBlocks + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int totalElements   = numBlocks * static_cast<int>(kNt * kNy * kNx);
    const int blocksForAll    = (totalElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int blockSize       = static_cast<int>(kNt * kNy * kNx);

    // Per the reference: padding flags for boundary frames. In our flow both
    // frames are always valid (Comb::decodeFrames pads via a duplicated
    // frame in getFrameBuffer), so both are non-padding. Kept as
    // parameters for future caller-set look-behind/look-ahead.
    constexpr bool padF0 = false;
    constexpr bool padF1 = false;

    calcDCKernel<<<blocksForLedger, kThreadsPerBlock>>>(
        dCvbsF0, dCvbsF1, padF0, padF1,
        cache.dLedgerY, cache.dLedgerX, cache.dLedgerDc,
        numBlocks, activeStartX, activeEndX,
        firstActiveY, lastActiveY, fieldWidth);

    packAndWindowKernel<<<blocksForAll, kThreadsPerBlock>>>(
        dCvbsF0, dCvbsF1, padF0, padF1,
        cache.dInBatch,
        cache.dLedgerY, cache.dLedgerX, cache.dLedgerDc,
        cache.dWinT, cache.dWinY, cache.dWinX,
        numBlocks, blockSize, activeStartX, activeEndX,
        firstActiveY, lastActiveY, fieldWidth);
    CHD_CUDA_CHECK_OR_CLEANUP(cudaDeviceSynchronize());

    CHD_CUFFT_CHECK(cufftExecZ2Z(cache.pFwd, cache.dInBatch, cache.dOutBatch, CUFFT_FORWARD));

    calcMagnitudeKernel<<<blocksForAll, kThreadsPerBlock>>>(
        cache.dOutBatch, cache.dTrtInput,
        numBlocks, blockSize, inputMagnitudeScale);
    CHD_CUDA_CHECK_OR_CLEANUP(cudaDeviceSynchronize());

    // ── ORT inference via IoBinding on device pointers ──────────────────
    // The CPU body's input/output tensor names ("input"/"output") match
    // both chroma_net v1 and v2 — same as the reference main.cu uses.
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
        chd::log::warn() << "nnTransform3D[CUDA] ORT IoBinding Run failed:" << e.what();
        cleanupFrameBuffers();
        return false;
    }

    applyMaskKernel<<<blocksForAll, kThreadsPerBlock>>>(
        cache.dOutBatch, cache.dMask, totalElements);
    CHD_CUDA_CHECK_OR_CLEANUP(cudaDeviceSynchronize());

    CHD_CUFFT_CHECK(cufftExecZ2Z(cache.pInv, cache.dOutBatch, cache.dInBatch, CUFFT_INVERSE));

    olaKernel<<<blocksForAll, kThreadsPerBlock>>>(
        cache.dInBatch,
        dAccF0, dAccF1, dWtF0, dWtF1,
        padF0, padF1,
        cache.dLedgerY, cache.dLedgerX,
        cache.dWinT, cache.dWinY, cache.dWinX,
        numBlocks, blockSize,
        activeStartX, activeEndX,
        firstActiveY, lastActiveY, fieldWidth);
    CHD_CUDA_CHECK_OR_CLEANUP(cudaDeviceSynchronize());

    // ── D2H accumulators ────────────────────────────────────────────────
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMemcpy(flat.data(), dAccF0, accBytes, cudaMemcpyDeviceToHost));
    unflattenFrom(currentAccChroma, flat.data(), fieldWidth, frameHeight);
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMemcpy(flat.data(), dWtF0, accBytes, cudaMemcpyDeviceToHost));
    unflattenFrom(currentWeightSum, flat.data(), fieldWidth, frameHeight);
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMemcpy(flat.data(), dAccF1, accBytes, cudaMemcpyDeviceToHost));
    unflattenFrom(nextAccChroma,    flat.data(), fieldWidth, frameHeight);
    CHD_CUDA_CHECK_OR_CLEANUP(cudaMemcpy(flat.data(), dWtF1, accBytes, cudaMemcpyDeviceToHost));
    unflattenFrom(nextWeightSum,    flat.data(), fieldWidth, frameHeight);

    cleanupFrameBuffers();
    return true;

#undef CHD_CUDA_CHECK_OR_CLEANUP
}

}  // namespace chd::decoders::nntransform3d
