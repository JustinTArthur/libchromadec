// SPDX-License-Identifier: GPL-3.0-or-later
//
// Native CoreML pipeline for nnTransform3D. See the header for the design.
//
// The window→DC-subtract→3D-FFT→magnitude pre-step and the
// mask→inverse-FFT→overlap-add post-step are the same DSP the comb CPU body
// and the CUDA/HIP pipelines run; the departures are (a) the 3D-convolution
// mask is produced by a native CoreML MLModel over a *batch* of tiles (so the
// conv reaches GPU/ANE), and (b) the FFT is pluggable:
//
//   Layer 1 (default) — CPU FFTW (double) around the CoreML conv.
//   Layer 2 (opt-in)  — GPU-resident: MPSGraph device FFT + Metal compute
//                       kernels for the magnitude/mask steps + a cpuAndGPU
//                       CoreML conv that reads/writes shared MTLBuffers via
//                       outputBackings. The spectrum stays in unified-memory
//                       Metal buffers across FFT → conv → IFFT; only the
//                       host-side pack/window (entry) and overlap-add (exit)
//                       touch CPU memory, with a double↔f32 conversion at the
//                       Metal boundary. Selected by CHD_NNTRANSFORM3D_COREML_FFT=mps
//                       on macOS 14+; falls back to Layer 1 on older macOS, no
//                       Metal device, or any setup/runtime failure.

#include "nntransform3d_pipeline_coreml.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <fftw3.h>

#include "nntransform3d_fft_cpu.h"   // kNt / kNy / kNx / kStepX / kStepY + plans
#include "nntransform3d_window.h"
#include "../../common/log.h"
#include "../../nn/coreml_engine.h"
#include "../../nn/inference_engine.h"

// Layer 2 (MPSGraph device FFT + Metal kernels) needs a macOS 14 SDK to compile
// the FFT selectors / complex dtype. Older SDKs build Layer 1 only.
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
#define CHD_HAS_MPSGRAPH_FFT 1
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#endif

namespace chd::decoders::nntransform3d {

namespace {

constexpr int kBlockSize   = kNt * kNy * kNx;    // 1024 points per tile
constexpr int kBatchBlocks = 256;                // tiles per CoreML inference

inline int idx3(int t, int y, int x) { return (t * kNy + y) * kNx + x; }

// ─── FFT strategy selection (probed once) ─────────────────────────────────────

enum class FftStrategy { Fftw, Mps };

FftStrategy probeFftStrategy()
{
    const char *env = std::getenv("CHD_NNTRANSFORM3D_COREML_FFT");
    const bool wantMps = (env != nullptr) && (std::string(env) == "mps");
    if (!wantMps) return FftStrategy::Fftw;

#if defined(CHD_HAS_MPSGRAPH_FFT)
    if (@available(macOS 14.0, *)) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device != nil) {
            chd::log::info() << "nnTransform3D[CoreML]: using GPU-resident MPSGraph FFT (Layer 2, f32)";
            return FftStrategy::Mps;
        }
        chd::log::warn() << "nnTransform3D[CoreML]: no Metal device; falling back to FFTW FFT";
        return FftStrategy::Fftw;
    }
#endif
    chd::log::warn() << "nnTransform3D[CoreML]: MPSGraph FFT requested but needs macOS 14; using FFTW FFT";
    return FftStrategy::Fftw;
}

FftStrategy fftStrategy()
{
    static const FftStrategy strategy = probeFftStrategy();
    return strategy;
}

// ─── GPU-resident MPS context (Layer 2) ───────────────────────────────────────

#if defined(CHD_HAS_MPSGRAPH_FFT)

// Metal kernels mirroring the CUDA calcMagnitude / applyMask kernels, in f32.
// magnitudeReflection: spectrum (complex) → [mag | centro-symmetric reflection]
// stacked as the model's 2-channel input. applyMask: spectrum *= mask.
static NSString *const kKernelSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void magnitudeReflection(device const float2 *spec       [[buffer(0)]],
                                device float         *modelInput [[buffer(1)]],
                                constant float       &invScale   [[buffer(2)]],
                                constant uint        &numBlocks  [[buffer(3)]],
                                uint gid [[thread_position_in_grid]])
{
    const uint total = numBlocks * 1024u;
    if (gid >= total) return;
    const uint b = gid / 1024u;
    const uint i = gid % 1024u;
    const int Nx = 16, Ny = 16;
    const int t = int(i) / (Ny * Nx);
    const int rem = int(i) % (Ny * Nx);
    const int y = rem / Nx;
    const int x = rem % Nx;

    float2 c = spec[gid];
    float mag = sqrt(c.x * c.x + c.y * c.y);

    int refT = (2 - t) % 4; if (refT < 0) refT += 4;
    int refY = (16 - y) % 16;
    int refX = (8 - x) % 16; if (refX < 0) refX += 16;
    uint idxRef = b * 1024u + uint(refT * Ny * Nx + refY * Nx + refX);
    float2 cr = spec[idxRef];
    float magRef = sqrt(cr.x * cr.x + cr.y * cr.y);

    modelInput[b * 2048u + 0u * 1024u + i] = mag * invScale;
    modelInput[b * 2048u + 1u * 1024u + i] = magRef * invScale;
}

kernel void applyMask(device float2      *spec  [[buffer(0)]],
                      device const float *mask  [[buffer(1)]],
                      constant uint       &total [[buffer(2)]],
                      uint gid [[thread_position_in_grid]])
{
    if (gid >= total) return;
    float g = mask[gid];
    spec[gid] *= g;
}
)METAL";

class ResidentMps {
public:
    // Allocate device + queue + pipelines + buffers sized for `maxBlocks`.
    // Idempotent; returns false (and stays unready) on any failure.
    bool ensure(int maxBlocks) API_AVAILABLE(macos(14.0))
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ready_) return true;
        if (triedAndFailed_) return false;

        device_ = MTLCreateSystemDefaultDevice();
        queue_  = device_ ? [device_ newCommandQueue] : nil;
        if (device_ == nil || queue_ == nil) { triedAndFailed_ = true; return false; }

        NSError *err = nil;
        id<MTLLibrary> lib = [device_ newLibraryWithSource:kKernelSource options:nil error:&err];
        if (lib == nil) {
            chd::log::warn() << "nnTransform3D[CoreML]: Metal kernel compile failed:"
                             << (err ? err.localizedDescription.UTF8String : "unknown");
            triedAndFailed_ = true;
            return false;
        }
        magPipe_  = pipelineFor(lib, @"magnitudeReflection");
        maskPipe_ = pipelineFor(lib, @"applyMask");
        if (magPipe_ == nil || maskPipe_ == nil) { triedAndFailed_ = true; return false; }

        const NSUInteger complexBytes = static_cast<NSUInteger>(maxBlocks) * kBlockSize * 2 * sizeof(float);
        const NSUInteger modelBytes   = static_cast<NSUInteger>(maxBlocks) * 2 * kBlockSize * sizeof(float);
        const NSUInteger maskBytes    = static_cast<NSUInteger>(maxBlocks) * kBlockSize * sizeof(float);
        bufTiles_      = [device_ newBufferWithLength:complexBytes options:MTLResourceStorageModeShared];
        bufSpec_       = [device_ newBufferWithLength:complexBytes options:MTLResourceStorageModeShared];
        bufIfft_       = [device_ newBufferWithLength:complexBytes options:MTLResourceStorageModeShared];
        bufModelInput_ = [device_ newBufferWithLength:modelBytes   options:MTLResourceStorageModeShared];
        bufMask_       = [device_ newBufferWithLength:maskBytes     options:MTLResourceStorageModeShared];
        if (bufTiles_ == nil || bufSpec_ == nil || bufIfft_ == nil ||
            bufModelInput_ == nil || bufMask_ == nil) { triedAndFailed_ = true; return false; }

        graphCache_ = [NSMutableDictionary dictionary];
        ready_ = true;
        return true;
    }

    float *modelInputContents() { return static_cast<float *>(bufModelInput_.contents); }
    float *maskContents()       { return static_cast<float *>(bufMask_.contents); }

    // Upload the packed (double) tiles into the f32 device input buffer.
    void uploadTiles(const double *interleaved, int batchBlocks)
    {
        float *dst = static_cast<float *>(bufTiles_.contents);
        const size_t n = static_cast<size_t>(batchBlocks) * kBlockSize * 2;
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(interleaved[i]);
    }

    // Read the f32 inverse-FFT result back into the (double) host buffer.
    void downloadIfft(double *interleaved, int batchBlocks)
    {
        const float *src = static_cast<const float *>(bufIfft_.contents);
        const size_t n = static_cast<size_t>(batchBlocks) * kBlockSize * 2;
        for (size_t i = 0; i < n; ++i) interleaved[i] = static_cast<double>(src[i]);
    }

    bool forwardFFT(int batchBlocks) API_AVAILABLE(macos(14.0))
    {
        return runFFT(bufTiles_, bufSpec_, batchBlocks, /*inverse=*/false);
    }
    bool inverseFFT(int batchBlocks) API_AVAILABLE(macos(14.0))
    {
        return runFFT(bufSpec_, bufIfft_, batchBlocks, /*inverse=*/true);
    }

    bool magnitude(int batchBlocks, float invScale)
    {
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:magPipe_];
        [enc setBuffer:bufSpec_ offset:0 atIndex:0];
        [enc setBuffer:bufModelInput_ offset:0 atIndex:1];
        [enc setBytes:&invScale length:sizeof(float) atIndex:2];
        uint nb = static_cast<uint>(batchBlocks);
        [enc setBytes:&nb length:sizeof(uint) atIndex:3];
        dispatch(enc, magPipe_, static_cast<NSUInteger>(batchBlocks) * kBlockSize);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb.status == MTLCommandBufferStatusCompleted;
    }

    bool applyMask(int batchBlocks)
    {
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:maskPipe_];
        [enc setBuffer:bufSpec_ offset:0 atIndex:0];
        [enc setBuffer:bufMask_ offset:0 atIndex:1];
        uint total = static_cast<uint>(batchBlocks) * kBlockSize;
        [enc setBytes:&total length:sizeof(uint) atIndex:2];
        dispatch(enc, maskPipe_, total);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return cb.status == MTLCommandBufferStatusCompleted;
    }

private:
    id<MTLComputePipelineState> pipelineFor(id<MTLLibrary> lib, NSString *name)
    {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (fn == nil) return nil;
        NSError *err = nil;
        return [device_ newComputePipelineStateWithFunction:fn error:&err];
    }

    void dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe, NSUInteger total)
    {
        NSUInteger tg = std::min<NSUInteger>(pipe.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(total, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }

    bool runFFT(id<MTLBuffer> inBuf, id<MTLBuffer> outBuf, int N, bool inverse) API_AVAILABLE(macos(14.0))
    {
        NSNumber *key = @(N * 2 + (inverse ? 1 : 0));
        NSArray *cached = graphCache_[key];
        MPSGraph *graph; MPSGraphTensor *ph; MPSGraphTensor *outT;
        if (cached != nil) {
            graph = cached[0]; ph = cached[1]; outT = cached[2];
        } else {
            graph = [MPSGraph new];
            NSArray<NSNumber *> *shape = @[ @(N), @(kNt), @(kNy), @(kNx) ];
            ph = [graph placeholderWithShape:shape dataType:MPSDataTypeComplexFloat32 name:nil];
            MPSGraphFFTDescriptor *desc = [MPSGraphFFTDescriptor descriptor];
            desc.inverse = inverse;
            desc.scalingMode = MPSGraphFFTScalingModeNone;   // FFTW convention
            outT = [graph fastFourierTransformWithTensor:ph axes:@[ @1, @2, @3 ]
                                              descriptor:desc name:nil];
            graphCache_[key] = @[ graph, ph, outT ];
        }

        NSArray<NSNumber *> *shape = @[ @(N), @(kNt), @(kNy), @(kNx) ];
        MPSGraphTensorData *inData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf shape:shape
                                                 dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *outData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf shape:shape
                                                 dataType:MPSDataTypeComplexFloat32];
        NSMutableDictionary *results =
            [NSMutableDictionary dictionaryWithObject:outData forKey:outT];
        @try {
            [graph runWithMTLCommandQueue:queue_
                                    feeds:@{ ph : inData }
                         targetOperations:nil
                        resultsDictionary:results];
        } @catch (NSException *e) {
            chd::log::warn() << "nnTransform3D[CoreML]: MPSGraph FFT run threw:" << e.reason.UTF8String;
            return false;
        }
        return true;
    }

    std::mutex                  mutex_;
    bool                        ready_ = false;
    bool                        triedAndFailed_ = false;
    id<MTLDevice>               device_  = nil;
    id<MTLCommandQueue>         queue_   = nil;
    id<MTLComputePipelineState> magPipe_  = nil;
    id<MTLComputePipelineState> maskPipe_ = nil;
    id<MTLBuffer>               bufTiles_      = nil;
    id<MTLBuffer>               bufSpec_       = nil;
    id<MTLBuffer>               bufIfft_       = nil;
    id<MTLBuffer>               bufModelInput_ = nil;
    id<MTLBuffer>               bufMask_       = nil;
    NSMutableDictionary        *graphCache_    = nil;
};

ResidentMps &residentMps()
{
    static ResidentMps instance;
    return instance;
}
#endif  // CHD_HAS_MPSGRAPH_FFT

}  // namespace

// ─── runCoreMLPipeline ────────────────────────────────────────────────────────

bool runCoreMLPipeline(
    const chd::metadata::LdDecodeMetaData::VideoParameters &videoParameters,
    const uint16_t *currentRaw,
    const uint16_t *nextRaw,
    std::vector<std::vector<double>> &currentAccChroma,
    std::vector<std::vector<double>> &currentWeightSum,
    std::vector<std::vector<double>> &nextAccChroma,
    std::vector<std::vector<double>> &nextWeightSum,
    chd::nn::CoreMLEngine &engine,
    std::mutex &runMutex,
    double inputMagnitudeScale)
{
    const int32_t firstActiveY = videoParameters.firstActiveFrameLine;
    const int32_t lastActiveY  = videoParameters.lastActiveFrameLine;
    const int32_t activeStartX = videoParameters.activeVideoStart;
    const int32_t activeEndX   = videoParameters.activeVideoEnd;
    const int32_t fieldWidth   = videoParameters.fieldWidth;

    const int32_t startY = firstActiveY - (kNy / 2);
    const int32_t startX = activeStartX - (kNx / 2);

    // Block top-left coordinates (matches the CUDA ledger + CPU body loop).
    std::vector<int> ledgerY;
    std::vector<int> ledgerX;
    for (int32_t y = startY; y <= lastActiveY; y += kStepY) {
        for (int32_t x = startX; x < activeEndX; x += kStepX) {
            ledgerY.push_back(y);
            ledgerX.push_back(x);
        }
    }
    const int numBlocks = static_cast<int>(ledgerY.size());
    if (numBlocks == 0) return true;

    const auto &windows = getSineWindows();
    const double *winX = windows.x;
    const double *winY = windows.y;
    const double *winT = windows.t;
    const double invScale = (inputMagnitudeScale != 0.0) ? (1.0 / inputMagnitudeScale) : 1.0;

    const uint16_t *frames[2] = { currentRaw, nextRaw };
    std::vector<std::vector<double>> *accFor[2]    = { &currentAccChroma, &nextAccChroma };
    std::vector<std::vector<double>> *weightFor[2] = { &currentWeightSum, &nextWeightSum };

    // Decide the FFT strategy once. Layer 2 needs the resident MPS context up.
    bool useResident = false;
#if defined(CHD_HAS_MPSGRAPH_FFT)
    if (fftStrategy() == FftStrategy::Mps) {
        if (engine.computeUnits() == chd::nn::CoreMLComputeUnits::All) {
            // The resident path keeps tensors in Metal buffers; a conv that
            // CoreML schedules on the ANE has to sync/copy across the
            // Metal/ANE boundary every chunk, which costs more than the
            // device FFT saves. Warn once rather than second-guess the
            // caller's compute-units choice.
            static std::once_flag warned;
            std::call_once(warned, [] {
                chd::log::warn() << "nnTransform3D[CoreML]: MPSGraph FFT combined with"
                                 << "compute units ALL forces Metal<->ANE transfers each"
                                 << "chunk and measures slower than the FFTW FFT; prefer"
                                 << "unsetting CHD_NNTRANSFORM3D_COREML_FFT with ALL";
            });
        }
        if (@available(macOS 14.0, *)) {
            useResident = residentMps().ensure(kBatchBlocks);
        }
    }
#endif

    // Per-thread reusable host buffers. inStore holds the packed tiles (and,
    // for both paths, the IFFT result the overlap-add reads). The Layer-1 host
    // path also uses specStore + modelInput; the resident path keeps those on
    // the GPU instead.
    static thread_local std::vector<double> inStore;
    static thread_local std::vector<double> specStore;
    static thread_local std::vector<float>  modelInput;
    inStore.resize(static_cast<size_t>(kBatchBlocks) * kBlockSize * 2);
    auto *inBatch   = reinterpret_cast<fftw_complex *>(inStore.data());
    fftw_complex *specBatch = nullptr;
    if (!useResident) {
        specStore.resize(static_cast<size_t>(kBatchBlocks) * kBlockSize * 2);
        modelInput.resize(static_cast<size_t>(kBatchBlocks) * 2 * kBlockSize);
        specBatch = reinterpret_cast<fftw_complex *>(specStore.data());
    }

    auto &plans = getThreadLocalCpuPlans();
    if (!useResident && (plans.forward == nullptr || plans.inverse == nullptr)) return false;

    for (int chunkStart = 0; chunkStart < numBlocks; chunkStart += kBatchBlocks) {
        const int chunkN = std::min(kBatchBlocks, numBlocks - chunkStart);

        // ── Pack + window each tile (mirrors comb.cpp CPU body passes 1+2) ──
        std::fill(inStore.begin(), inStore.begin() + static_cast<size_t>(chunkN) * kBlockSize * 2, 0.0);
        for (int j = 0; j < chunkN; ++j) {
            const int32_t y = ledgerY[chunkStart + j];
            const int32_t x = ledgerX[chunkStart + j];
            fftw_complex *tile = inBatch + static_cast<size_t>(j) * kBlockSize;

            double blockDc = 0.0;
            int32_t sampleCount = 0;
            for (int32_t t = 0; t < kNt; ++t) {
                const bool oddField = (t % 2) != 0;
                const uint16_t *cvbs = frames[t / 2];
                for (int32_t dy = 0; dy < kNy; ++dy) {
                    const int32_t absY = y + dy;
                    if (absY < firstActiveY || absY > lastActiveY) continue;
                    if (((absY % 2) != 0) != oddField) continue;
                    const uint16_t *line = cvbs + (absY * fieldWidth);
                    for (int32_t dx = 0; dx < kNx; ++dx) {
                        const int32_t absX = x + dx;
                        if (absX < activeStartX || absX >= activeEndX) continue;
                        const double value = static_cast<double>(line[absX]);
                        tile[idx3(t, dy, dx)][0] = value;
                        blockDc += value;
                        ++sampleCount;
                    }
                }
            }
            if (sampleCount == 0) continue;
            blockDc /= static_cast<double>(sampleCount);

            for (int32_t t = 0; t < kNt; ++t) {
                const bool oddField = (t % 2) != 0;
                for (int32_t dy = 0; dy < kNy; ++dy) {
                    const int32_t absY = y + dy;
                    const bool yActive = absY >= firstActiveY && absY <= lastActiveY;
                    const bool oddLine = (absY % 2) != 0;
                    for (int32_t dx = 0; dx < kNx; ++dx) {
                        const int32_t absX = x + dx;
                        const bool xActive = absX >= activeStartX && absX < activeEndX;
                        double *cell = tile[idx3(t, dy, dx)];
                        if (yActive && xActive && (oddLine == oddField)) {
                            cell[0] = (cell[0] - blockDc) * winT[t] * winY[dy] * winX[dx];
                        } else {
                            cell[0] = 0.0;
                            cell[1] = 0.0;
                        }
                    }
                }
            }
        }

        bool chunkResident = false;
#if defined(CHD_HAS_MPSGRAPH_FFT)
        if (useResident) {
            if (@available(macOS 14.0, *)) {
                // FFT → magnitude → conv → mask → IFFT, all in shared Metal
                // buffers. Only download into inStore on full success, so a
                // failure leaves the packed input intact for the FFTW fallback.
                //
                // residentMps() is a single shared instance whose MTLBuffers are
                // per-frame scratch, so the whole sequence runs under runMutex.
                // Concurrent DecoderPool workers would otherwise clobber each
                // other's spectrum between steps. (The GPU serialises this work
                // regardless; this is an opt-in, conv-bound path.)
                ResidentMps &mps = residentMps();
                std::lock_guard<std::mutex> runLock(runMutex);
                mps.uploadTiles(inStore.data(), chunkN);
                bool ok = mps.forwardFFT(chunkN) && mps.magnitude(chunkN, static_cast<float>(invScale));
                if (ok) {
                    // Pad the model-input buffer to the fixed kBatchBlocks batch
                    // (see the Layer-1 note); the FFT/kernels still run at chunkN.
                    if (chunkN < kBatchBlocks) {
                        std::memset(mps.modelInputContents() + static_cast<size_t>(chunkN) * 2 * kBlockSize,
                                    0, static_cast<size_t>(kBatchBlocks - chunkN) * 2 * kBlockSize * sizeof(float));
                    }
                    ok = engine.predictResident(mps.modelInputContents(),
                                                { kBatchBlocks, 2, kNt, kNy, kNx },
                                                mps.maskContents(),
                                                { kBatchBlocks, 1, kNt, kNy, kNx });
                }
                ok = ok && mps.applyMask(chunkN) && mps.inverseFFT(chunkN);
                if (ok) {
                    mps.downloadIfft(inStore.data(), chunkN);
                    chunkResident = true;
                } else {
                    chd::log::warn() << "nnTransform3D[CoreML]: resident path failed; using FFTW for this frame";
                    useResident = false;   // stop trying for the rest of the frame
                    specStore.resize(static_cast<size_t>(kBatchBlocks) * kBlockSize * 2);
                    modelInput.resize(static_cast<size_t>(kBatchBlocks) * 2 * kBlockSize);
                    specBatch = reinterpret_cast<fftw_complex *>(specStore.data());
                    if (plans.forward == nullptr || plans.inverse == nullptr) return false;
                }
            }
        }
#endif

        if (!chunkResident) {
            // ── Layer 1: CPU FFTW + host magnitude + CoreML conv + host mask ──
            for (int j = 0; j < chunkN; ++j) {
                fftw_execute_dft(plans.forward, inBatch + j * kBlockSize, specBatch + j * kBlockSize);
            }
            for (int j = 0; j < chunkN; ++j) {
                const fftw_complex *spec = specBatch + static_cast<size_t>(j) * kBlockSize;
                float *mag = modelInput.data() + static_cast<size_t>(j) * 2 * kBlockSize;
                float *ref = mag + kBlockSize;
                for (int i = 0; i < kBlockSize; ++i) {
                    mag[i] = static_cast<float>(std::sqrt(spec[i][0] * spec[i][0] +
                                                          spec[i][1] * spec[i][1]) * invScale);
                }
                int reflectedIndex = 0;
                for (int ft = 0; ft < kNt; ++ft) {
                    const int refT = ((2 - ft) % kNt + kNt) % kNt;
                    for (int fy = 0; fy < kNy; ++fy) {
                        const int refY = (kNy - fy) % kNy;
                        for (int fx = 0; fx < kNx; ++fx) {
                            const int refX = ((kNx / 2) - fx + kNx) % kNx;
                            ref[reflectedIndex++] = mag[idx3(refT, refY, refX)];
                        }
                    }
                }
            }

            // Inference always runs at the fixed kBatchBlocks batch the model
            // is converted for — a fixed CoreML input shape lets the GPU
            // parallelise the conv across the batch (~30x vs a dynamic batch).
            // Conv is per-batch-element independent, so zero-padding the last
            // partial chunk leaves the real tiles' results unchanged.
            if (chunkN < kBatchBlocks) {
                std::fill(modelInput.begin() + static_cast<size_t>(chunkN) * 2 * kBlockSize,
                          modelInput.end(), 0.0f);
            }
            std::unique_ptr<chd::nn::RunResult> result;
            {
                std::lock_guard<std::mutex> runLock(runMutex);
                try {
                    chd::nn::TensorSpec input;
                    input.data  = modelInput.data();
                    input.shape = { kBatchBlocks, 2, kNt, kNy, kNx };
                    result = engine.run({ input }, {});
                } catch (const std::exception &e) {
                    chd::log::warn() << "nnTransform3D[CoreML] inference failed; falling back to 2D chroma:"
                                     << e.what();
                    return false;
                }
            }
            if (result == nullptr || result->count() == 0) return false;
            const float *mask = result->data(0);

            for (int j = 0; j < chunkN; ++j) {
                fftw_complex *spec = specBatch + static_cast<size_t>(j) * kBlockSize;
                const float *m = mask + static_cast<size_t>(j) * kBlockSize;
                for (int i = 0; i < kBlockSize; ++i) {
                    spec[i][0] *= m[i];
                    spec[i][1] *= m[i];
                }
            }
            for (int j = 0; j < chunkN; ++j) {
                fftw_execute_dft(plans.inverse, specBatch + j * kBlockSize, inBatch + j * kBlockSize);
            }
        }

        // ── Overlap-add into the chroma accumulators (reads inStore) ──────────
        for (int j = 0; j < chunkN; ++j) {
            const int32_t y = ledgerY[chunkStart + j];
            const int32_t x = ledgerX[chunkStart + j];
            const fftw_complex *tile = inBatch + static_cast<size_t>(j) * kBlockSize;
            for (int32_t t = 0; t < kNt; ++t) {
                const bool oddField = (t % 2) != 0;
                std::vector<std::vector<double>> &acc    = *accFor[t / 2];
                std::vector<std::vector<double>> &weight = *weightFor[t / 2];
                for (int32_t dy = 0; dy < kNy; ++dy) {
                    const int32_t absY = y + dy;
                    if (absY < firstActiveY || absY > lastActiveY) continue;
                    if (((absY % 2) != 0) != oddField) continue;
                    for (int32_t dx = 0; dx < kNx; ++dx) {
                        const int32_t absX = x + dx;
                        if (absX < activeStartX || absX >= activeEndX) continue;
                        const double value  = tile[idx3(t, dy, dx)][0] / static_cast<double>(kBlockSize);
                        const double w      = winT[t] * winY[dy] * winX[dx];
                        acc[absY][absX]    += value * w;
                        weight[absY][absX] += w * w;
                    }
                }
            }
        }
    }

    return true;
}

}  // namespace chd::decoders::nntransform3d
