// SPDX-License-Identifier: GPL-3.0-or-later
//
// Objective-C++ implementation of CoreMLEngine. See coreml_engine.h.

#include "coreml_engine.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <mutex>
#include <stdexcept>
#include <string>

#include "../common/log.h"

namespace chd::nn {

namespace {

MLComputeUnits toMLComputeUnits(CoreMLComputeUnits units) {
    switch (units) {
        case CoreMLComputeUnits::CpuAndGpu: return MLComputeUnitsCPUAndGPU;
        case CoreMLComputeUnits::All:       return MLComputeUnitsAll;
        case CoreMLComputeUnits::CpuOnly:   return MLComputeUnitsCPUOnly;
    }
    return MLComputeUnitsCPUAndGPU;
}

MLModel *buildMLModel(NSURL *url, CoreMLComputeUnits units, NSError **error) {
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = toMLComputeUnits(units);
    return [MLModel modelWithContentsOfURL:url configuration:config error:error];
}

// C-contiguous strides (in elements) for a shape, as an NSArray<NSNumber*>.
NSArray<NSNumber *> *contiguousStrides(const std::vector<int64_t> &shape) {
    const size_t n = shape.size();
    std::vector<int64_t> strides(n, 1);
    for (size_t i = n; i-- > 1;) {
        strides[i - 1] = strides[i] * shape[i];
    }
    NSMutableArray<NSNumber *> *out = [NSMutableArray arrayWithCapacity:n];
    for (size_t i = 0; i < n; ++i) [out addObject:@(strides[i])];
    return out;
}

NSArray<NSNumber *> *shapeToNSArray(const std::vector<int64_t> &shape) {
    NSMutableArray<NSNumber *> *out = [NSMutableArray arrayWithCapacity:shape.size()];
    for (int64_t d : shape) [out addObject:@(d)];
    return out;
}

}  // namespace

// ─── Impl ───────────────────────────────────────────────────────────────────

struct CoreMLEngine::Impl {
    MLModel *model = nil;
    // Cached feature names as NSStrings for prediction wiring.
    NSArray<NSString *> *inputFeatureNames  = nil;
    NSArray<NSString *> *outputFeatureNames = nil;

    // Compiled (.mlmodelc) URL, kept so the Layer-2 resident path can build a
    // second cpuAndGPU model on demand without recompiling.
    NSURL     *compiledURL   = nil;
    MLModel   *residentModel = nil;   // lazily built, cpuAndGPU
    std::mutex residentMutex;
    std::mutex modelMutex;            // guards model + units_ on fallback
};

// ─── RunResult ────────────────────────────────────────────────────────────────

namespace {

// Owns contiguous float32 copies of one prediction's outputs. CoreML output
// MLMultiArrays may be fp16 (ANE) / fp64 as well as fp32, so we normalise to
// float here and present them through the RunResult interface.
class CoreMLRunResult : public RunResult {
public:
    struct Tensor {
        std::vector<float>   data;
        std::vector<int64_t> shape;
    };

    explicit CoreMLRunResult(std::vector<Tensor> tensors)
        : tensors_(std::move(tensors)) {}

    size_t count() const override { return tensors_.size(); }
    const float *data(size_t index) const override { return tensors_[index].data.data(); }
    std::vector<int64_t> shape(size_t index) const override { return tensors_[index].shape; }

private:
    std::vector<Tensor> tensors_;
};

// Copy an MLMultiArray into a contiguous float32 vector, converting dtype.
// CoreML output MLMultiArrays are NOT necessarily C-contiguous — the framework
// pads strides for alignment at some shapes — so the copy must honour
// arr.strides rather than reading arr.count elements linearly. (Reading them
// linearly silently scrambles the output at the padded shapes while looking
// fine at the contiguous ones.)
CoreMLRunResult::Tensor copyMultiArray(MLMultiArray *arr) {
    CoreMLRunResult::Tensor t;
    const size_t rank = arr.shape.count;
    std::vector<int64_t> shape(rank), strides(rank);   // strides in elements
    for (size_t i = 0; i < rank; ++i) {
        shape[i]   = arr.shape[i].longLongValue;
        strides[i] = arr.strides[i].longLongValue;
    }
    t.shape.assign(shape.begin(), shape.end());

    const NSInteger n = arr.count;                     // logical element count
    t.data.resize(static_cast<size_t>(n));

    const void *src = arr.dataPointer;
    auto read = [&](int64_t off) -> float {
        switch (arr.dataType) {
            case MLMultiArrayDataTypeFloat32: return static_cast<const float  *>(src)[off];
            case MLMultiArrayDataTypeDouble:  return static_cast<float>(static_cast<const double *>(src)[off]);
            case MLMultiArrayDataTypeFloat16: return static_cast<float>(static_cast<const __fp16 *>(src)[off]);
            default:
                throw std::runtime_error("CoreMLEngine: unsupported output MLMultiArray dataType");
        }
    };

    // Fast path when the array is already C-contiguous (common, e.g. the
    // nnTransform3D tile output): bulk per-dtype copy, no per-element dispatch.
    bool contiguous = true;
    int64_t expect = 1;
    for (size_t i = rank; i-- > 0; ) {
        if (strides[i] != expect) { contiguous = false; break; }
        expect *= shape[i];
    }
    if (contiguous) {
        switch (arr.dataType) {
            case MLMultiArrayDataTypeFloat32:
                std::copy(static_cast<const float *>(src),
                          static_cast<const float *>(src) + n, t.data.begin());
                break;
            case MLMultiArrayDataTypeDouble: {
                const double *p = static_cast<const double *>(src);
                for (NSInteger i = 0; i < n; ++i) t.data[static_cast<size_t>(i)] = static_cast<float>(p[i]);
                break;
            }
            case MLMultiArrayDataTypeFloat16: {
                const __fp16 *p = static_cast<const __fp16 *>(src);
                for (NSInteger i = 0; i < n; ++i) t.data[static_cast<size_t>(i)] = static_cast<float>(p[i]);
                break;
            }
            default:
                throw std::runtime_error("CoreMLEngine: unsupported output MLMultiArray dataType");
        }
        return t;
    }

    // Strided gather: walk logical row-major indices, map through strides.
    std::vector<int64_t> idx(rank, 0);
    for (NSInteger lin = 0; lin < n; ++lin) {
        int64_t off = 0;
        for (size_t d = 0; d < rank; ++d) off += idx[d] * strides[d];
        t.data[static_cast<size_t>(lin)] = read(off);
        for (size_t d = rank; d-- > 0; ) {             // increment, last dim fastest
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
    return t;
}

}  // namespace

// ─── Construction ─────────────────────────────────────────────────────────────

CoreMLEngine::CoreMLEngine(const std::string &modelPath, CoreMLComputeUnits units)
    : impl_(std::make_unique<Impl>()), units_(units)
{
    @autoreleasepool {
        NSError *error = nil;
        NSString *path = [NSString stringWithUTF8String:modelPath.c_str()];
        NSURL *url = [NSURL fileURLWithPath:path];

        // A `.mlpackage` must be compiled to a `.mlmodelc` before loading; a
        // path that is already `.mlmodelc` loads directly. Compile when the
        // extension isn't the compiled form.
        NSURL *loadURL = url;
        if (![[path pathExtension] isEqualToString:@"mlmodelc"]) {
            NSURL *compiled = [MLModel compileModelAtURL:url error:&error];
            if (compiled == nil) {
                throw std::runtime_error(
                    std::string("CoreMLEngine: compile failed: ") +
                    (error ? error.localizedDescription.UTF8String : "unknown error"));
            }
            loadURL = compiled;
        }

        impl_->compiledURL = loadURL;
        impl_->model = buildMLModel(loadURL, units, &error);
        if (impl_->model == nil) {
            throw std::runtime_error(
                std::string("CoreMLEngine: model load failed: ") +
                (error ? error.localizedDescription.UTF8String : "unknown error"));
        }

        MLModelDescription *desc = impl_->model.modelDescription;
        impl_->inputFeatureNames  = desc.inputDescriptionsByName.allKeys;
        impl_->outputFeatureNames = desc.outputDescriptionsByName.allKeys;
        for (NSString *n in impl_->inputFeatureNames)  inputNames_.emplace_back(n.UTF8String);
        for (NSString *n in impl_->outputFeatureNames) outputNames_.emplace_back(n.UTF8String);
    }
}

CoreMLEngine::~CoreMLEngine() = default;

// ─── run ────────────────────────────────────────────────────────────────────

std::unique_ptr<RunResult> CoreMLEngine::run(
    const std::vector<TensorSpec>  &inputs,
    const std::vector<std::string> &outputNames)
{
    @autoreleasepool {
        NSError *error = nil;
        NSMutableDictionary<NSString *, MLFeatureValue *> *features =
            [NSMutableDictionary dictionaryWithCapacity:inputs.size()];

        for (size_t i = 0; i < inputs.size(); ++i) {
            const TensorSpec &t = inputs[i];
            NSString *name = t.name.empty()
                ? impl_->inputFeatureNames[i]
                : [NSString stringWithUTF8String:t.name.c_str()];

            // Wrap the caller's host buffer without copying — it stays alive
            // for the duration of this synchronous prediction.
            MLMultiArray *arr =
                [[MLMultiArray alloc] initWithDataPointer:const_cast<float *>(t.data)
                                                    shape:shapeToNSArray(t.shape)
                                                 dataType:MLMultiArrayDataTypeFloat32
                                                  strides:contiguousStrides(t.shape)
                                              deallocator:^(void *){ /* not owned */ }
                                                    error:&error];
            if (arr == nil) {
                throw std::runtime_error(
                    std::string("CoreMLEngine: failed to wrap input '") +
                    name.UTF8String + "': " +
                    (error ? error.localizedDescription.UTF8String : "unknown error"));
            }
            features[name] = [MLFeatureValue featureValueWithMultiArray:arr];
        }

        MLDictionaryFeatureProvider *provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
        if (provider == nil) {
            throw std::runtime_error(
                std::string("CoreMLEngine: feature provider init failed: ") +
                (error ? error.localizedDescription.UTF8String : "unknown error"));
        }

        MLModel *model;
        {
            std::lock_guard<std::mutex> lock(impl_->modelMutex);
            model = impl_->model;
        }
        id<MLFeatureProvider> out = [model predictionFromFeatures:provider error:&error];
        if (out == nil) {
            // One-shot graceful fallback to CPU-only: some GPU/driver/model
            // combinations fail to prepare or run on the GPU (e.g. the ANE
            // can't take a 3D conv); the CPU path always works.
            std::lock_guard<std::mutex> lock(impl_->modelMutex);
            if (units_ != CoreMLComputeUnits::CpuOnly) {
                chd::log::warn() << "CoreMLEngine: predict failed under current compute units;"
                                    " falling back to CPU-only:"
                                 << (error ? error.localizedDescription.UTF8String : "unknown");
                NSError *err2 = nil;
                MLModel *cpu = buildMLModel(impl_->compiledURL, CoreMLComputeUnits::CpuOnly, &err2);
                if (cpu != nil) {
                    impl_->model = cpu;
                    units_ = CoreMLComputeUnits::CpuOnly;
                    outputGeneration_.fetch_add(1, std::memory_order_release);
                    error = nil;
                    out = [cpu predictionFromFeatures:provider error:&error];
                }
            }
            if (out == nil) {
                throw std::runtime_error(
                    std::string("CoreMLEngine: prediction failed: ") +
                    (error ? error.localizedDescription.UTF8String : "unknown error"));
            }
        }

        // Resolve which outputs to fetch (declared order, or the caller's list).
        NSArray<NSString *> *wanted = impl_->outputFeatureNames;
        if (!outputNames.empty()) {
            NSMutableArray<NSString *> *names =
                [NSMutableArray arrayWithCapacity:outputNames.size()];
            for (const auto &n : outputNames) [names addObject:[NSString stringWithUTF8String:n.c_str()]];
            wanted = names;
        }

        std::vector<CoreMLRunResult::Tensor> tensors;
        tensors.reserve(wanted.count);
        for (NSString *name in wanted) {
            MLFeatureValue *val = [out featureValueForName:name];
            MLMultiArray *arr = val.multiArrayValue;
            if (arr == nil) {
                throw std::runtime_error(std::string("CoreMLEngine: output '") +
                                         name.UTF8String + "' is not a multi-array");
            }
            tensors.push_back(copyMultiArray(arr));
        }

        return std::make_unique<CoreMLRunResult>(std::move(tensors));
    }
}

// ─── predictResident (Layer 2 GPU-resident path) ─────────────────────────────

bool CoreMLEngine::predictResident(const float *input, const std::vector<int64_t> &inputShape,
                                   float *output, const std::vector<int64_t> &outputShape)
{
    @autoreleasepool {
        NSError *error = nil;

        // Lazily build a cpuAndGPU model (the resident path forbids the ANE,
        // which can't share an arbitrary Metal buffer). Reuses the already
        // compiled .mlmodelc so this doesn't recompile.
        MLModel *model = nil;
        {
            std::lock_guard<std::mutex> lock(impl_->residentMutex);
            if (impl_->residentModel == nil) {
                if (impl_->compiledURL == nil) return false;
                impl_->residentModel = buildMLModel(impl_->compiledURL,
                                                    CoreMLComputeUnits::CpuAndGpu, &error);
                if (impl_->residentModel == nil) {
                    chd::log::warn() << "CoreMLEngine: resident (cpuAndGPU) model build failed:"
                                     << (error ? error.localizedDescription.UTF8String : "unknown");
                    return false;
                }
            }
            model = impl_->residentModel;
        }

        NSString *inName  = impl_->inputFeatureNames[0];
        NSString *outName = impl_->outputFeatureNames[0];

        // Wrap the caller's buffers in place — no copy. On unified memory these
        // point at the same bytes the MPSGraph FFT / Metal kernels touch.
        MLMultiArray *inArr =
            [[MLMultiArray alloc] initWithDataPointer:const_cast<float *>(input)
                                                shape:shapeToNSArray(inputShape)
                                             dataType:MLMultiArrayDataTypeFloat32
                                              strides:contiguousStrides(inputShape)
                                          deallocator:^(void *){ /* not owned */ }
                                                error:&error];
        MLMultiArray *outArr =
            [[MLMultiArray alloc] initWithDataPointer:output
                                                shape:shapeToNSArray(outputShape)
                                             dataType:MLMultiArrayDataTypeFloat32
                                              strides:contiguousStrides(outputShape)
                                          deallocator:^(void *){ /* not owned */ }
                                                error:&error];
        if (inArr == nil || outArr == nil) return false;

        MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{ inName : [MLFeatureValue featureValueWithMultiArray:inArr] }
                         error:&error];
        if (provider == nil) return false;

        // outputBackings makes CoreML write the result straight into outArr
        // (our buffer) instead of allocating + handing back a fresh array.
        MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
        options.outputBackings = @{ outName : outArr };

        id<MLFeatureProvider> result = [model predictionFromFeatures:provider
                                                             options:options
                                                               error:&error];
        if (result == nil) {
            chd::log::warn() << "CoreMLEngine: resident prediction failed:"
                             << (error ? error.localizedDescription.UTF8String : "unknown");
            return false;
        }
        return true;
    }
}

}  // namespace chd::nn
