// SPDX-License-Identifier: GPL-3.0-or-later
//
// Smoke test: instantiate the ldzeug2 NN decoders, bind ONNX sessions (the
// committed synthetic fixture by default, or real ldzeug2 weights when the
// matching CHD_TEST_LDZEUG_* env var points at them), configure for NTSC, and
// verify the wiring — including binding a session built from an in-memory
// buffer. No inference is run, so the model's contents don't matter. Like the
// nnTransform3D test, full per-frame decode validation against a golden
// reference is deferred to a follow-up.
//
// Skips gracefully only when the build was made without NN support, or, in the
// unlikely case the committed fixture is missing, when no model can be located.

#include <chromadec/version.h>
#include <chromadec/video.h>   // chd_shutdown

#include "nn_test_model.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__has_include)
#  if !__has_include(<onnxruntime_cxx_api.h>)
#    define CHD_TEST_NO_ONNXRUNTIME
#  endif
#endif

#ifndef CHD_TEST_NO_ONNXRUNTIME
#include "../../src/decoders/ldzeug/ldzeug_base.h"
#include "../../src/decoders/ldzeug/ldzeug_color_cnn.h"
#include "../../src/decoders/ldzeug/ldzeug_luma_sep.h"
#include "../../src/metadata/core.h"
#include "../../src/decoders/source_field.h"
#include "../../src/nn/inference_engine.h"
#include "../../src/nn/ort_engine.h"
#include "../../src/nn/ort_session.h"
#include "../../src/nn/provider_select.h"
#include "../../src/output/component_frame.h"
#endif

namespace {

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        return 1; \
    } \
} while (0)

#ifndef CHD_TEST_NO_ONNXRUNTIME

// SessionOptions honouring CHD_TEST_NN_BACKEND, so a GPU job can validate the
// execution provider it exists to prove rather than whatever the AUTO chain
// reached first.
chd::nn::SessionOptions testSessionOptions() {
    chd::nn::SessionOptions opts;
    if (const int b = chd_test::backendFromEnv(); b != 0) {
        opts.requestedProvider = static_cast<chd_nn_backend_t>(b);
    }
    return opts;
}

// Real ldzeug2 weights, when available, are supplied via environment variables
// — never hardcoded. Each lookup falls back to the committed synthetic fixture
// so these tests run unconditionally:
//   color_cnn:        $CHD_TEST_LDZEUG_COLOR_CNN_MODEL
//   luma_sep field:   $CHD_TEST_LDZEUG_LUMA_SEP_FIELD_MODEL
//   luma_sep frame:   $CHD_TEST_LDZEUG_LUMA_SEP_FRAME_MODEL
chd::metadata::LdDecodeMetaData::VideoParameters makeNtscVp() {
    chd::metadata::LdDecodeMetaData::VideoParameters vp;
    vp.system = chd::metadata::NTSC;
    vp.fieldWidth  = 910;
    vp.fieldHeight = 263;
    vp.fSC = 315.0e6 / 88.0;
    vp.sampleRate = vp.fSC * 4.0;
    vp.activeVideoStart = 134;
    vp.activeVideoEnd   = 894;
    vp.firstActiveFrameLine = 39;
    vp.lastActiveFrameLine  = 524;
    vp.black16bIre    = 16128;
    vp.white16bIre    = 51200;
    vp.blanking16bIre = 15360;
    vp.isValid = true;
    return vp;
}

// Bind a color_cnn decoder to `session`, configure for NTSC, and assert the
// wiring (no temporal context). Shared by the file- and memory-load variants.
int bindColorCnnAndConfigure(const std::shared_ptr<chd::nn::OrtSession> &session) {
    using namespace chd::decoders::ldzeug;
    LdzeugColorCnnDecoder decoder;
    decoder.setNnModel(std::make_shared<chd::nn::OrtEngine>(session));
    decoder.setMode(LdzeugDecoderBase::Mode::Field);
    decoder.setChromaPhase(0.0);
    decoder.setChromaGain(1.0);

    REQUIRE(decoder.configure(makeNtscVp()));
    // ldzeug decoders are per-field/frame and need no temporal context.
    REQUIRE(decoder.getLookBehind() == 0);
    REQUIRE(decoder.getLookAhead()  == 0);
    return 0;
}

int testColorCnnConfigure() {
    const std::string model = chd_test::modelFromEnvOrFixture("CHD_TEST_LDZEUG_COLOR_CNN_MODEL");
    if (model.empty()) {
        std::cout << "Skipping LdzeugColorCnn test (no model fixture available).\n";
        return 0;
    }

    // From a file path.
    chd::nn::SessionOptions opts = testSessionOptions();
    std::shared_ptr<chd::nn::OrtSession> fileSession;
    try {
        fileSession = std::make_shared<chd::nn::OrtSession>(model, opts);
    } catch (const std::exception &e) {
        std::cerr << "FAIL: OrtSession(color_cnn) from file failed: " << e.what() << "\n";
        return 1;
    }
    REQUIRE(fileSession != nullptr);
    if (int rc = bindColorCnnAndConfigure(fileSession)) return rc;

    // From an in-memory buffer — exercises the in-memory OrtSession
    // constructor (the path chd_nn_model_load_from_memory forwards to) for an
    // ldzeug decoder too.
    std::vector<char> bytes = chd_test::readFileBytes(model);
    REQUIRE(!bytes.empty());
    std::shared_ptr<chd::nn::OrtSession> memSession;
    try {
        memSession = std::make_shared<chd::nn::OrtSession>(bytes.data(), bytes.size(), opts);
    } catch (const std::exception &e) {
        std::cerr << "FAIL: OrtSession(color_cnn) from memory failed: " << e.what() << "\n";
        return 1;
    }
    REQUIRE(memSession != nullptr);
    if (int rc = bindColorCnnAndConfigure(memSession)) return rc;

    std::cout << "LdzeugColorCnn bound from file + memory (" << bytes.size()
              << " bytes) via " << model << "\n";
    return 0;
}

int testLumaSepConfigure() {
    using namespace chd::decoders::ldzeug;
    const std::string fieldModel =
        chd_test::modelFromEnvOrFixture("CHD_TEST_LDZEUG_LUMA_SEP_FIELD_MODEL");
    const std::string frameModel =
        chd_test::modelFromEnvOrFixture("CHD_TEST_LDZEUG_LUMA_SEP_FRAME_MODEL");
    if (fieldModel.empty() || frameModel.empty()) {
        std::cout << "Skipping LdzeugLumaSep test (no model fixture available).\n";
        return 0;
    }

    {
        chd::nn::SessionOptions opts = testSessionOptions();
        auto session = std::make_shared<chd::nn::OrtSession>(fieldModel, opts);
        LdzeugLumaSepDecoder decoder;
        decoder.setNnModel(std::make_shared<chd::nn::OrtEngine>(session));
        decoder.setMode(LdzeugDecoderBase::Mode::Field);
        REQUIRE(decoder.configure(makeNtscVp()));
        REQUIRE(decoder.getLookAhead() == 0);
        // chromaBandpass should default to true (matches ldzeug2 default).
        decoder.setChromaBandpass(false);
        decoder.setChromaBandpass(true);
        std::cout << "LdzeugLumaSep (field) bound from " << fieldModel << "\n";
    }

    {
        chd::nn::SessionOptions opts = testSessionOptions();
        auto session = std::make_shared<chd::nn::OrtSession>(frameModel, opts);
        LdzeugLumaSepDecoder decoder;
        decoder.setNnModel(std::make_shared<chd::nn::OrtEngine>(session));
        decoder.setMode(LdzeugDecoderBase::Mode::Frame);
        REQUIRE(decoder.configure(makeNtscVp()));
        std::cout << "LdzeugLumaSep (frame) bound from " << frameModel << "\n";
    }

    return 0;
}

int testNoSessionRejected() {
    using namespace chd::decoders::ldzeug;
    LdzeugColorCnnDecoder color;
    REQUIRE(color.configure(makeNtscVp()));
    std::vector<chd::decoders::SourceField> inputs;
    std::vector<chd::output::ComponentFrame> outputs;
    bool threw = false;
    try {
        color.decodeFrames(inputs, 0, 0, outputs);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    REQUIRE(threw);

    LdzeugLumaSepDecoder lumaSep;
    REQUIRE(lumaSep.configure(makeNtscVp()));
    threw = false;
    try {
        lumaSep.decodeFrames(inputs, 0, 0, outputs);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    REQUIRE(threw);
    return 0;
}

int testRejectsNonNtsc() {
    using namespace chd::decoders::ldzeug;
    chd::metadata::LdDecodeMetaData::VideoParameters palVp = makeNtscVp();
    palVp.system = chd::metadata::PAL;

    LdzeugColorCnnDecoder color;
    REQUIRE(!color.configure(palVp));

    LdzeugLumaSepDecoder lumaSep;
    REQUIRE(!lumaSep.configure(palVp));
    return 0;
}

// An engine that echoes the requested shape, with any dimension replaced by a
// non-negative override, the way a stale compiled program would. Its buffer
// matches the shape it reports, so reading past it is caught under ASan.
class FakeEngine : public chd::nn::InferenceEngine {
public:
    struct Overrides {
        int64_t height = -1;
        int64_t width = -1;
        bool dropChannelDim = false;
    };
    explicit FakeEngine(Overrides o) : overrides_(o) {}

    std::unique_ptr<chd::nn::RunResult> run(
        const std::vector<chd::nn::TensorSpec> &inputs,
        const std::vector<std::string> & = {}) override {
        std::vector<int64_t> shape = inputs.at(0).shape;
        if (overrides_.height >= 0) shape[2] = overrides_.height;
        if (overrides_.width >= 0) shape[3] = overrides_.width;
        if (overrides_.dropChannelDim) shape.erase(shape.begin() + 1);
        return std::make_unique<Result>(std::move(shape));
    }
    chd_nn_backend_t activeBackend() const override { return CHD_NN_ORT_CPU; }
    const std::vector<std::string> &inputNames() const override { return names_; }
    const std::vector<std::string> &outputNames() const override { return names_; }

private:
    class Result : public chd::nn::RunResult {
    public:
        explicit Result(std::vector<int64_t> shape) : shape_(std::move(shape)) {
            size_t n = 1;
            for (const int64_t d : shape_) n *= static_cast<size_t>(d);
            data_.resize(n);
        }
        size_t count() const override { return 1; }
        const float *data(size_t) const override { return data_.data(); }
        std::vector<int64_t> shape(size_t) const override { return shape_; }

    private:
        std::vector<int64_t> shape_;
        std::vector<float> data_;
    };

    Overrides overrides_;
    std::vector<std::string> names_;
};

// Decode one frame through a FakeEngine; true if the decoder threw.
template <typename DecoderT>
bool decodeThrows(chd::decoders::ldzeug::LdzeugDecoderBase::Mode mode,
                  FakeEngine::Overrides overrides) {
    const auto vp = makeNtscVp();
    DecoderT decoder;
    decoder.setNnModel(std::make_shared<FakeEngine>(overrides));
    decoder.setMode(mode);
    if (!decoder.configure(vp)) throw std::logic_error("configure failed");

    std::vector<chd::decoders::SourceField> fields(2);
    for (int i = 0; i < 2; ++i) {
        fields[i].field.isFirstField = (i == 0);
        fields[i].field.fieldPhaseID = i + 1;
        fields[i].data.assign(static_cast<size_t>(vp.fieldWidth) * vp.fieldHeight, 0);
    }
    std::vector<chd::output::ComponentFrame> frames(1);
    try {
        decoder.decodeFrames(fields, 0, 2, frames);
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

// Every run site: the matching shape decodes, each wrong dimension is refused.
template <typename DecoderT>
int checkOutputValidation() {
    using Mode = chd::decoders::ldzeug::LdzeugDecoderBase::Mode;
    for (const Mode mode : { Mode::Field, Mode::Frame }) {
        // A stale program can return the other mode's height (263 or 526).
        const int64_t otherModeHeight = (mode == Mode::Field) ? 526 : 263;
        REQUIRE(!decodeThrows<DecoderT>(mode, {}));
        REQUIRE(decodeThrows<DecoderT>(mode, { otherModeHeight, -1, false }));
        REQUIRE(decodeThrows<DecoderT>(mode, { 240, -1, false }));
        REQUIRE(decodeThrows<DecoderT>(mode, { -1, 1135, false }));
        REQUIRE(decodeThrows<DecoderT>(mode, { -1, -1, true }));
    }
    return 0;
}

int testOutputShapeValidated() {
    using namespace chd::decoders::ldzeug;
    if (checkOutputValidation<LdzeugColorCnnDecoder>() != 0) return 1;
    return checkOutputValidation<LdzeugLumaSepDecoder>();
}

int testBuiltinMIGraphXCacheDir() {
    chd::nn::EngineCacheConfig cache;

    cache.dir = "/cache/default";
    cache.dirIsLibraryDefault = true;
    REQUIRE(chd::nn::builtinMIGraphXCacheDir(cache).empty());

    cache.dir = "/cache/910x263";
    cache.dirIsLibraryDefault = false;
    REQUIRE(chd::nn::builtinMIGraphXCacheDir(cache) == "/cache/910x263");

    cache.dir.clear();
    REQUIRE(chd::nn::builtinMIGraphXCacheDir(cache).empty());
    return 0;
}

#endif  // CHD_TEST_NO_ONNXRUNTIME

}  // namespace

int main() {
    if (chd_has_feature("nn") != 1) {
        std::cout << "Skipping ldzeug tests (with_nn=false).\n";
        return 0;
    }
#ifdef CHD_TEST_NO_ONNXRUNTIME
    std::cout << "Skipping ldzeug tests (no ORT headers visible at test compile).\n";
    return 0;
#else
    int rc = 0;
    rc |= testNoSessionRejected();
    rc |= testRejectsNonNtsc();
    rc |= testColorCnnConfigure();
    rc |= testLumaSepConfigure();
    rc |= testOutputShapeValidated();
    rc |= testBuiltinMIGraphXCacheDir();
    // Tear down the Ort::Env explicitly before C++ static destruction
    // (avoids the static-destruction-order hazard).
    chd_shutdown();
    if (rc == 0) std::cout << "All ldzeug wiring tests passed.\n";
    return rc;
#endif
}
