// SPDX-License-Identifier: GPL-3.0-or-later
//
// Smoke test: instantiate NtscDecoder with nnTransform3D=true, bind an ONNX
// session (the synthetic test fixture by default, or real chroma_net weights
// when $CHD_TEST_NN_MODEL points at them), configure for NTSC, and verify the
// wiring (look-ahead, configure success, no crash) — both when the
// session is built from a file path and from an in-memory buffer. Those checks
// run no inference, so the model's contents don't matter. One further test runs
// real tile passes on a small crop with a mask-shaped synthetic model. Full
// frame-decode validation against a golden reference is deferred to a follow-up
// once an encode-orc fixture pipeline exists.
//
// Skips gracefully when the build was made without NN support
// (chd_has_feature("nn") == 0), or, in the unlikely case the committed
// fixture is missing, when no model can be located at all.

#include <chromadec/version.h>
#include <chromadec/video.h>   // chd_shutdown

#include "nn_test_model.h"

#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#if defined(__has_include)
#  if !__has_include(<onnxruntime_cxx_api.h>)
#    define CHD_TEST_NO_ONNXRUNTIME
#  endif
#endif

#ifndef CHD_TEST_NO_ONNXRUNTIME
// Internal headers — we link the static lib for test access (matches the
// existing test_encode_orc_color_bars pattern).
#include "../../src/decoders/comb/comb.h"
#include "../../src/decoders/comb/ntsc_decoder.h"
#include "../../src/decoders/nntransform3d/nntransform3d_window.h"
#include "../../src/decoders/source_field.h"
#include "../../src/metadata/core.h"
#include "../../src/nn/ort_engine.h"
#include "../../src/output/component_frame.h"
#include "../../src/nn/ort_session.h"
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

int testLookAround() {
    using namespace chd::decoders::comb;
    Comb::Configuration cfg;
    cfg.dimensions = 3;
    cfg.nnTransform3D = false;
    REQUIRE(cfg.getLookAhead() == 1);
    REQUIRE(cfg.getLookBehind() == 1);

    // Tiles span two frames: (N-1,N) and (N,N+1).
    cfg.nnTransform3D = true;
    REQUIRE(cfg.getLookAhead() == 1);
    REQUIRE(cfg.getLookBehind() == 1);
    return 0;
}

int testConfigureWithoutSession() {
    using namespace chd::decoders::comb;
    Comb::Configuration cfg;
    cfg.dimensions = 3;
    cfg.nnTransform3D = true;
    cfg.nnInputMagnitudeScale = 128.0;

    NtscDecoder decoder(cfg);
    chd::metadata::LdDecodeMetaData::VideoParameters vp;
    vp.system = chd::metadata::NTSC;
    vp.fieldWidth = 910;
    vp.fieldHeight = 263;
    vp.fSC = 315.0e6 / 88.0;
    vp.sampleRate = vp.fSC * 4.0;
    vp.activeVideoStart = 134;
    vp.activeVideoEnd = 894;
    vp.firstActiveFrameLine = 39;
    vp.lastActiveFrameLine = 524;
    vp.black16bIre = 16128;
    vp.white16bIre = 51200;
    vp.blanking16bIre = 15360;
    vp.isValid = true;

    // Configure without binding a session: should still succeed.
    // decodeFrames would fall back to 2D (we don't exercise it here
    // because nnTransform3D=true requires real input frames for the
    // unique pathway).
    REQUIRE(decoder.configure(vp));
    REQUIRE(decoder.getLookAhead() == 1);
    return 0;
}

// Build an nnTransform3D NtscDecoder, bind `session`, configure it for NTSC,
// and assert the wiring held (look-ahead + configure success). Shared by
// the file- and memory-load session tests so they verify identical behaviour
// regardless of how the session's model bytes were sourced.
int bindSessionAndConfigure(const std::shared_ptr<chd::nn::OrtSession> &session) {
    using namespace chd::decoders::comb;
    Comb::Configuration cfg;
    cfg.dimensions = 3;
    cfg.nnTransform3D = true;
    cfg.nnInputMagnitudeScale = 128.0;

    NtscDecoder decoder(cfg);
    decoder.setNnModel(std::make_shared<chd::nn::OrtEngine>(session));

    chd::metadata::LdDecodeMetaData::VideoParameters vp;
    vp.system = chd::metadata::NTSC;
    vp.fieldWidth = 910;
    vp.fieldHeight = 263;
    vp.fSC = 315.0e6 / 88.0;
    vp.sampleRate = vp.fSC * 4.0;
    vp.activeVideoStart = 134;
    vp.activeVideoEnd = 894;
    vp.firstActiveFrameLine = 39;
    vp.lastActiveFrameLine = 524;
    vp.black16bIre = 16128;
    vp.white16bIre = 51200;
    vp.blanking16bIre = 15360;
    vp.isValid = true;
    REQUIRE(decoder.configure(vp));
    REQUIRE(decoder.getLookAhead() == 1);
    return 0;
}

int testConfigureWithSession() {
    const std::string modelPath = chd_test::modelFromEnvOrFixture("CHD_TEST_NN_MODEL");
    if (modelPath.empty()) {
        std::cout << "Skipping nnTransform3D session test (no model fixture available).\n";
        return 0;
    }

    chd::nn::SessionOptions opts = testSessionOptions();
    std::shared_ptr<chd::nn::OrtSession> session;
    try {
        session = std::make_shared<chd::nn::OrtSession>(modelPath, opts);
    } catch (const std::exception &e) {
        std::cerr << "FAIL: OrtSession construction from file failed: " << e.what() << "\n";
        return 1;
    }
    REQUIRE(session != nullptr);

    if (int rc = bindSessionAndConfigure(session)) return rc;

    std::cout << "nnTransform3D session bound from file " << modelPath << "\n";
    return 0;
}

int testConfigureWithMemorySession() {
    const std::string modelPath = chd_test::modelFromEnvOrFixture("CHD_TEST_NN_MODEL");
    if (modelPath.empty()) {
        std::cout << "Skipping nnTransform3D in-memory session test (no model fixture available).\n";
        return 0;
    }

    // Read the same model file into a buffer and construct the session from
    // memory, exercising the in-memory OrtSession constructor (the one the C
    // ABI's chd_nn_model_load_from_memory forwards to) end-to-end against a
    // real model — not just the failure paths covered in test_nn_framework.
    std::vector<char> bytes = chd_test::readFileBytes(modelPath);
    REQUIRE(!bytes.empty());

    chd::nn::SessionOptions opts = testSessionOptions();
    std::shared_ptr<chd::nn::OrtSession> session;
    try {
        session = std::make_shared<chd::nn::OrtSession>(bytes.data(), bytes.size(), opts);
    } catch (const std::exception &e) {
        std::cerr << "FAIL: OrtSession construction from memory failed: " << e.what() << "\n";
        return 1;
    }
    REQUIRE(session != nullptr);

    if (int rc = bindSessionAndConfigure(session)) return rc;

    std::cout << "nnTransform3D session bound from memory (" << bytes.size()
              << " bytes) via " << modelPath << "\n";
    return 0;
}

// A small crop keeps the tile count, and so these tests, short.
chd::metadata::LdDecodeMetaData::VideoParameters croppedNtscParameters() {
    chd::metadata::LdDecodeMetaData::VideoParameters vp;
    vp.system = chd::metadata::NTSC;
    vp.fieldWidth = 910;
    vp.fieldHeight = 263;
    vp.fSC = 315.0e6 / 88.0;
    vp.sampleRate = vp.fSC * 4.0;
    vp.activeVideoStart = 300;
    vp.activeVideoEnd = 364;
    vp.pictureStart = vp.tileReadStart = 300;
    vp.tileReadEnd = 364;
    vp.firstActiveFrameLine = 100;
    vp.lastActiveFrameLine = 163;
    vp.black16bIre = 16128;
    vp.white16bIre = 51200;
    vp.blanking16bIre = 15360;
    vp.isValid = true;
    return vp;
}

std::unique_ptr<chd::decoders::comb::NtscDecoder> maskModelDecoder() {
    using namespace chd::decoders::comb;
    Comb::Configuration cfg;
    cfg.dimensions = 3;
    cfg.nnTransform3D = true;
    auto decoder = std::make_unique<NtscDecoder>(cfg);

    chd::nn::SessionOptions opts;
    opts.requestedProvider = CHD_NN_ORT_CPU;
    decoder->setNnModel(std::make_shared<chd::nn::OrtEngine>(std::make_shared<chd::nn::OrtSession>(
        std::string(CHD_TEST_FIXTURE_DIR) + "/tiny_chroma_mask.onnx", opts)));
    return decoder;
}

std::vector<chd::decoders::SourceField> noiseFields(
    size_t count, const chd::metadata::LdDecodeMetaData::VideoParameters &vp) {
    std::vector<chd::decoders::SourceField> fields(count);
    std::minstd_rand noise(1234);
    for (size_t i = 0; i < count; i++) {
        fields[i].field.isFirstField = (i % 2) == 0;
        fields[i].field.fieldPhaseID = static_cast<int32_t>((i % 4) + 1);
        fields[i].data.resize(static_cast<size_t>(vp.fieldWidth) * vp.fieldHeight);
        for (auto &sample : fields[i].data) {
            sample = static_cast<uint16_t>(20000 + (noise() % 20000));
        }
    }
    return fields;
}

// Both tile passes that overlap a frame have to reach it however frames are
// grouped into decode calls, so a one-frame call must match a batched decode.
int testOneFrameCallMatchesBatch() {
    using chd::decoders::SourceField;
    using chd::output::ComponentFrame;

    const auto vp = croppedNtscParameters();
    auto decoder = maskModelDecoder();
    REQUIRE(decoder->configure(vp));

    // Four frames: the frame before, the two decoded, and the one after.
    const auto fields = noiseFields(8, vp);

    std::vector<ComponentFrame> batch(2);
    decoder->decodeFrames(fields, 2, 6, batch);

    const std::vector<SourceField> window(fields.begin() + 2, fields.end());
    std::vector<ComponentFrame> single(1);
    decoder->decodeFrames(window, 2, 4, single);

    REQUIRE(*single[0].getY() == *batch[1].getY());
    REQUIRE(*single[0].getU() == *batch[1].getU());
    REQUIRE(*single[0].getV() == *batch[1].getV());
    return 0;
}

// Tightening the crop only trims the output: the picture inside it is unchanged.
int testCropOnlyTrimsOutput() {
    using chd::output::ComponentFrame;

    const auto wide = croppedNtscParameters();
    auto tight = wide;
    tight.firstActiveFrameLine = 109;  // an odd origin moved the old grid
    tight.lastActiveFrameLine = 150;
    tight.activeVideoStart = 309;
    tight.activeVideoEnd = 355;

    auto wideDecoder = maskModelDecoder();
    auto tightDecoder = maskModelDecoder();
    REQUIRE(wideDecoder->configure(wide));
    REQUIRE(tightDecoder->configure(tight));

    const auto fields = noiseFields(6, wide);
    std::vector<ComponentFrame> wideFrame(1), tightFrame(1);
    wideDecoder->decodeFrames(fields, 2, 4, wideFrame);
    tightDecoder->decodeFrames(fields, 2, 4, tightFrame);

    // Stay clear of the crop edges, where the chroma post-filters run out of input.
    for (int32_t line = 120; line <= 140; line++) {
        for (int32_t x = 325; x < 340; x++) {
            REQUIRE(wideFrame[0].u(line)[x] == tightFrame[0].u(line)[x]);
            REQUIRE(wideFrame[0].v(line)[x] == tightFrame[0].v(line)[x]);
        }
    }
    return 0;
}

int testTileGridFollowsPicture() {
    using chd::decoders::nntransform3d::makeTileGrid;

    auto vp = croppedNtscParameters();
    vp.pictureStart = vp.tileReadStart = vp.activeVideoStart = 134;
    vp.tileReadEnd = vp.activeVideoEnd = 894;
    vp.firstActiveFrameLine = 40;
    vp.lastActiveFrameLine = 524;
    auto grid = makeTileGrid(vp);
    REQUIRE(grid.startY == 32 && grid.startX == 126);
    REQUIRE(grid.firstLine == 40 && grid.lastLine == 524);
    REQUIRE(grid.firstSample == 134 && grid.endSample == 894);

    // Line 39 joins the area, reached by one more row of the same lattice.
    vp.firstActiveFrameLine = 39;
    grid = makeTileGrid(vp);
    REQUIRE(grid.firstLine == 39 && grid.startY == 24);

    // A crop past the picture widens the read area along the same lattice.
    vp.tileReadStart = 120;
    grid = makeTileGrid(vp);
    REQUIRE(grid.firstSample == 120 && grid.startX == 110);

    // Encoder-style rows anchor on their own picture start.
    vp.pictureStart = vp.tileReadStart = 147;
    grid = makeTileGrid(vp);
    REQUIRE(grid.startX == 139);
    return 0;
}

// A padding frame holds no picture, so its samples must not reach the output.
int testPaddingFrameIsIgnored() {
    using chd::output::ComponentFrame;

    const auto vp = croppedNtscParameters();
    auto decoder = maskModelDecoder();
    REQUIRE(decoder->configure(vp));

    auto fields = noiseFields(6, vp);
    fields[4].isPadding = fields[5].isPadding = true;
    std::vector<ComponentFrame> padded(1);
    decoder->decodeFrames(fields, 2, 4, padded);

    fields[4].data.assign(fields[4].data.size(), static_cast<uint16_t>(vp.black16bIre));
    fields[5].data.assign(fields[5].data.size(), static_cast<uint16_t>(vp.black16bIre));
    std::vector<ComponentFrame> otherSamples(1);
    decoder->decodeFrames(fields, 2, 4, otherSamples);
    REQUIRE(*padded[0].getU() == *otherSamples[0].getU());

    // The same samples do matter once they count as picture.
    fields[4].isPadding = fields[5].isPadding = false;
    std::vector<ComponentFrame> notPadded(1);
    decoder->decodeFrames(fields, 2, 4, notPadded);
    REQUIRE(*padded[0].getU() != *notPadded[0].getU());
    return 0;
}

#endif  // CHD_TEST_NO_ONNXRUNTIME

}  // namespace

int main() {
    if (chd_has_feature("nn") != 1) {
        std::cout << "Skipping nnTransform3D tests (with_nn=false).\n";
        return 0;
    }
#ifdef CHD_TEST_NO_ONNXRUNTIME
    std::cout << "Skipping nnTransform3D tests (no ORT headers visible at test compile).\n";
    return 0;
#else
    int rc = 0;
    rc |= testLookAround();
    rc |= testConfigureWithoutSession();
    rc |= testConfigureWithSession();
    rc |= testConfigureWithMemorySession();
    rc |= testOneFrameCallMatchesBatch();
    rc |= testPaddingFrameIsIgnored();
    rc |= testCropOnlyTrimsOutput();
    rc |= testTileGridFollowsPicture();
    // Tear down the Ort::Env explicitly so its destructor runs before
    // C++ static destruction (the same destruction-order hazard as
    // test_nn_framework).
    chd_shutdown();
    if (rc == 0) std::cout << "All nnTransform3D wiring tests passed.\n";
    return rc;
#endif
}
