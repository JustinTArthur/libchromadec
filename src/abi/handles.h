// SPDX-License-Identifier: GPL-3.0-or-later
//
// Concrete layouts for the opaque ABI handle types declared in
// <chromadec/types.h>. The public header forward-declares
// `struct chd_video`, `struct chd_decoder`, etc.; this header gives
// those structs their internal C++ definition. Internal code in
// src/abi/*.cpp includes this to translate between the C ABI and the
// internal C++ classes.
//
// Public callers never see these definitions — they only ever hold
// pointers to the forward-declared opaque types.

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <chromadec/decoder.h>
#include <chromadec/dropout.h>
#include <chromadec/frame.h>

#include "../decoders/decoder_base.h"
#include "../decoders/registry.h"
#include "../dropout/multi_source_alignment.h"
#include "../metadata/core.h"
#include "../output/output_writer.h"
#include "../reader/source.h"

// chd_nn_model is only populated when the build has NN support (the C ABI
// rejects loads cleanly when with_nn=false). The model handle holds a
// backend-agnostic InferenceEngine (ORT or native CoreML), so this only
// needs the ORT-free interface header.
#if defined(CHD_WITH_NN)
#include "../nn/inference_engine.h"
#endif

// One opened source plus the metadata the decode path needs: a primary
// composite, one plane of a Y/C pair, or an extra source for multi-source
// dropout correction (whose DropoutCorrector::correctFrame overload needs
// per-source Field metadata, dropouts + bPSNR, that the source itself
// doesn't carry). For sources opened with an ld-decode sidecar the metadata
// is loaded from the .db/.json sidecar; for CVBS sources it's synthesized
// from the source's parameters() + field count.
struct chd_video_source {
    std::unique_ptr<chd::reader::ISource> source;
    std::unique_ptr<chd::metadata::LdDecodeMetaData> metadata;
    bool metadataSynthesized = false;
};

extern "C" {
struct chd_video {
    // Primary source. For a Y/C capture opened with chd_video_open_yc this
    // is the luma plane and `chromaSource` below holds the chroma plane;
    // otherwise `chromaSource` is null and this is the whole composite.
    // `metadata` is always non-null after a successful chd_video_open_*;
    // metadataSynthesized==false means it was loaded from an ld-decode
    // sqlite/json sidecar, true means it was synthesized in memory from the
    // source's ISource::parameters() + field count.
    std::unique_ptr<chd::metadata::LdDecodeMetaData> metadata;
    std::unique_ptr<chd::reader::ISource> source;
    std::string primaryPath;
    bool metadataSynthesized = false;
    std::vector<chd_video_source> extraSources;

    // Decode-level Y/C merge: when non-null, `source` carries luma and this
    // carries the separately-decoded chroma plane (a luma + chroma .tbc pair,
    // or a CVBS .cvbsy/.cvbsc pair whose centred chroma the reader re-centres
    // on blanking). The luma plane is decoded with a Mono
    // decoder and the chroma plane with the configured colour decoder; their
    // U/V are merged.
    std::unique_ptr<chd::reader::ISource> chromaSource;
    std::unique_ptr<chd::metadata::LdDecodeMetaData> chromaMetadata;
    std::vector<chd_video_source> chromaExtraSources;
};

// Per-decoder state. The lifecycle has two phases:
//   1. Uncommitted: allocated by chd_decoder_create; options are stashed
//      into the optionMaps + nnModelPending; commit() has not run yet so
//      `decoder` and `committed` are unset.
//   2. Committed: chd_decoder_commit has built the concrete Decoder,
//      configured it, snapshotted the post-padding VideoParameters, and
//      wired up the OutputWriter. From here on, set_option_* + set_nn_model
//      reject changes (commit is one-shot per the public header).
struct chd_decoder {
    chd_video_t *video = nullptr;        // non-owning back-pointer
    chd_decoder_kind_t kind = CHD_DEC_AUTO;

    // Pending options stashed by chd_decoder_set_option_*. Drained on
    // commit; left around afterwards in case future surfaces want
    // to inspect what the caller asked for.
    chd::decoders::registry::OptionMaps optionMaps;
#if defined(CHD_WITH_NN)
    std::shared_ptr<chd::nn::InferenceEngine> nnModelPending;
#endif

    bool dropoutOptsSet = false;
    chd_dropout_opts_t dropoutOpts{};

    // Per-frame dropout stats from the most recent chd_decode_frame call.
    // Guarded by statsMutex (below); with concurrent async workers, the
    // value reflects whichever frame finished last.
    chd_dropout_stats_t lastDropoutStats{};

    // SECAM click-concealment results, guarded by statsMutex: spans per
    // decoded frame (merged into chd_decoder_get_dropout_spans) and the
    // effective thresholds from the most recent decode.
    std::map<int64_t, std::vector<chd_dropout_span_t>> concealedSpansByFrame;
    double lastClickEnvDipDb = 0.0;
    double lastClickFreqOvershoot = 0.0;
    bool haveClickThresholds = false;

    // Set by commit. Once true, the fields below are populated and the
    // decode entry points work.
    bool committed = false;

    chd_decoder_kind_t resolvedKind = CHD_DEC_AUTO;

    // Hardening follow-up: N decoder instances, one per worker
    // thread. Each Decoder subclass keeps mutable per-call scratch (Comb
    // FrameBuffer, TransformPal FFTW plans, OutputWriter line buffers)
    // that isn't documented as reentrant, so true async parallelism
    // requires per-worker instances. They share the same OrtSession for
    // NN kinds (Ort::Session::Run is thread-safe per the ORT docs); the
    // Comb body serialises its own Run() calls via a per-instance
    // nnRunMutex which is fine.
    //
    // decoderMutexes is the same size as decoders — one mutex per
    // instance so async workers never wait on each other.
    std::vector<std::unique_ptr<chd::decoders::Decoder>> decoders;
    std::vector<std::unique_ptr<std::mutex>>             decoderMutexes;

    // Decode-level Y/C merge (video->chromaSource != nullptr). `decoders`
    // above are then Mono instances decoding the luma plane; these decode the
    // chroma plane with the configured colour kind. Indexed by the same worker
    // index as `decoders` and only ever touched while that worker holds
    // decoderMutexes[workerIdx], so they need no separate mutex.
    std::vector<std::unique_ptr<chd::decoders::Decoder>> chromaDecoders;
    int32_t chromaLookBehind = 0;
    int32_t chromaLookAhead  = 0;

    // Configured pipeline state. videoParameters is the post-padding
    // copy from OutputWriter::updateConfiguration (which mutates the
    // active-region bounds when padding > 1). OutputWriter::convert is
    // stateless after updateConfiguration so workers share one writer.
    chd::metadata::LdDecodeMetaData::VideoParameters videoParameters{};
    chd::output::OutputWriter outputWriter;
    chd::output::OutputWriter::Configuration outputConfig{};
    chd_pixel_format_t outputPixelFormat = CHD_PIXEL_YUV444P16;
    int32_t threadCount = 0;

    int32_t lookBehind = 0;
    int32_t lookAhead  = 0;

    // Multi-source dropout alignment, built lazily on the first decode that
    // has extra sources attached (its constructor scans every source's field
    // VBI, so it must run once). Read-only afterwards, shared across workers.
    std::unique_ptr<chd::dropout::MultiSourceAlignment> multiSourceAlignment;
    std::once_flag multiSourceAlignmentOnce;

    // Same, for the chroma plane's extra sources in a Y/C merge.
    std::unique_ptr<chd::dropout::MultiSourceAlignment> chromaMultiSourceAlignment;
    std::once_flag chromaMultiSourceAlignmentOnce;

    // Protects lastDropoutStats updates + reads. Small fast path —
    // worker takes it after the decode body to publish the stats.
    std::mutex statsMutex;
};

// chd_frame owns the rendered pixel data for one decoded frame. The format
// field follows the configured output pixel format from chd_decoder_commit;
// for u16 formats `u16Plane` holds the OutputWriter convert() output, and
// for the float formats (CHD_PIXEL_YUV444PS / CHD_PIXEL_RGBS / CHD_PIXEL_GRAYS)
// the three `floatPlane`s hold contiguous float planes converted directly
// from the decoder's ComponentFrame.
struct chd_frame {
    chd_frame_info_t info{};
    chd_pixel_format_t format = CHD_PIXEL_YUV444P16;

    // YUV444P16 / RGB48 / GRAY16: packed/planar as written by
    // chd::output::OutputWriter::convert. outputWidth/outputHeight match
    // info.width / info.height (the active crop plus any black pad border).
    // YUV440P16 appends the two subsampled chroma planes after the full-height
    // Y plane.
    chd::output::OutputFrame u16Plane;

    // YUV444PS / RGBS: three contiguous planes, each width*height floats.
    // GRAYS: only floatPlane[0] populated. YUV440PS: planes 1/2 hold
    // width*chroma440.{cb,cr}Height floats.
    std::vector<float> floatPlane[3];

    int32_t outputWidth  = 0;
    int32_t outputHeight = 0;

    // 4:4:0 frames: chroma plane geometry, the per-output-row component map
    // backing chd_frame_chroma_row_component (0 = Db, 1 = Dr, sized
    // outputHeight; empty otherwise), and the ident report.
    chd::output::OutputWriter::Chroma440Geometry chroma440{};
    std::vector<int8_t> rowComponent;
    chd_chroma_ident_report_t identReport{};
    bool hasIdentReport = false;
};

struct chd_cancel {
    std::atomic<bool> requested{false};
};

#if defined(CHD_WITH_NN)
struct chd_nn_model {
    std::shared_ptr<chd::nn::InferenceEngine> engine;
};
#else
struct chd_nn_model {
    // Placeholder so the opaque-handle type still has a definition when NN
    // is disabled at build time. Any call into chd_nn_* returns
    // CHD_E_INTERNAL via the c_api_nn.cpp stubs.
    int unused;
};
#endif
}  // extern "C"
