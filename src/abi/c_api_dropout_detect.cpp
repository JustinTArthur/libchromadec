// SPDX-License-Identifier: GPL-3.0-or-later
//
// Decode-free dropout detection on the C ABI.
//
// Dropouts are detected upstream (by ld-decode) and stored in the source's
// sidecar metadata; this library consumes them during dropout correction. The
// functions here expose those detected regions to consumers without running the
// chroma decoder: they load only the frame's two fields and their metadata,
// then map each stored (startx, endx, fieldLine) region into the committed
// active-output pixel framing.
//
// The mapping mirrors the decode path:
//   - horizontal: out_x = sample_x - activeVideoStart, clamped to the active
//     width (activeVideoStart/End carry the post-padding values snapshotted at
//     commit).
//   - vertical: a field-relative line (1-based) becomes an interlaced
//     ComponentFrame row 2*(fieldLine-1)+offset, then crop+pad into output
//     space with firstActiveFrameLine and topPadLines, the same crop and
//     vertical padding the decode path applies.

#include <chromadec/decoder.h>
#include <chromadec/dropout.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "../common/error_state.h"
#include "../decoders/source_field.h"
#include "../metadata/core.h"
#include "handles.h"

namespace {

chd_status_t arg_error(const char *what) {
    chd::detail::set_last_error(std::string("chromadec: ") + what + ": null argument");
    return CHD_E_INVALID_ARG;
}

// Committed active-output framing, in the coordinate space chd_decode_frame
// produces and dropout spans are expressed in.
struct OutputGeometry {
    int32_t activeVideoStart;
    int32_t activeWidth;   // the picture crop
    int32_t width;         // the emitted frame, crop plus black pad border
    int32_t height;
    int32_t firstActiveFrameLine;
    int32_t leftPadSamples;
    int32_t topPadLines;
    int32_t fieldWidth;
};

OutputGeometry committedGeometry(const chd_decoder_t *d) {
    OutputGeometry g;
    g.activeVideoStart     = d->videoParameters.activeVideoStart;
    g.activeWidth          = d->outputWriter.getActiveWidth();
    g.width                = d->outputWriter.getOutputWidth();
    g.firstActiveFrameLine = d->videoParameters.firstActiveFrameLine;
    g.leftPadSamples       = d->outputWriter.getLeftPadSamples();
    g.topPadLines          = d->outputWriter.getTopPadLines();
    g.height               = d->outputWriter.getOutputHeight();
    g.fieldWidth           = d->videoParameters.fieldWidth;
    return g;
}

bool validDetectMode(chd_dropout_detect_mode_t mode) {
    return mode == CHD_DROPOUT_DETECTED || mode == CHD_DROPOUT_OVERCORRECT;
}

int32_t outputPlaneCount(chd_pixel_format_t format) {
    return (format == CHD_PIXEL_GRAY16 || format == CHD_PIXEL_GRAYS) ? 1 : 3;
}

// A mask is single-channel; emit it in the precision domain of the committed
// output format so the mask clip pairs with the decode clip without extra
// configuration: any float output format -> GRAYS, anything else -> GRAY16.
chd_pixel_format_t maskFormatFor(chd_pixel_format_t committed) {
    switch (committed) {
        case CHD_PIXEL_GRAYS:
        case CHD_PIXEL_YUV444PS:
        case CHD_PIXEL_RGBS:
        case CHD_PIXEL_YUV440PS:
            return CHD_PIXEL_GRAYS;
        default:
            return CHD_PIXEL_GRAY16;
    }
}

// Append this field's dropouts, mapped into output framing, to `spans`.
void appendFieldSpans(const chd::decoders::SourceField &sf, const OutputGeometry &g,
                      chd_dropout_detect_mode_t mode,
                      std::vector<chd_dropout_span_t> &spans) {
    const auto &dropOuts = sf.field.dropOuts;
    const int32_t offset = sf.getOffset();  // 0 top field, 1 bottom field
    const int32_t avs = g.activeVideoStart;
    const int32_t ave = g.activeVideoStart + g.activeWidth;

    for (int32_t i = 0; i < dropOuts.size(); i++) {
        const int32_t fieldLine = dropOuts.fieldLine(i);  // 1-based
        const int32_t frameRow  = (2 * (fieldLine - 1)) + offset;
        const int32_t y = g.topPadLines + frameRow - g.firstActiveFrameLine;
        if (y < 0 || y >= g.height) continue;

        int32_t x0 = dropOuts.startx(i);
        int32_t x1 = dropOuts.endx(i);
        if (mode == CHD_DROPOUT_OVERCORRECT) {
            // Mirror DropoutCorrector::populateDropoutsVector's overcorrect
            // widening (±24 samples, clamped to the field) so the reported
            // region is the footprint overcorrect-mode concealment would touch.
            const int32_t overCorrectionDots = 24;
            x0 = (x0 > overCorrectionDots) ? x0 - overCorrectionDots : 0;
            x1 = (x1 < g.fieldWidth - overCorrectionDots) ? x1 + overCorrectionDots : g.fieldWidth;
        }
        if (x0 < avs) x0 = avs;
        if (x1 > ave) x1 = ave;
        if (x1 <= x0) continue;  // fully outside the active region horizontally

        // Field sample positions land in the padded frame's coordinate space:
        // shift past the left border, as the y above already shifts past the top.
        chd_dropout_span_t span;
        span.y       = y;
        span.x_start = x0 - avs + g.leftPadSamples;
        span.x_end   = x1 - avs + g.leftPadSamples;
        span.origin  = CHD_DROPOUT_ORIGIN_SOURCE_METADATA;
        spans.push_back(span);
    }
}

// Load the frame's two fields (no look-behind/ahead, no chroma decode) and
// collect their dropout spans in output framing, ordered by (y, x_start).
chd_status_t collectSpans(chd_decoder_t *d, int64_t frame_index,
                          chd_dropout_detect_mode_t mode,
                          std::vector<chd_dropout_span_t> &spans) {
    chd::metadata::LdDecodeMetaData *meta = d->video->metadata.get();
    if (meta == nullptr) {
        chd::detail::set_last_error("chromadec dropout query: missing metadata");
        return CHD_E_METADATA_MISSING;
    }
    const bool firstFieldFirst = d->firstFieldFirst(*meta);
    const int32_t numFrames = meta->getNumberOfFrames(firstFieldFirst);
    if (frame_index < 0 || frame_index >= numFrames) {
        chd::detail::set_last_error("chromadec dropout query: frame_index out of range");
        return CHD_E_OUT_OF_RANGE;
    }

    std::vector<chd::decoders::SourceField> fields;
    int32_t startIndex = 0;
    int32_t endIndex   = 0;
    const int32_t firstFrameNumber = static_cast<int32_t>(frame_index) + 1;
    chd::decoders::SourceField::loadFields(
        *d->video->source, *meta, firstFieldFirst,
        firstFrameNumber, /*numFrames=*/1, /*lookBehind=*/0, /*lookAhead=*/0,
        fields, startIndex, endIndex);

    const OutputGeometry g = committedGeometry(d);
    const int32_t total = static_cast<int32_t>(fields.size());
    if (startIndex < total)     appendFieldSpans(fields[startIndex], g, mode, spans);
    if (startIndex + 1 < total) appendFieldSpans(fields[startIndex + 1], g, mode, spans);

    // Merge decoder-detected concealment spans for this frame (present once
    // the frame has been decoded with click concealment enabled). One
    // enumeration path, filterable by origin.
    {
        std::lock_guard<std::mutex> sl(d->statsMutex);
        const auto it = d->concealedSpansByFrame.find(frame_index);
        if (it != d->concealedSpansByFrame.end()) {
            spans.insert(spans.end(), it->second.begin(), it->second.end());
        }
    }

    std::sort(spans.begin(), spans.end(),
              [](const chd_dropout_span_t &a, const chd_dropout_span_t &b) {
                  if (a.y != b.y) return a.y < b.y;
                  return a.x_start < b.x_start;
              });
    return CHD_OK;
}

}  // namespace

extern "C" {

chd_status_t chd_decoder_get_output_info(const chd_decoder_t *d, chd_output_info_t *out) {
    if (d == nullptr || out == nullptr) return arg_error("chd_decoder_get_output_info");
    if (!d->committed) {
        chd::detail::set_last_error("chd_decoder_get_output_info: chd_decoder_commit not called");
        return CHD_E_INVALID_ARG;
    }
    const OutputGeometry g = committedGeometry(d);
    out->format     = d->outputPixelFormat;
    out->width      = g.width;
    out->height     = g.height;
    out->num_planes = outputPlaneCount(d->outputPixelFormat);
    chd::metadata::LdDecodeMetaData *meta = d->video->metadata.get();
    // The frames this decoder can decode, under its own field pairing (which
    // reverse_field_order can make one fewer than chd_video_get_info's).
    out->num_frames = (meta != nullptr) ? meta->getNumberOfFrames(d->firstFieldFirst(*meta)) : 0;
    return CHD_OK;
}

chd_status_t chd_decoder_get_dropout_spans(chd_decoder_t *d, int64_t frame_index,
                                           chd_dropout_detect_mode_t mode,
                                           chd_dropout_span_t **out_spans,
                                           size_t *out_count) {
    if (d == nullptr || out_spans == nullptr || out_count == nullptr)
        return arg_error("chd_decoder_get_dropout_spans");
    *out_spans = nullptr;
    *out_count = 0;
    if (!d->committed) {
        chd::detail::set_last_error("chd_decoder_get_dropout_spans: chd_decoder_commit not called");
        return CHD_E_INVALID_ARG;
    }
    if (!validDetectMode(mode)) {
        chd::detail::set_last_error("chd_decoder_get_dropout_spans: unknown detect mode");
        return CHD_E_INVALID_ARG;
    }

    std::vector<chd_dropout_span_t> spans;
    const chd_status_t rc = collectSpans(d, frame_index, mode, spans);
    if (rc != CHD_OK) return rc;
    if (spans.empty()) return CHD_OK;  // *out_count == 0, *out_spans == nullptr

    const size_t bytes = spans.size() * sizeof(chd_dropout_span_t);
    auto *arr = static_cast<chd_dropout_span_t *>(std::malloc(bytes));
    if (arr == nullptr) {
        chd::detail::set_last_error("chd_decoder_get_dropout_spans: out of memory");
        return CHD_E_OOM;
    }
    std::memcpy(arr, spans.data(), bytes);
    *out_spans = arr;
    *out_count = spans.size();
    return CHD_OK;
}

void chd_dropout_spans_free(chd_dropout_span_t *spans) {
    std::free(spans);
}

chd_status_t chd_decode_dropout_mask(chd_decoder_t *d, int64_t frame_index,
                                     chd_dropout_detect_mode_t mode, chd_frame_t **out) {
    if (d == nullptr || out == nullptr) return arg_error("chd_decode_dropout_mask");
    *out = nullptr;
    if (!d->committed) {
        chd::detail::set_last_error("chd_decode_dropout_mask: chd_decoder_commit not called");
        return CHD_E_INVALID_ARG;
    }
    if (!validDetectMode(mode)) {
        chd::detail::set_last_error("chd_decode_dropout_mask: unknown detect mode");
        return CHD_E_INVALID_ARG;
    }

    std::vector<chd_dropout_span_t> spans;
    const chd_status_t rc = collectSpans(d, frame_index, mode, spans);
    if (rc != CHD_OK) return rc;

    const OutputGeometry g = committedGeometry(d);
    if (g.width <= 0 || g.height <= 0) {
        chd::detail::set_last_error("chd_decode_dropout_mask: invalid output geometry");
        return CHD_E_INTERNAL;
    }

    const chd_pixel_format_t maskFormat = maskFormatFor(d->outputPixelFormat);
    const size_t pixels = static_cast<size_t>(g.width) * static_cast<size_t>(g.height);

    auto frame = std::make_unique<chd_frame>();
    frame->format       = maskFormat;
    frame->outputWidth  = g.width;
    frame->outputHeight = g.height;

    if (maskFormat == CHD_PIXEL_GRAYS) {
        frame->floatPlane[0].assign(pixels, 0.0f);
        for (const auto &s : spans) {
            float *row = frame->floatPlane[0].data() + static_cast<size_t>(s.y) * g.width;
            for (int32_t x = s.x_start; x < s.x_end; x++) row[x] = 1.0f;
        }
    } else {
        frame->u16Plane.assign(pixels, 0);
        for (const auto &s : spans) {
            uint16_t *row = frame->u16Plane.data() + static_cast<size_t>(s.y) * g.width;
            for (int32_t x = s.x_start; x < s.x_end; x++) row[x] = 0xFFFF;
        }
    }

    frame->info.format      = maskFormat;
    frame->info.width       = g.width;
    frame->info.height      = g.height;
    frame->info.num_planes  = 1;
    frame->info.frame_index = frame_index;

    *out = frame.release();
    return CHD_OK;
}

}  // extern "C"
