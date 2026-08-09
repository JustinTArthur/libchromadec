/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef CHROMADEC_DROPOUT_H
#define CHROMADEC_DROPOUT_H

#include <chromadec/enum.h>
#include <chromadec/errors.h>
#include <chromadec/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct chd_dropout_opts {
    int enabled;
    int overcorrect;        /* extend dropout boundaries by ±24 samples */
    int intra_field_only;   /* skip cross-field replacement candidates */
    /* Extra sources are added separately via chd_video_add_extra_source_*. */

    /* Reserved for future ABI extensions; zero-initialise the struct (e.g.
     * `chd_dropout_opts_t o = {0}`). See docs/abi-stability.md. */
    void *reserved[4];
} chd_dropout_opts_t;

typedef struct chd_dropout_stats {
    int32_t corrected;
    int32_t failed;
    int64_t total_distance;
} chd_dropout_stats_t;

chd_status_t chd_decoder_set_dropout(chd_decoder_t *d, const chd_dropout_opts_t *opts);

/* Stats from the most recent chd_decode_frame call on this decoder. */
chd_status_t chd_decoder_get_last_dropout_stats(const chd_decoder_t *d,
                                                 chd_dropout_stats_t *out);

/* Where a reported span came from. Numeric-gapped per family so future
 * origins (extension metadata, other decoder detectors) can slot in. */
typedef CHD_ENUM(chd_dropout_origin) {
    CHD_DROPOUT_ORIGIN_SOURCE_METADATA     = 0,

    CHD_DROPOUT_ORIGIN_DECODER_CONCEALMENT = 8
} chd_dropout_origin_t;

/* A run of dropped samples on one output row, expressed in the committed
 * active-output framing: x is half-open [x_start, x_end) within [0, output
 * width); y is a row within [0, output height). The mapping from the stored
 * field-relative dropouts honours the committed crop, padding, line overrides
 * and field order, so spans align pixel-for-pixel with frames decoded by the
 * same committed decoder. origin distinguishes upstream-flagged regions from
 * samples the decoder itself detected and concealed (SECAM FM click
 * concealment; present after the frame has been decoded). */
typedef struct chd_dropout_span {
    int32_t y;
    int32_t x_start;
    int32_t x_end;
    chd_dropout_origin_t origin;
} chd_dropout_span_t;

/* Which dropout regions a detection query reports (mutually exclusive). */
typedef CHD_ENUM(chd_dropout_detect_mode) {
    CHD_DROPOUT_DETECTED    = 0,  /* raw flagged regions */
    CHD_DROPOUT_OVERCORRECT = 1   /* detected, widened by the overcorrect margin */
} chd_dropout_detect_mode_t;

/* Low-level: the dropout spans for one frame, without running the chroma
 * decoder. mode selects which regions are reported (see chd_dropout_detect_mode).
 * On success *out_spans points to a newly-allocated array of *out_count spans
 * owned by the caller (free with chd_dropout_spans_free); a frame with no
 * dropouts yields *out_count == 0 and *out_spans == NULL. Requires a committed
 * decoder. */
chd_status_t chd_decoder_get_dropout_spans(chd_decoder_t *d, int64_t frame_index,
                                           chd_dropout_detect_mode_t mode,
                                           chd_dropout_span_t **out_spans,
                                           size_t *out_count);

void chd_dropout_spans_free(chd_dropout_span_t *spans);

/* Convenience: the same dropout regions (per mode) rasterised into a
 * self-describing single-plane frame matching the committed output framing —
 * clean samples 0, dropped samples set. The mask follows the committed output
 * format's precision domain: GRAYS (1.0 dropped) for a float-committed decoder,
 * GRAY16 (0xFFFF dropped) otherwise. Free with chd_frame_free. Does not run the
 * chroma decoder. */
chd_status_t chd_decode_dropout_mask(chd_decoder_t *d, int64_t frame_index,
                                     chd_dropout_detect_mode_t mode,
                                     chd_frame_t **out);

#ifdef __cplusplus
}
#endif
#endif
