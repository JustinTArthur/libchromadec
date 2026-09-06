/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef CHROMADEC_DECODER_H
#define CHROMADEC_DECODER_H

#include <chromadec/enum.h>
#include <chromadec/errors.h>
#include <chromadec/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef CHD_ENUM(chd_decoder_kind) {
    CHD_DEC_AUTO              = 0,
    CHD_DEC_MONO              = 1,
    CHD_DEC_NTSC_1D           = 2,
    CHD_DEC_NTSC_2D           = 3,
    CHD_DEC_NTSC_3D           = 4,
    CHD_DEC_NTSC_3D_NO_ADAPT  = 5,
    CHD_DEC_PAL_2D            = 6,
    CHD_DEC_TRANSFORM_2D      = 7,
    CHD_DEC_TRANSFORM_3D      = 8,
    CHD_DEC_NN_TRANSFORM3D    = 9,
    CHD_DEC_LDZEUG_COLOR_CNN  = 10,
    CHD_DEC_LDZEUG_LUMA_SEP   = 11,
    CHD_DEC_LDZEUG_LUMA_SEP_FRAME = 12,
    /* Geometry/metadata only: commit resolves the output framing but builds no
     * chroma decode engines, and requires no NN model. */
    CHD_DEC_NONE                  = 13,
    CHD_DEC_SECAM                 = 20,
    CHD_DEC_HVD_2D                = 30,
    CHD_DEC_HVD_3D                = 31
} chd_decoder_kind_t;

chd_status_t chd_decoder_create(chd_video_t *v, chd_decoder_kind_t kind, chd_decoder_t **out);
void         chd_decoder_free(chd_decoder_t *d);

/* Strongly-typed option setters. CHD_E_INVALID_ARG if not meaningful for kind.
 * Not safe to call concurrently with chd_decode_frame on the same decoder. */
chd_status_t chd_decoder_set_option_f64(chd_decoder_t *d, const char *name, double v);
chd_status_t chd_decoder_set_option_i32(chd_decoder_t *d, const char *name, int32_t v);
chd_status_t chd_decoder_set_option_bool(chd_decoder_t *d, const char *name, int v);
chd_status_t chd_decoder_set_option_str(chd_decoder_t *d, const char *name, const char *v);
chd_status_t chd_decoder_has_option(const chd_decoder_t *d, const char *name);

chd_status_t chd_decoder_set_nn_model(chd_decoder_t *d, chd_nn_model_t *m);

/* Apply pending options. Required before chd_decode_frame. Cheap to call repeatedly. */
chd_status_t chd_decoder_commit(chd_decoder_t *d);

/* Output framing after commit: the active region post-crop/padding, the pixel
 * format, and the frame count, for sizing buffers or clips before decoding.
 * Also serves the decode-free dropout-span path, where no frame is produced to
 * query for dimensions. Requires a committed decoder. */
chd_status_t chd_decoder_get_output_info(const chd_decoder_t *d, chd_output_info_t *out);

/* Stable option-name registry. */
#define CHD_OPT_CHROMA_GAIN                 "chroma_gain"               /* f64 */
#define CHD_OPT_CHROMA_PHASE_DEG            "chroma_phase_deg"          /* f64 */
#define CHD_OPT_CHROMA_NR_LEVEL             "chroma_nr_level"           /* f64 */
#define CHD_OPT_LUMA_NR_LEVEL               "luma_nr_level"             /* f64 */
#define CHD_OPT_PADDING_MULTIPLE            "padding_multiple"          /* i32, default 1 (no padding); black border, both axes */
#define CHD_OPT_REVERSE_FIELD_ORDER         "reverse_field_order"       /* bool */
#define CHD_OPT_PHASE_COMPENSATION          "phase_compensation"        /* bool, NTSC */
#define CHD_OPT_COMB_ADAPT_THRESHOLD        "comb_adapt_threshold"      /* f64, NTSC 3D */
#define CHD_OPT_COMB_CHROMA_WEIGHT          "comb_chroma_weight"        /* f64, NTSC 3D */
#define CHD_OPT_COMB_SHOW_MAP               "comb_show_map"             /* bool, NTSC 3D */
#define CHD_OPT_CHROMA_IDENT_MODE           "chroma_ident_mode"         /* str: "auto"|"porch"|"bottles"|"manual", SECAM */
#define CHD_OPT_CHROMA_IDENT_MANUAL         "chroma_ident_manual"       /* str: "db_first"|"dr_first"; required iff mode is "manual" */
#define CHD_OPT_CHROMA_CLICK_NR_LEVEL       "chroma_click_nr_level"     /* f64 0.0-1.0, default 1.0, SECAM FM click concealment (0.0 = off) */
#define CHD_OPT_CHROMA_CLICK_ENV_DIP_DB     "chroma_click_env_dip_db"   /* f64 dB, absolute override of the adaptive envelope-dip threshold */
#define CHD_OPT_CHROMA_CLICK_FREQ_OVERSHOOT "chroma_click_freq_overshoot" /* f64 multiples of the max-deviation bound, absolute override */
#define CHD_OPT_TRANSFORM_THRESHOLD         "transform_threshold"       /* f64 */
#define CHD_OPT_TRANSFORM_THRESHOLDS_FILE   "transform_thresholds_file" /* str */
#define CHD_OPT_FIRST_ACTIVE_SAMPLE         "first_active_sample"       /* i32, inclusive, 0-indexed */
#define CHD_OPT_LAST_ACTIVE_SAMPLE          "last_active_sample"        /* i32, inclusive, 0-indexed */
#define CHD_OPT_FIRST_ACTIVE_FRAME_LINE     "first_active_frame_line"   /* i32, inclusive, 0-indexed woven frame line */
#define CHD_OPT_LAST_ACTIVE_FRAME_LINE      "last_active_frame_line"    /* i32, inclusive, 0-indexed woven frame line */
#define CHD_OPT_NN_INPUT_MAGNITUDE_SCALE    "nn_input_magnitude_scale"  /* f64 (nnTransform3D only) */
#define CHD_OPT_NN_CHROMA_BANDPASS          "nn_chroma_bandpass"        /* bool (ldzeug2_luma_sep only) */
#define CHD_OPT_HVD_CG_ITERATIONS           "hvd_cg_iterations"         /* i32 >= 0, HVD only: total conjugate-gradient budget across the solver's outer passes. 0 decodes with the holographic init alone; the default (2) favours speed, raise it when tuning for final quality */
#define CHD_OPT_HVD_TEMPORAL_STRENGTH       "hvd_temporal_strength"     /* f64 >= 0, CHD_DEC_HVD_3D only: weight of the cross-field data terms. 0 (default) adapts per window to the measured Y/C ambiguity; a positive value forces that fixed strength */
#define CHD_OPT_OUTPUT_FORMAT               "output_format"             /* str: "yuv444p16"|"yuv444ps"|"rgb48"|"rgbs"|"gray16"|"grays"|"yuv440p16"|"yuv440ps" */
#define CHD_OPT_OUTPUT_CLAMP                "output_clamp"              /* str: "none"|"legal_rgb_sdr"|"legal_rgb_hdr"|"legal_ycbcr_bt601" */
#define CHD_OPT_COLOR_DIFFERENCE_PRECISION  "color_difference_precision"  /* str: "classic"|"modern", default "modern" */
#define CHD_OPT_BROADCAST_SCALING_PRECISION "broadcast_scaling_precision" /* str: "classic"|"modern"|"scientific", default "scientific" */
#define CHD_OPT_OUTPUT_Y4M_HEADERS          "output_y4m_headers"        /* bool */
#define CHD_OPT_THREAD_COUNT                "thread_count"              /* i32, 0=auto */

/* Effective SECAM click-concealment thresholds applied to the most recent
 * chd_decode_frame on this decoder (after the adaptive formula or expert
 * overrides): the envelope-dip depth in dB and the deviation-overshoot
 * multiple. CHD_E_UNSUPPORTED before any decode or when concealment is off. */
chd_status_t chd_decoder_get_chroma_click_thresholds(const chd_decoder_t *d,
                                                     double *env_dip_db,
                                                     double *freq_overshoot);

chd_status_t chd_decode_frame(chd_decoder_t *d, int64_t frame_index, chd_frame_t **out);

typedef void (*chd_frame_done_cb)(void *user, chd_status_t s, int64_t idx, chd_frame_t *f);
chd_status_t chd_decode_frames_async(chd_decoder_t *d,
                                      const int64_t *indices, size_t n,
                                      chd_frame_done_cb cb, void *user,
                                      chd_cancel_t *cancel_or_null);

#ifdef __cplusplus
}
#endif
#endif
