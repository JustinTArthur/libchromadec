/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef CHROMADEC_TYPES_H
#define CHROMADEC_TYPES_H

#include <stdint.h>
#include <stddef.h>

#include <chromadec/enum.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct chd_video    chd_video_t;
typedef struct chd_decoder  chd_decoder_t;
typedef struct chd_frame    chd_frame_t;
typedef struct chd_nn_model chd_nn_model_t;
typedef struct chd_cancel   chd_cancel_t;

typedef CHD_ENUM(chd_video_standard) {
    CHD_STD_UNKNOWN = 0,
    CHD_STD_NTSC,
    CHD_STD_PAL,
    CHD_STD_PAL_M,

    CHD_STD_SECAM   = 8
} chd_video_standard_t;

typedef CHD_ENUM(chd_sample_encoding) {
    CHD_ENC_UNKNOWN = 0,
    CHD_ENC_CVBS_U10_4FSC,
    CHD_ENC_CVBS_U16_4FSC,
    CHD_ENC_CVBS_TPG21_4FSC,
    CHD_ENC_CVBS_S16_FSC,
    CHD_ENC_RAW_S16_28M,
    CHD_ENC_RAW_S16_40M
} chd_sample_encoding_t;

typedef CHD_ENUM(chd_signal_state) {
    CHD_SIG_UNKNOWN = 0,
    CHD_SIG_STANDARD_TBC_LOCKED,
    CHD_SIG_STANDARD_TBC_UNLOCKED,
    CHD_SIG_STANDARD_RAW,
    CHD_SIG_NONSTANDARD_TBC_LOCKED,
    CHD_SIG_NONSTANDARD_TBC_UNLOCKED,
    CHD_SIG_NONSTANDARD_RAW
} chd_signal_state_t;

typedef CHD_ENUM(chd_frame_layout) {
    CHD_FRAME_LAYOUT_UNKNOWN = 0,
    CHD_FRAME_LAYOUT_FIELD_RASTER,
    CHD_FRAME_LAYOUT_FRAME_NATIVE
} chd_frame_layout_t;

typedef CHD_ENUM(chd_plane) {
    CHD_PLANE_Y  = 0,
    CHD_PLANE_CB = 1,
    CHD_PLANE_CR = 2,
    CHD_PLANE_R  = 3,
    CHD_PLANE_G  = 4,
    CHD_PLANE_B  = 5
} chd_plane_t;

typedef CHD_ENUM(chd_pixel_format) {
    CHD_PIXEL_YUV444P16 = 0,
    CHD_PIXEL_YUV444PS  = 1,
    CHD_PIXEL_RGB48     = 2,
    CHD_PIXEL_RGBS      = 3,
    CHD_PIXEL_GRAY16    = 4,
    CHD_PIXEL_GRAYS     = 5,

    /* 4:4:0: full-width chroma planes holding only really-decoded rows
     * (line-sequential chroma; see chd_frame_get_plane_info). */
    CHD_PIXEL_YUV440P16 = 8,
    CHD_PIXEL_YUV440PS  = 9
} chd_pixel_format_t;

typedef CHD_ENUM(chd_clamp) {
    CHD_CLAMP_NONE               = 0,
    CHD_CLAMP_LEGAL_RGB_SDR      = 1,
    CHD_CLAMP_LEGAL_RGB_HDR      = 2,
    CHD_CLAMP_LEGAL_YCBCR_BT601  = 3
} chd_clamp_t;

typedef struct chd_video_params {
    chd_video_standard_t  standard;
    chd_sample_encoding_t encoding;
    chd_signal_state_t    signal_state;
    chd_frame_layout_t    layout;
    int      is_subcarrier_locked;
    int      is_second_field_first;
} chd_video_params_t;

/* Source geometry and levels, reported by chd_video_get_info. The four
 * first_/last_ crop bounds are inclusive on both axes. Frame lines are
 * 0-indexed lines of the woven interlaced frame (field 1 on the even lines,
 * field 2 on the odd); use chd_video_frame_line_to_signal_line /
 * chd_video_signal_line_to_frame_line to convert to and from the analogue
 * standards' field-sequential signal line numbers. Sample bounds are 0-indexed
 * row positions within a stored row, whose origin depends on the source's
 * horizontal alignment and so need not match the sample numbering of EBU Tech
 * 3280 or SMPTE ST 244. */
typedef struct chd_video_info {
    chd_video_standard_t  standard;
    chd_sample_encoding_t encoding;
    chd_signal_state_t    signal_state;
    chd_frame_layout_t    layout;
    int32_t  field_width;
    int32_t  field_height;
    int32_t  samples_per_frame;
    double   sample_rate_hz;
    double   fsc_hz;
    int32_t  first_active_sample;
    int32_t  last_active_sample;
    int32_t  first_active_frame_line;
    int32_t  last_active_frame_line;
    int32_t  black_16b_ire;
    int32_t  white_16b_ire;
    int32_t  blanking_16b_ire;
    int64_t  num_frames;
    int      is_widescreen;
    int      is_subcarrier_locked;
    int      is_first_field_first;
} chd_video_info_t;

typedef struct chd_frame_info {
    chd_pixel_format_t format;
    int32_t  width;
    int32_t  height;
    int32_t  num_planes;
    int64_t  frame_index;
} chd_frame_info_t;

/* Per-plane geometry, reported by chd_frame_get_plane_info. Full-height
 * planes report the frame dimensions and first_frame_row 0. 4:4:0 chroma
 * planes weave the two fields by row parity, matching the luma plane, and
 * report their subsampled height and the output frame row plane row 0 was
 * decoded from; each of their rows is a real decoded line, located per row
 * via chd_frame_chroma_row_component. */
typedef struct chd_plane_info {
    int32_t width;
    int32_t height;
    int32_t first_frame_row;
} chd_plane_info_t;

/* Colour-difference component carried by one frame row of a
 * line-sequential (SECAM) decode. */
typedef CHD_ENUM(chd_chroma_row_component) {
    CHD_CHROMA_ROW_DB = 0,
    CHD_CHROMA_ROW_DR = 1
} chd_chroma_row_component_t;

/* How per-line Db/Dr identity was decided for a frame. */
typedef CHD_ENUM(chd_chroma_ident_mechanism) {
    CHD_CHROMA_IDENT_PORCH   = 0,
    CHD_CHROMA_IDENT_BOTTLES = 1,
    CHD_CHROMA_IDENT_CONTENT = 2,
    CHD_CHROMA_IDENT_MANUAL  = 3
} chd_chroma_ident_mechanism_t;

/* Per-frame ident summary, reported by chd_frame_get_chroma_ident.
 * confidence is the fraction of measured lines agreeing with the majority
 * lattice (1.0 for manual); field_confidence splits it by field in frame
 * order. */
typedef struct chd_chroma_ident_report {
    chd_chroma_ident_mechanism_t mechanism;
    double confidence;
    double field_confidence[2];
    chd_chroma_row_component_t first_row_component;
} chd_chroma_ident_report_t;

/* Committed output framing, reported by chd_decoder_get_output_info. width and
 * height are the active picture crop plus any black border CHD_OPT_PADDING_MULTIPLE
 * adds around it: the same dimensions chd_decode_frame produces and the
 * coordinate space dropout spans are expressed in. */
typedef struct chd_output_info {
    chd_pixel_format_t format;
    int32_t  width;
    int32_t  height;
    int32_t  num_planes;
    int64_t  num_frames;
} chd_output_info_t;

#ifdef __cplusplus
}
#endif
#endif
