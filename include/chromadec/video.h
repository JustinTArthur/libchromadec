/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef CHROMADEC_VIDEO_H
#define CHROMADEC_VIDEO_H

#include <chromadec/errors.h>
#include <chromadec/types.h>

#ifdef __cplusplus
extern "C" {
#endif

chd_status_t chd_init(void);
void         chd_shutdown(void);

/* Open a single-file composite capture: an ld-decode `.tbc` or a CVBS
 * `.cvbs`. The parameters the raw samples omit come from a metadata
 * sidecar file next to the data. metadata_path_or_null:
 *   - NULL  → library auto-locates the sidecar next to path: an ld-decode
 *             `<path>.db` / `<path>.json` (a `.tbcy` / `.tbcc` plane looks
 *             beside its `.tbc` name), else a CVBS `<basename>.meta`
 *   - explicit path to a `.db`, `.json`, or `.meta` sidecar
 * override_or_null:
 *   - NULL  → all parameters come from the sidecar
 *   - standard/encoding/signal_state are required when no sidecar is found;
 *     layout, is_subcarrier_locked, and is_second_field_first merge
 *     field-wise over sidecar metadata
 *   - with an ld-decode sidecar, a non-zero standard re-declares the colour
 *     standard over the sidecar's (same line standard only; e.g. SECAM for
 *     an ME-SECAM capture whose sidecar says PAL)
 * The sidecar flavour (ld-decode vs CVBS) is detected automatically. */
chd_status_t chd_video_open_composite(const char *path,
                                      const char *metadata_path_or_null,
                                      const chd_video_params_t *override_or_null,
                                      chd_video_t **out);

/* Open a dual-file Y/C capture: a CVBS `.cvbsy` + `.cvbsc` pair, or a luma
 * `.tbc` + chroma `.tbc` pair. The luma plane is decoded for Y and the chroma
 * plane for U/V, then merged. Sidecar resolution and flavour detection follow
 * chd_video_open_composite. metadata_path_or_null applies to the luma plane;
 * the chroma plane uses its own sidecar if present, else the luma one (many
 * producers write a single shared sidecar for the pair). */
chd_status_t chd_video_open_yc(const char *luma_path,
                               const char *chroma_path,
                               const char *metadata_path_or_null,
                               const chd_video_params_t *override_or_null,
                               chd_video_t **out);

void         chd_video_free(chd_video_t *v);

chd_status_t chd_video_get_info(const chd_video_t *v, chd_video_info_t *out);

/* Convert between a line of our source woven-frame raster and the
 * field-sequential signal line number the analogue standards use (SMPTE ST 170
 * / ST 244, ITU-R BT.470 / BT.1700, EBU Tech 3280).
 *
 * A frame line is the 0-indexed inclusive woven-raster line: the space
 * chd_video_info's first_/last_active_frame_line and the
 * CHD_OPT_*_ACTIVE_FRAME_LINE options live in. The raster weaves the two
 * fields, so field 1 (the top field) is signal lines 1..field_height on the
 * even frame lines and field 2 is signal lines field_height+1..(2*field_height)-1
 * on the odd frame lines -- a bijection over the frame's (2*field_height)-1
 * lines (the second field's trailing line is padding and has no frame line).
 *
 * The mapping is not monotonic: vertically adjacent frame lines belong to
 * different fields and so differ in signal number by about field_height. A
 * crop whose first frame line is above its last can therefore have a
 * first_active_frame_line whose signal number exceeds its last's (e.g. the
 * 525-line default active region, frame lines 39..524, is signal lines
 * 283..263).
 *
 * field_height is read from v, so a conversion matches the numbering the
 * decoder applies. Out-of-range input returns CHD_E_OUT_OF_RANGE: frame lines
 * run 0..(2*field_height)-2, signal lines 1..(2*field_height)-1.
 *
 * These map the SOURCE raster only. chd_plane_info's first_frame_row and
 * chd_dropout_span's y are OUTPUT rows (measured from the committed, cropped and
 * padded frame); relate them by offsetting for the crop origin and padding
 * first, not by calling these directly. */
chd_status_t chd_video_frame_line_to_signal_line(const chd_video_t *v,
                                                  int32_t frame_line, int32_t *out);
chd_status_t chd_video_signal_line_to_frame_line(const chd_video_t *v,
                                                  int32_t signal_line, int32_t *out);

/* Convert between a stored-row sample position and the sample numbering of
 * the 4fsc interface standards (SMPTE ST 244 for 525-line, EBU Tech 3280-E
 * for 625-line; SECAM converts with the 625/50 geometry).
 *
 * A row sample is the 0-indexed position within a stored row: the space
 * chd_video_info's first_/last_active_sample and the CHD_OPT_*_ACTIVE_SAMPLE
 * options live in. A standard sample is the interface standard's numbering
 * of the same line, where sample 0 is the first sample of the digital active
 * line and 0H falls late in the row. The two differ by a rotation (wrapping
 * at field_width) that depends on the source's horizontal alignment
 * (sync-start vs blanking-start rows), read from v so the result matches the
 * raster being decoded: the ST 244 digital active line, standard samples
 * 0..767, is row samples 125..892 on a line-locked NTSC source and 142..909
 * on a subcarrier-locked (blanking-start) one.
 *
 * The rotation uses the uniform-row convention. On subcarrier-locked PAL,
 * 0H drifts by 4/625 of a sample per line across the frame, so a converted
 * window is exact at the frame's first line and sub-sample off elsewhere
 * (the standards' own uniform-row storage shares this property).
 *
 * Both numberings run 0..field_width-1; anything else returns
 * CHD_E_OUT_OF_RANGE. A source whose rows are not the standard 4fsc line
 * width returns CHD_E_UNSUPPORTED. These map the SOURCE raster only;
 * chd_dropout_span's x_start/x_end are output columns. */
chd_status_t chd_video_standard_sample_to_row_sample(const chd_video_t *v,
                                                     int32_t standard_sample, int32_t *out);
chd_status_t chd_video_row_sample_to_standard_sample(const chd_video_t *v,
                                                     int32_t row_sample, int32_t *out);

/* Extra sources for multi-source dropout correction. metadata_path_or_null
 * follows the same resolution as the matching open function. */
chd_status_t chd_video_add_extra_source_composite(chd_video_t *v,
                                                  const char *path,
                                                  const char *metadata_path_or_null);
chd_status_t chd_video_add_extra_source_yc(chd_video_t *v,
                                           const char *luma_path,
                                           const char *chroma_path,
                                           const char *metadata_path_or_null);

#ifdef __cplusplus
}
#endif
#endif
