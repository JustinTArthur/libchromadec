/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Decode frames through one pinned NN backend and write the output planes
 * raw, for byte-level comparison between two configurations of the same
 * model — engine precisions, execution providers, or backends.
 *
 * Usage:
 *   nn_plane_dump <file.tbc> <model> <backend> <first> <n> <out.raw> [options]
 *     backend   auto | cpu | cuda | tensorrt | migraphx | directml | coreml |
 *               coreml_native
 *     first, n  first frame index and frame count to decode
 *     -s SCALE  nn_input_magnitude_scale (nnTransform3D; 1.0 v1, 128.0 v2)
 *     -p PREC   compute precision: fp32 (default) | fp16
 *     -u UNITS  native-CoreML compute units: cpu_and_gpu (default) | all |
 *               cpu_only
 *
 * Output: for each frame, planes Y, Cb, Cr as contiguous uint16 rows
 * (yuv444p16, width x height each, no padding). Compare two dumps
 * element-wise; identical configurations produce identical bytes.
 *
 * "DONE" on stdout marks all frames written and flushed. Judge a run by that
 * sentinel and the file size rather than the exit status alone: GPU runtimes
 * have historically crashed in library teardown after main returns, which
 * scripts would otherwise misread as a decode failure.
 */

#include <chromadec/chromadec.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_backend(const char *s, chd_nn_backend_t *out)
{
    if (!strcmp(s, "auto"))          *out = CHD_NN_BACKEND_AUTO;
    else if (!strcmp(s, "cpu"))      *out = CHD_NN_ORT_CPU;
    else if (!strcmp(s, "cuda"))     *out = CHD_NN_ORT_CUDA;
    else if (!strcmp(s, "tensorrt")) *out = CHD_NN_ORT_TENSORRT;
    else if (!strcmp(s, "migraphx")) *out = CHD_NN_ORT_MIGRAPHX;
    else if (!strcmp(s, "directml")) *out = CHD_NN_ORT_DIRECTML;
    else if (!strcmp(s, "coreml"))   *out = CHD_NN_ORT_COREML;
    else if (!strcmp(s, "coreml_native")) *out = CHD_NN_COREML;
    else return 0;
    return 1;
}

static void fail(const char *what, chd_status_t st)
{
    fprintf(stderr, "%s: %s (%s)\n", what, chd_status_str(st),
            chd_last_error() ? chd_last_error() : "-");
    exit(1);
}

int main(int argc, char **argv)
{
    if (argc < 7) {
        fprintf(stderr, "usage: %s <file.tbc> <model> <backend> <first> <n> "
                        "<out.raw> [-s scale] [-p fp32|fp16] [-u units]\n", argv[0]);
        return 2;
    }
    const char *tbc = argv[1], *model_path = argv[2], *out_path = argv[6];
    chd_nn_backend_t backend;
    if (!parse_backend(argv[3], &backend)) { fprintf(stderr, "bad backend '%s'\n", argv[3]); return 2; }
    const int first = atoi(argv[4]), n = atoi(argv[5]);

    double scale = -1.0;
    chd_nn_compute_precision_t precision = CHD_NN_PRECISION_FP32;
    chd_nn_coreml_compute_t units = CHD_NN_COREML_CPU_AND_GPU;
    for (int i = 7; i < argc; ++i) {
        if (!strcmp(argv[i], "-s") && i + 1 < argc) scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
            const char *pr = argv[++i];
            if      (!strcmp(pr, "fp32")) precision = CHD_NN_PRECISION_FP32;
            else if (!strcmp(pr, "fp16")) precision = CHD_NN_PRECISION_FP16_ALLOWED;
            else { fprintf(stderr, "bad precision '%s'\n", pr); return 2; }
        } else if (!strcmp(argv[i], "-u") && i + 1 < argc) {
            const char *u = argv[++i];
            if      (!strcmp(u, "cpu_and_gpu")) units = CHD_NN_COREML_CPU_AND_GPU;
            else if (!strcmp(u, "all"))         units = CHD_NN_COREML_ALL;
            else if (!strcmp(u, "cpu_only"))    units = CHD_NN_COREML_CPU_ONLY;
            else { fprintf(stderr, "bad units '%s'\n", u); return 2; }
        } else { fprintf(stderr, "unknown arg '%s'\n", argv[i]); return 2; }
    }

    chd_video_t *video = NULL;
    chd_status_t st = chd_video_open_composite(tbc, NULL, NULL, &video);
    if (st != CHD_OK) fail("open", st);

    chd_nn_session_opts_t nopts;
    chd_nn_session_opts_default(&nopts);
    nopts.backend        = backend;
    nopts.precision      = precision;
    nopts.coreml_compute = units;
    chd_nn_model_t *model = NULL;
    st = chd_nn_model_load_from_file(model_path, &nopts, &model);
    if (st != CHD_OK) fail("model load", st);
    chd_nn_backend_t active;
    chd_nn_model_get_active_backend(model, &active);
    fprintf(stderr, "active backend: %d\n", (int)active);

    chd_decoder_t *dec = NULL;
    st = chd_decoder_create(video, CHD_DEC_NN_TRANSFORM3D, &dec);
    if (st != CHD_OK) fail("decoder create", st);
    chd_decoder_set_option_i32(dec, CHD_OPT_THREAD_COUNT, 1);
    chd_decoder_set_option_i32(dec, CHD_OPT_PADDING_MULTIPLE, 1);
    chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "yuv444p16");
    if (scale >= 0.0) {
        chd_decoder_set_option_f64(dec, CHD_OPT_NN_INPUT_MAGNITUDE_SCALE, scale);
    }
    st = chd_decoder_set_nn_model(dec, model);
    if (st != CHD_OK) fail("attach model", st);
    st = chd_decoder_commit(dec);
    if (st != CHD_OK) fail("commit", st);

    FILE *out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }

    static const chd_plane_t planes[3] = { CHD_PLANE_Y, CHD_PLANE_CB, CHD_PLANE_CR };
    for (int i = 0; i < n; ++i) {
        chd_frame_t *frame = NULL;
        st = chd_decode_frame(dec, first + i, &frame);
        if (st != CHD_OK) fail("decode", st);
        for (int p = 0; p < 3; ++p) {
            chd_plane_info_t pi;
            st = chd_frame_get_plane_info(frame, planes[p], &pi);
            if (st != CHD_OK) fail("plane info", st);
            const void *data = NULL;
            ptrdiff_t stride = 0;
            st = chd_frame_get_plane(frame, planes[p], &data, &stride);
            if (st != CHD_OK) fail("plane", st);
            for (int r = 0; r < pi.height; ++r) {
                fwrite((const char *)data + (ptrdiff_t)r * stride, 2, (size_t)pi.width, out);
            }
        }
        chd_frame_free(frame);
        fprintf(stderr, "frame %d ok\n", first + i);
    }
    fclose(out);
    printf("DONE\n");
    fflush(stdout);

    chd_decoder_free(dec);
    chd_video_free(video);
    chd_nn_model_free(model);
    chd_shutdown();
    return 0;
}
