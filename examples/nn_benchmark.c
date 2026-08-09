/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Per-frame timing for the neural decoders, through the public C ABI only.
 *
 * Decodes the same frames of a composite .tbc once per requested execution
 * provider and prints a comparison against the CPU baseline, reporting the
 * provider each run actually resolved to so a run that quietly landed on the
 * CPU is visible rather than merely slow.
 *
 * The CPU baseline runs FIRST and every other backend is optional. Both are
 * deliberate: a GPU backend can fail for reasons that have nothing to do with
 * this program — a provider whose libraries do not resolve, a driver that is
 * absent, a GPU architecture its kernel libraries do not cover — and when that
 * happens the run should still yield the baseline and the surviving backends.
 *
 * Recoverable failures (a provider that will not attach, a model that will not
 * load) are caught, reported as FAILED for that backend, and the sweep
 * continues. A backend can also die *uncatchably*: ROCm's rocBLAS calls
 * abort() mid-inference on a GPU architecture it ships no kernels for, which
 * takes the process with it. Nothing in-process can survive that, which is the
 * other reason the baseline goes first — and why a caller sweeping several
 * decoders should run each in its own process.
 *
 * Usage:
 *   nn_benchmark <file.tbc> <kind> <model.onnx|-> [options]
 *     kind      nntransform3d | color_cnn | luma_sep | luma_sep_frame | ntsc3d
 *     -b LIST   comma-separated backends to compare, in order. Default "cpu".
 *               Names: auto|cpu|cuda|tensorrt|migraphx|directml|coreml|coreml_native
 *               "cpu" is prepended unless already present or -no-cpu is given.
 *     -no-cpu   omit the CPU baseline (then speedups are not reported)
 *     -M PATH   artifact for the native CoreML backend. That backend loads a
 *               .mlpackage/.mlmodelc while every ONNX Runtime backend loads the
 *               .onnx, so comparing them in one sweep needs both paths.
 *     -n N      timed frames (default 8)
 *     -w N      warm-up frames, timed separately (default 2)
 *     -s SCALE  nn_input_magnitude_scale (nnTransform3D; 1.0 v1, 128.0 v2)
 *     -c DIR    engine cache dir; "" disables caching (cold-start timing)
 *     -t N      thread_count (default 1)
 *     -u UNITS  native-CoreML compute units: cpu_and_gpu (default) | all |
 *               cpu_only. "all" adds the ANE, which only engages for an
 *               fp16-converted .mlpackage.
 *     -p PREC   compute precision: fp32 (default) | fp16 (backends without a
 *               reduced-precision engine mode ignore it)
 *
 * `ntsc3d` takes no model and exists as the non-neural baseline; pass "-" for
 * the model argument and it ignores -b.
 *
 * Exit status is 0 when at least one backend produced a measurement, 1 when
 * every requested backend failed or the inputs were unusable. Benchmarks are
 * not a gate — the test suite is — so a partially complete sweep is a success.
 */

/* Examples build by default on every platform, so this has to compile with
 * MSVC as well as gcc/clang. clock_gettime is POSIX — absent on MSVC, and
 * hidden by glibc under -std=c11 without the feature macro. */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif

#include <chromadec/chromadec.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <time.h>
#endif

#define MAX_BACKENDS 8

/* Monotonic milliseconds. Wall-clock would do for frame times this long, but a
 * clock that can step backwards has no business in a benchmark. */
static double now_ms(void)
{
#if defined(_WIN32)
    LARGE_INTEGER freq, ticks;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&ticks);
    return (double)ticks.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
#endif
}

static int cmp_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static const char *backend_name(chd_nn_backend_t b)
{
    switch (b) {
        case CHD_NN_BACKEND_AUTO: return "auto";
        case CHD_NN_ORT_AUTO:     return "ort-auto";
        case CHD_NN_ORT_CPU:      return "ORT CPU";
        case CHD_NN_ORT_CUDA:     return "ORT CUDA";
        case CHD_NN_ORT_TENSORRT: return "ORT TensorRT";
        case CHD_NN_ORT_COREML:   return "ORT CoreML";
        case CHD_NN_ORT_DIRECTML: return "ORT DirectML";
        case CHD_NN_ORT_MIGRAPHX: return "ORT MIGraphX";
        case CHD_NN_COREML:       return "CoreML (native)";
        default:                  return "unknown";
    }
}

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

static int parse_kind(const char *s, chd_decoder_kind_t *out, int *needs_model)
{
    *needs_model = 1;
    if (!strcmp(s, "nntransform3d"))       *out = CHD_DEC_NN_TRANSFORM3D;
    else if (!strcmp(s, "color_cnn"))      *out = CHD_DEC_LDZEUG_COLOR_CNN;
    else if (!strcmp(s, "luma_sep"))       *out = CHD_DEC_LDZEUG_LUMA_SEP;
    else if (!strcmp(s, "luma_sep_frame")) *out = CHD_DEC_LDZEUG_LUMA_SEP_FRAME;
    else if (!strcmp(s, "ntsc3d"))       { *out = CHD_DEC_NTSC_3D; *needs_model = 0; }
    else return 0;
    return 1;
}

struct cell {
    chd_nn_backend_t requested;
    chd_nn_backend_t active;
    int    ok;
    double mean, median, load_ms;
    char   error[512];
};

struct run_config {
    const char        *tbc;
    const char        *kind_s;
    chd_decoder_kind_t kind;
    const char        *model_path;
    int                needs_model;
    int                n_timed, n_warm, threads;
    double             scale;
    const char        *cache_dir;   /* NULL = library default */
    int                have_cache_opt;
    const char        *coreml_model; /* native-CoreML artifact, NULL if none */
    chd_nn_coreml_compute_t    coreml_units;
    chd_nn_compute_precision_t precision;
};

/* Native CoreML loads a .mlpackage; the ORT backends load the .onnx. */
static const char *model_for(const struct run_config *cfg, chd_nn_backend_t b)
{
    if (b == CHD_NN_COREML && cfg->coreml_model != NULL) return cfg->coreml_model;
    return cfg->model_path;
}

/* Measure one backend. Every failure path fills cell->error and returns without
 * touching cell->ok, so the caller can report it and move on. */
static void run_cell(const struct run_config *cfg, struct cell *cell)
{
    chd_video_t   *video = NULL;
    chd_nn_model_t *model = NULL;
    chd_decoder_t *dec   = NULL;
    double        *t     = NULL;
    chd_status_t   st;

#define CELL_FAIL(...) do { \
        snprintf(cell->error, sizeof cell->error, __VA_ARGS__); \
        goto cleanup; \
    } while (0)

    st = chd_video_open_composite(cfg->tbc, NULL, NULL, &video);
    if (st != CHD_OK) CELL_FAIL("open: %s (%s)", chd_status_str(st),
                                chd_last_error() ? chd_last_error() : "-");

    if (cfg->needs_model) {
        chd_nn_session_opts_t nopts;
        chd_nn_session_opts_default(&nopts);
        nopts.backend = cell->requested;
        if (cfg->have_cache_opt) nopts.engine_cache_dir = cfg->cache_dir;
        nopts.coreml_compute = cfg->coreml_units;
        nopts.precision      = cfg->precision;

        const double t0 = now_ms();
        st = chd_nn_model_load_from_file(model_for(cfg, cell->requested), &nopts, &model);
        cell->load_ms = now_ms() - t0;
        if (st != CHD_OK) CELL_FAIL("model load: %s (%s)", chd_status_str(st),
                                    chd_last_error() ? chd_last_error() : "-");
        chd_nn_model_get_active_backend(model, &cell->active);
    }

    st = chd_decoder_create(video, cfg->kind, &dec);
    if (st != CHD_OK) CELL_FAIL("decoder create: %s", chd_status_str(st));
    chd_decoder_set_option_i32(dec, CHD_OPT_THREAD_COUNT, cfg->threads);
    chd_decoder_set_option_i32(dec, CHD_OPT_PADDING_MULTIPLE, 1);
    if (cfg->scale >= 0.0 &&
        chd_decoder_has_option(dec, CHD_OPT_NN_INPUT_MAGNITUDE_SCALE) == 1) {
        chd_decoder_set_option_f64(dec, CHD_OPT_NN_INPUT_MAGNITUDE_SCALE, cfg->scale);
    }
    if (model) {
        st = chd_decoder_set_nn_model(dec, model);
        if (st != CHD_OK) CELL_FAIL("attach model: %s", chd_status_str(st));
    }
    st = chd_decoder_commit(dec);
    if (st != CHD_OK) CELL_FAIL("commit: %s (%s)", chd_status_str(st),
                                chd_last_error() ? chd_last_error() : "-");

    const int total = cfg->n_warm + cfg->n_timed;
    t = (double *)malloc(sizeof(double) * (size_t)total);
    if (t == NULL) CELL_FAIL("out of memory");

    /* nnTransform3D reads one frame behind and two ahead, so start at 1;
     * boundary frames decode against black padding and time differently. */
    for (int i = 0; i < total; ++i) {
        chd_frame_t *frame = NULL;
        const double t0 = now_ms();
        st = chd_decode_frame(dec, 1 + i, &frame);
        if (st != CHD_OK) CELL_FAIL("decode frame %d: %s (%s)", 1 + i,
                                    chd_status_str(st),
                                    chd_last_error() ? chd_last_error() : "-");
        t[i] = now_ms() - t0;
        chd_frame_free(frame);
        if (i < cfg->n_warm) printf("    warm-up %d: %8.1f ms\n", i, t[i]);
    }

    double sum = 0.0;
    for (int i = cfg->n_warm; i < total; ++i) sum += t[i];
    cell->mean = sum / cfg->n_timed;

    qsort(t + cfg->n_warm, (size_t)cfg->n_timed, sizeof(double), cmp_double);
    cell->median = t[cfg->n_warm + cfg->n_timed / 2];
    cell->ok = 1;

#undef CELL_FAIL
cleanup:
    free(t);
    if (dec)   chd_decoder_free(dec);
    if (video) chd_video_free(video);
    if (model) chd_nn_model_free(model);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <file.tbc> <kind> <model.onnx|-> [-b list] "
                        "[-no-cpu] [-n frames] [-w warmup] [-s scale] "
                        "[-c cachedir] [-t threads] [-u units] [-p precision]\n", argv[0]);
        return 2;
    }

    struct run_config cfg = { .tbc = argv[1], .kind_s = argv[2], .model_path = argv[3],
                              .n_timed = 8, .n_warm = 2, .threads = 1, .scale = -1.0,
                              .cache_dir = NULL, .have_cache_opt = 0,
                              .coreml_units = CHD_NN_COREML_CPU_AND_GPU,
                              .precision = CHD_NN_PRECISION_FP32 };
    if (!parse_kind(cfg.kind_s, &cfg.kind, &cfg.needs_model)) {
        fprintf(stderr, "unknown kind '%s'\n", cfg.kind_s);
        return 2;
    }

    chd_nn_backend_t requested[MAX_BACKENDS];
    int n_requested = 0, want_cpu = 1, have_b = 0;
    char blist[256] = "";

    for (int i = 4; i < argc; ++i) {
        if (!strcmp(argv[i], "-b") && i + 1 < argc) {
            snprintf(blist, sizeof blist, "%s", argv[++i]);
            have_b = 1;
        } else if (!strcmp(argv[i], "-no-cpu")) want_cpu = 0;
        else if (!strcmp(argv[i], "-M") && i + 1 < argc) cfg.coreml_model = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) cfg.n_timed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) cfg.n_warm  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) cfg.threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) cfg.scale   = atof(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) {
            cfg.cache_dir = argv[++i]; cfg.have_cache_opt = 1;
        } else if (!strcmp(argv[i], "-u") && i + 1 < argc) {
            const char *u = argv[++i];
            if      (!strcmp(u, "cpu_and_gpu")) cfg.coreml_units = CHD_NN_COREML_CPU_AND_GPU;
            else if (!strcmp(u, "all"))         cfg.coreml_units = CHD_NN_COREML_ALL;
            else if (!strcmp(u, "cpu_only"))    cfg.coreml_units = CHD_NN_COREML_CPU_ONLY;
            else { fprintf(stderr, "bad units '%s'\n", u); return 2; }
        } else if (!strcmp(argv[i], "-p") && i + 1 < argc) {
            const char *pr = argv[++i];
            if      (!strcmp(pr, "fp32")) cfg.precision = CHD_NN_PRECISION_FP32;
            else if (!strcmp(pr, "fp16")) cfg.precision = CHD_NN_PRECISION_FP16_ALLOWED;
            else { fprintf(stderr, "bad precision '%s'\n", pr); return 2; }
        } else { fprintf(stderr, "unknown arg '%s'\n", argv[i]); return 2; }
    }

    /* The CPU baseline goes first so that a GPU backend which aborts the
     * process cannot take it down with it. */
    if (cfg.needs_model && want_cpu) requested[n_requested++] = CHD_NN_ORT_CPU;
    if (cfg.needs_model && have_b) {
        for (char *tok = strtok(blist, ","); tok != NULL; tok = strtok(NULL, ",")) {
            chd_nn_backend_t b;
            if (!parse_backend(tok, &b)) { fprintf(stderr, "bad backend '%s'\n", tok); return 2; }
            if (b == CHD_NN_ORT_CPU && want_cpu) continue;   /* already first */
            if (n_requested >= MAX_BACKENDS) { fprintf(stderr, "too many backends\n"); return 2; }
            requested[n_requested++] = b;
        }
    }
    if (!cfg.needs_model) { requested[0] = CHD_NN_BACKEND_AUTO; n_requested = 1; }
    if (n_requested == 0) { fprintf(stderr, "no backends to run\n"); return 2; }

    if (cfg.needs_model && chd_has_feature("nn") != 1) {
        fprintf(stderr, "this build has no NN support\n");
        return 1;
    }
    if (chd_init() != CHD_OK) { fprintf(stderr, "chd_init failed\n"); return 1; }

    printf("file    : %s\n", cfg.tbc);
    printf("decoder : %s", cfg.kind_s);
    if (cfg.needs_model) {
        printf("   model %s", cfg.model_path);
        if (cfg.coreml_model) printf("   (native CoreML: %s)", cfg.coreml_model);
        if (cfg.scale >= 0.0) printf("   scale %.1f", cfg.scale);
    }
    printf("\n");
    printf("timing  : %d warm-up + %d timed frames, thread_count=%d\n\n",
           cfg.n_warm, cfg.n_timed, cfg.threads);

    struct cell cells[MAX_BACKENDS];
    memset(cells, 0, sizeof cells);
    for (int i = 0; i < n_requested; ++i) {
        cells[i].requested = requested[i];
        cells[i].active    = CHD_NN_BACKEND_AUTO;
        printf("  %s:\n", cfg.needs_model ? backend_name(requested[i]) : "(no NN backend)");
        fflush(stdout);
        run_cell(&cfg, &cells[i]);
        if (cells[i].ok) {
            printf("    %.1f ms/frame mean, %.1f median  (%.3f fps)\n",
                   cells[i].mean, cells[i].median, 1000.0 / cells[i].mean);
            if (cfg.needs_model) {
                printf("    resolved to %s%s, load+compile %.1f ms\n",
                       backend_name(cells[i].active),
                       (requested[i] != CHD_NN_BACKEND_AUTO &&
                        cells[i].active != requested[i]) ? "  <-- NOT what was requested" : "",
                       cells[i].load_ms);
            }
        } else {
            printf("    FAILED: %s\n", cells[i].error);
        }
        printf("\n");
        fflush(stdout);
    }

    /* Summary, with speedup against the CPU baseline when there is one. */
    double baseline = 0.0;
    for (int i = 0; i < n_requested; ++i) {
        if (cells[i].ok && cells[i].active == CHD_NN_ORT_CPU) { baseline = cells[i].mean; break; }
    }

    printf("%-16s %12s %12s %10s\n", "backend", "mean ms", "median ms", "vs CPU");
    int measured = 0;
    for (int i = 0; i < n_requested; ++i) {
        const char *nm = cfg.needs_model ? backend_name(cells[i].requested) : cfg.kind_s;
        if (!cells[i].ok) { printf("%-16s %12s %12s %10s\n", nm, "-", "-", "FAILED"); continue; }
        ++measured;
        if (baseline > 0.0 && cells[i].mean > 0.0 && cells[i].active != CHD_NN_ORT_CPU) {
            printf("%-16s %12.1f %12.1f %9.1fx\n", nm, cells[i].mean, cells[i].median,
                   baseline / cells[i].mean);
        } else {
            printf("%-16s %12.1f %12.1f %10s\n", nm, cells[i].mean, cells[i].median,
                   baseline > 0.0 ? "baseline" : "-");
        }
        /* Machine-readable, one line per measured cell. */
        fprintf(stderr, "RESULT\t%s\t%s\t%.2f\t%.2f\t%.1f\n", cfg.kind_s,
                cfg.needs_model ? backend_name(cells[i].active) : "n/a",
                cells[i].mean, cells[i].median, cells[i].load_ms);
    }
    if (measured < n_requested) {
        printf("\n%d of %d backends failed; see FAILED lines above.\n",
               n_requested - measured, n_requested);
    }

    chd_shutdown();
    return measured > 0 ? 0 : 1;
}
