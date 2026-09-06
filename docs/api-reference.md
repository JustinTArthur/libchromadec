# C API reference

The complete public C ABI of libchromadec. Every exported symbol is prefixed
`chd_` and declared in a header under `<chromadec/>`. Include the umbrella
header to pull in the whole surface:

```c
#include <chromadec/chromadec.h>
```

or include only the headers you need (`<chromadec/video.h>`,
`<chromadec/decoder.h>`, …). The library is pure C at the boundary and C++17
internally; nothing in these headers requires a C++ compiler to consume.

## Conventions

These rules hold across the whole ABI. They are stated once here rather than
repeated on every function.

### Return values and error detail

- A function that can fail returns [`chd_status_t`](#status-codes); `CHD_OK`
  (`0`) is success, any other value is failure.
- Accessors that cannot meaningfully fail return their value directly
  (`chd_version`, `chd_cancel_is_requested`, `chd_nn_backend_is_available`,
  `chd_has_feature`).
- **Output parameters are written only on `CHD_OK`.** On failure, a `**out`
  handle is left untouched. Initialise your pointer to `NULL` and check the
  status before using it.
- After a failure, [`chd_last_error()`](#chd_last_error) returns a
  human-readable, thread-local detail string for the *most recent* failing
  call on the calling thread. [`chd_status_str()`](#chd_status_str) maps a
  status code to a stable static string.
- **Check the status first, then read the detail.** The detail is only
  meaningful once a call has returned something other than `CHD_OK`; a failure
  the library recovered from internally can leave a detail behind, so a
  non-empty string is not on its own evidence that your call failed.
- The detail is the reason recorded by whichever layer actually detected the
  failure, prefixed with the entry point you called. Opening a `.tbc` whose
  sidecar lists fewer fields than it declares reports that mismatch, not
  "failed to read sidecar metadata"; a decoder that rejects its input names
  what it rejected. The status code narrows the category, the detail names the
  cause.
- Failures always travel this return path. The [diagnostic
  sink](#diagnostics) is a separate channel carrying commentary that has no
  return path of its own, and it is silent until you install a callback.

### Ownership and lifetime

- Every `chd_*_create` / `chd_*_open_*` / `chd_*_load` that yields a handle
  transfers ownership to the caller. Release it with the matching
  `chd_*_free`. Freeing `NULL` is always safe.
- Handles are **opaque**. You only ever hold a `chd_video_t *`,
  `chd_decoder_t *`, etc. Their layout is private and may change between
  minor versions without breaking your binary.
- Decoded frames ([`chd_frame_t`](#frames)) are owned by the caller and freed
  with [`chd_frame_free`](#chd_frame_free). This includes frames delivered to an
  async callback.
- Plane pointers from [`chd_frame_get_plane`](#chd_frame_get_plane) are
  **borrowed**: do not free them; they remain valid only until the owning
  frame is freed.

### Threading

- The error-detail string is **thread-local**. Each thread sees only the
  result of its own calls.
- A single `chd_decoder_t` is **not** safe for concurrent option mutation and
  decoding. Setting options / committing must not overlap a `chd_decode_frame`
  call on the same decoder.
- Distinct handles are independent and may be used concurrently from different
  threads.
- [`chd_decode_frames_async`](#chd_decode_frames_async) invokes your callback
  from internal worker threads; the callback must be thread-safe. So does the
  [diagnostic sink](#diagnostics), which is process-wide rather than
  per-handle.

### Lifecycle

Call [`chd_init`](#chd_init) once before any other call.
[`chd_shutdown`](#chd_shutdown) is an optional deterministic teardown for
NN state; skipping it is safe. See [Library lifecycle](#library-lifecycle).

The [diagnostic sink](#diagnostics) is the exception: `chd_set_log_callback`
and `chd_set_log_level` may be called before `chd_init`, which is where you
want them if you would rather not miss anything. The log state is never torn
down, so they are equally safe from an `atexit` handler or a static
destructor.

### Enums

Public enums are declared through the `CHD_ENUM` macro from
`<chromadec/enum.h>`, included for you and never written in consumer code:

```c
typedef CHD_ENUM(chd_log_level) { ... } chd_log_level_t;
```

In C++ and C23 this fixes the underlying type to `int32_t`; C11 sees a plain
enum with the same 4-byte representation. Every `int32_t` is a valid value of
the enum type, so out-of-range arguments are rejected by validation instead of
being undefined behaviour.

---

## Library lifecycle

Declared in `<chromadec/video.h>` (lifecycle) and `<chromadec/version.h>`
(version / feature probes).

### chd_init

```c
chd_status_t chd_init(void);
```

Initialise the library. Call once at startup before any other `chd_*` call.
Idempotent and currently cheap, but always pair it with a successful return
check; future versions may perform real one-time setup here.

### chd_shutdown

```c
void chd_shutdown(void);
```

Tear down process-wide library state: currently the ONNX Runtime environment
singleton created the first time an NN model is loaded.

!!! note "Optional, and never automatic"
    Calling it is a choice, not a requirement. Without it the ONNX Runtime
    environment is deliberately left alive through process exit — tearing it
    down from static destructors or `atexit` would unload GPU
    execution-provider libraries while their runtimes' own exit handlers are
    still queued, which is a crash, so the library never does either. Call
    `chd_shutdown()` at most once, after every decoder and model is freed,
    when you want the teardown to happen at a moment you control (leak
    checkers, hosts that reload the library). If no NN model was ever
    loaded, it is a harmless no-op.

### chd_version / chd_version_string

```c
void        chd_version(int *major, int *minor, int *patch);
const char *chd_version_string(void);   /* e.g. "0.1.0" */
```

Runtime library version. Any of the three out-pointers to `chd_version` may be
`NULL`. The string returned by `chd_version_string` is static and must not be
freed. Compile-time equivalents are the `CHROMADEC_VERSION_*` macros in
`<chromadec/version.h>`.

### chd_has_feature

```c
int chd_has_feature(const char *feature);   /* 1 = compiled in, 0 = not */
```

Query optional build features. Recognised names: `"nn"`, `"onnxruntime"`,
`"coreml"`, `"cuda"`, `"rocm"`, `"fftw"`, `"sqlite"`. Returns `0` for `NULL` or
any unknown name.

`"nn"` reports whether the neural-decoder framework is present — true when *any*
inference backend is built. The individual backends have their own flags:
`"onnxruntime"` (the ONNX Runtime backend, build option `with_onnxruntime`) and
`"coreml"` (the native CoreML backend, build option `with_coreml`). The two are
independent: a macOS build with `-Dwith_onnxruntime=false -Dwith_coreml=enabled`
reports `"nn"`=1, `"coreml"`=1, `"onnxruntime"`=0, and runs the ldzeug and
nnTransform3D decoders entirely on native CoreML (`.onnx` models and the
`CHD_NN_ORT_*` backends are then unavailable — see
[`chd_nn_model_load_from_file`](#chd_nn_model_load_from_file)).

---

## Error handling

Declared in `<chromadec/errors.h>`.

### Status codes

`chd_status_t` is the return type of every fallible call.

| Code                            | Meaning                                                                                   |
|---------------------------------|-------------------------------------------------------------------------------------------|
| `CHD_OK`                        | Success (value `0`).                                                                      |
| `CHD_E_INVALID_ARG`             | A null/out-of-domain argument, or a call made out of order (e.g. decoding before commit). |
| `CHD_E_FILE_NOT_FOUND`          | An input or sidecar path does not exist.                                                  |
| `CHD_E_IO`                      | Read/write failure on an otherwise-present file.                                          |
| `CHD_E_FORMAT_UNSUPPORTED`      | The container/sample encoding is not handled.                                             |
| `CHD_E_METADATA_MISSING`        | Required metadata (sidecar) could not be located.                                         |
| `CHD_E_METADATA_CORRUPT`        | Metadata was found but failed to parse.                                                   |
| `CHD_E_PRESET_UNKNOWN`          | An unknown preset name was requested.                                                     |
| `CHD_E_DECODER_UNKNOWN`         | Unknown decoder kind.                                                                     |
| `CHD_E_DECODER_INCOMPATIBLE`    | Decoder kind is invalid for this video standard/encoding, or the frame geometry is outside what it can decode. |
| `CHD_E_NN_MODEL_LOAD`           | The NN model file failed to load.                                                         |
| `CHD_E_NN_BACKEND_UNAVAILABLE`  | The requested inference backend is not available in this build/host.                      |
| `CHD_E_NN_INFERENCE`            | Inference failed at runtime.                                                              |
| `CHD_E_OUT_OF_RANGE`            | A frame index (or similar) is outside the valid range.                                    |
| `CHD_E_CANCELLED`               | The operation was cancelled via a [`chd_cancel_t`](#cancellation).                        |
| `CHD_E_INTERNAL`                | An unexpected internal error.                                                             |
| `CHD_E_OOM`                     | Allocation failure.                                                                       |
| `CHD_E_UNSUPPORTED`             | The query does not apply to this object (e.g. a chroma-ident query on a non-4:4:0 frame). |

### chd_status_str

```c
const char *chd_status_str(chd_status_t s);
```

Map a status code to a stable, static, English string (e.g.
`"CHD_E_FILE_NOT_FOUND"`-style). Never `NULL`; never freed. Suitable for logs
and assertions; it does not vary by call site.

### chd_last_error

```c
const char *chd_last_error(void);
```

Return a thread-local, human-readable description of the **most recent failing
call on the calling thread**, including call-site context the status code
alone cannot convey (which path, which option, etc.). The pointer is valid
until the next `chd_*` call on the same thread. Never freed, and never `NULL`:
with nothing yet recorded on this thread it returns an empty string.

Read it only in response to a non-`CHD_OK` status. The library records a detail
wherever it detects a failure, including failures it then handles itself, so
the string can be left non-empty by a call that went on to succeed. Nothing
resets it on success; [`chd_clear_last_error`](#chd_clear_last_error) is there
if you would rather manage that yourself.

The message describes the input, not the internals: it names the entry point
you called and the file, option, or metadata field at fault, never the internal
class that noticed.

### chd_clear_last_error

```c
void chd_clear_last_error(void);
```

Reset the calling thread's error-detail string to empty.

---

## Diagnostics { #diagnostics }

Declared in `<chromadec/log.h>`.

Beyond pass/fail, a decode produces commentary worth surfacing: which sidecar
was picked up, how many burst lines were measurable, why a SECAM field ident
fell back. None of that has a return value to travel on, so it goes to a sink
you install.

**The library never writes to a console on its own.** No sink is installed by
default, and until one is, nothing is emitted anywhere. That is deliberate: a
library does not own the process's stderr, and a consumer with its own log
system, a GUI, or a host plugin API wants the messages routed there instead.
Failures are always reported through [`chd_status_t`](#status-codes) plus
[`chd_last_error()`](#chd_last_error) whether or not anything is listening
here, so ignoring this section costs you no error reporting.

The two channels do overlap in one place. A failure that reaches you as a
status is also announced here, at `CHD_LOG_ERROR`, carrying the same text — so
a program that reports both would say it twice. Those messages are marked
[`CHD_LOG_F_RETURNED`](#chd_log_f_returned), which is what tells them apart
from an error that has no other way of reaching anyone.

### chd_set_log_callback

```c
typedef CHD_ENUM(chd_log_level) {
    CHD_LOG_DEBUG = 0,
    CHD_LOG_INFO  = 1,
    CHD_LOG_WARN  = 2,
    CHD_LOG_ERROR = 3,
    CHD_LOG_OFF   = 4   /* threshold only; never delivered */
} chd_log_level_t;

typedef unsigned int chd_log_flags_t;
#define CHD_LOG_F_RETURNED 0x1u

typedef void (*chd_log_fn)(chd_log_level_t level, chd_log_flags_t flags,
                           const char *message, void *user_data);

void chd_set_log_callback(chd_log_fn fn_or_null, void *user_data);
```

Install the sink, or pass `NULL` to uninstall and go back to silence. The
`user_data` pointer is handed back to every call, untouched by the library.

- `message` is NUL-terminated UTF-8 with **no trailing newline**, and is valid
  only for the duration of the call. Copy it if you need to keep it.
- The sink is called from whichever thread produced the message, including the
  decode worker threads behind
  [`chd_decode_frames_async`](#chd_decode_frames_async), so it **must be
  thread-safe**.
- The sink **must not call back into libchromadec**, this function included.
- Once `chd_set_log_callback` returns, the previous sink is neither running nor
  reachable, so freeing its `user_data` at that point is safe. The call blocks
  for as long as a dispatch already in flight takes to return, which is one
  reason to keep a sink quick.
- The sink is **process-wide, not per-handle**, and there is one of it. In a
  host that loads several consumers of libchromadec into the same process (a
  plugin architecture, say), the last one to install wins and the library will
  not chain to the sink it replaced. If you are writing a plugin rather than an
  application, consider installing a sink only when your host has actually
  asked for the messages.

### Message flags { #chd_log_f_returned }

`flags` is a bitmask describing the message beyond its severity. It is not an
enumeration, because a combination of enumerators is not itself an enumerator.
Test the bits you know and ignore the rest; future versions may set more.

|Flag|Meaning|
|---|---|
|`CHD_LOG_F_RETURNED`|This message is also the calling thread's [`chd_last_error()`](#chd_last_error) detail, so the failure it describes is already on its way back to you as a non-`CHD_OK` [`chd_status_t`](#status-codes).|

Only `CHD_LOG_ERROR` messages are ever marked, and a marked message is never
the *only* notice you get. That makes the dedup rule simple: if your program
already reports failures from the return path, drop marked messages.

```c
static void log_sink(chd_log_level_t level, chd_log_flags_t flags,
                     const char *message, void *user_data)
{
    /* The caller is about to see this as a status; reporting it here as well
       would say the same thing twice. */
    if (flags & CHD_LOG_F_RETURNED) return;
    host_log(user_data, level, message);
}
```

Keep them instead if the sink is your program's only output, such as a log pane
in a GUI, or a `--verbose` trace where seeing the failure in sequence with the
diagnostics around it is the point.

An unmarked `CHD_LOG_ERROR` is the case that makes the flag worth having: it is
a failure with no return path of its own, so dropping *those* would lose them.

Note the converse does not hold: not every failing call emits a diagnostic at
all. Argument validation at the ABI boundary records the detail and returns
without saying anything here, since nothing has gone wrong that a log would
illuminate. The flag tells you a message duplicates a status, never that a
status will be accompanied by a message.

### chd_log_to_stderr

```c
void chd_log_to_stderr(void);
```

Install a built-in sink that writes `LEVEL: message` lines to stderr. A
convenience for command-line consumers that want the old
write-it-to-the-terminal behaviour in one call; it replaces any sink already
installed. It ignores `flags`, so a failure you also print from
`chd_last_error()` will appear twice; write your own three-line sink if that
matters.

### chd_set_log_level / chd_get_log_level { #chd_set_log_level }

```c
void            chd_set_log_level(chd_log_level_t min_level);
chd_log_level_t chd_get_log_level(void);
```

Drop messages below `min_level`. The default is `CHD_LOG_INFO`, so debug
output is opt-in even once a sink exists. `CHD_LOG_OFF` suppresses everything
without uninstalling the sink. A value outside the enum is clamped into it,
rather than silencing the library on a typo.

The threshold and the sink are both process-wide. Set them before you start
decoding; changing them mid-decode is safe but which in-flight messages make it
through is a race, by nature.

What each level costs you:

|Level|Volume|What it carries|
|---|---|---|
|`CHD_LOG_ERROR`|Rare|A failure that has no return path of its own, or one the library is about to report through `chd_status_t` as well. [`CHD_LOG_F_RETURNED`](#chd_log_f_returned) tells the two apart.|
|`CHD_LOG_WARN`|Rare|The library did something other than what the input asked for: a decoder fallback, or an out-of-range value in a sidecar replaced with the standard's default.|
|`CHD_LOG_INFO`|Per run, plus periodic progress on the batch pipeline path|Which source and sidecar were resolved, geometry, thread count, decode throughput.|
|`CHD_LOG_DEBUG`|High, and dominated by metadata and source parsing|Per-field and per-sidecar-record detail. Useful when a capture will not open; not something to leave installed for a long decode.|

Nothing is emitted per pixel or per scan line at any level.

### chd_log_is_enabled

```c
int chd_log_is_enabled(chd_log_level_t level);
```

Whether a message at `level` would reach a sink right now: non-zero if one is
installed and `level` is at or above the threshold. Suppressed messages are
already cheap inside the library, so this is for *your* side of the fence,
guarding diagnostic data you would otherwise build and throw away.

`CHD_LOG_OFF` always answers `0`. It is a threshold value, so no message
carries it and none can be delivered at it.

### chd_log_level_str

```c
const char *chd_log_level_str(chd_log_level_t level);
```

Stable static name for a level: `"DEBUG"`, `"INFO"`, `"WARN"`, `"ERROR"`,
`"OFF"`, or `"UNKNOWN"`. Never freed.

### Example

```c
static void log_sink(chd_log_level_t level, chd_log_flags_t flags,
                     const char *message, void *user_data)
{
    FILE *out = (FILE *)user_data;
    fprintf(out, "[chromadec/%s]%s %s\n", chd_log_level_str(level),
            (flags & CHD_LOG_F_RETURNED) ? " (returned)" : "", message);
}

chd_set_log_callback(log_sink, stdout);
chd_set_log_level(CHD_LOG_DEBUG);
```

When the ONNX Runtime backend is built in, ORT's own diagnostics are routed
through this sink too rather than to its default stderr logger, so a consumer
sees one stream. ORT applies its own threshold first, fixed at its warning
level when the environment is created on the first model load, so raising
`chd_set_log_level` to `CHD_LOG_DEBUG` reveals more of libchromadec but no more
of ORT.

---

## Core types

Declared in `<chromadec/types.h>`. Opaque handle types:

```c
typedef struct chd_video    chd_video_t;
typedef struct chd_decoder  chd_decoder_t;
typedef struct chd_frame    chd_frame_t;
typedef struct chd_nn_model chd_nn_model_t;
typedef struct chd_cancel   chd_cancel_t;
```

### Enumerations

| Enum | Values |
|---|---|
| `chd_video_standard_t` | `CHD_STD_UNKNOWN`, `CHD_STD_NTSC`, `CHD_STD_PAL`, `CHD_STD_PAL_M`, `CHD_STD_SECAM` |
| `chd_sample_encoding_t` | `CHD_ENC_UNKNOWN`, `CHD_ENC_CVBS_U10_4FSC`, `CHD_ENC_CVBS_U16_4FSC`, `CHD_ENC_CVBS_TPG21_4FSC`, `CHD_ENC_CVBS_S16_4FSC`, `CHD_ENC_RAW_S16_28M`, `CHD_ENC_RAW_S16_40M` |
| `chd_signal_state_t` | `CHD_SIG_UNKNOWN`, `CHD_SIG_STANDARD_TBC_LOCKED`, `CHD_SIG_STANDARD_TBC_UNLOCKED`, `CHD_SIG_STANDARD_RAW`, `CHD_SIG_NONSTANDARD_TBC_LOCKED`, `CHD_SIG_NONSTANDARD_TBC_UNLOCKED`, `CHD_SIG_NONSTANDARD_RAW` |
| `chd_frame_layout_t` | `CHD_FRAME_LAYOUT_UNKNOWN`, `CHD_FRAME_LAYOUT_FIELD_RASTER`, `CHD_FRAME_LAYOUT_FRAME_NATIVE` |
| `chd_plane_t` | `CHD_PLANE_Y`, `CHD_PLANE_CB`, `CHD_PLANE_CR`, `CHD_PLANE_R`, `CHD_PLANE_G`, `CHD_PLANE_B` |
| `chd_pixel_format_t` | `CHD_PIXEL_YUV444P16`, `CHD_PIXEL_YUV444PS`, `CHD_PIXEL_RGB48`, `CHD_PIXEL_RGBS`, `CHD_PIXEL_GRAY16`, `CHD_PIXEL_GRAYS`, `CHD_PIXEL_YUV440P16`, `CHD_PIXEL_YUV440PS` |
| `chd_chroma_row_component_t` | `CHD_CHROMA_ROW_DB`, `CHD_CHROMA_ROW_DR` |
| `chd_chroma_ident_mechanism_t` | `CHD_CHROMA_IDENT_PORCH`, `CHD_CHROMA_IDENT_BOTTLES`, `CHD_CHROMA_IDENT_CONTENT`, `CHD_CHROMA_IDENT_MANUAL` |
| `chd_clamp_t` | `CHD_CLAMP_NONE`, `CHD_CLAMP_LEGAL_RGB_SDR`, `CHD_CLAMP_LEGAL_RGB_HDR`, `CHD_CLAMP_LEGAL_YCBCR_BT601` |

`chd_frame_layout_t` is the container addressing of a CVBS data file:
`FIELD_RASTER` for the fixed padded `field_width x field_height` blocks every
`.tbc` uses, `FRAME_NATIVE` for the CVBS specification's exact frame-addressed
totals. See [file formats](file-formats.md#container-layouts) for what each
layout means and how it is detected.

### chd_video_params_t

Caller-supplied parameters used to override or supply metadata when opening a
source (see [`chd_video_open_composite`](#chd_video_open_composite)). The
first three identify the capture when no sidecar is found: `standard`,
`encoding`, and `signal_state` are required there. When an ld-decode sidecar
is present, a non-zero `standard` re-declares the colour standard over the
sidecar's, for captures whose sidecar cannot express it (a vhs-decode ME-SECAM
sidecar says `PAL`); the declared standard must keep the capture's line
standard, so a 525-line capture cannot be re-declared 625-line or vice
versa. `CHD_STD_SECAM` works the same way for CVBS sources: it selects the
byte-compatible 625/50 `PAL` preset for geometry (the CVBS specification has
no SECAM preset yet; see [file formats](file-formats.md#shape-of-the-format))
and re-declares the opened source SECAM, requiring a `PAL` preset when a
`.meta` sidecar is present. The remaining three merge over sidecar metadata,
because the CVBS
`.meta` schema does not carry them: `layout` (`CHD_FRAME_LAYOUT_UNKNOWN`
means auto-detect), `is_subcarrier_locked` (set to mark a subcarrier-locked,
blanking-start field raster; field rasters default to line-locked), and
`is_second_field_first` (set to declare a field-swapped capture, where the
temporally-first field of each stored pair is not the interlace first field).

```c
typedef struct chd_video_params {
    chd_video_standard_t  standard;
    chd_sample_encoding_t encoding;
    chd_signal_state_t    signal_state;
    chd_frame_layout_t    layout;
    int      is_subcarrier_locked;
    int      is_second_field_first;
} chd_video_params_t;
```

### chd_video_info_t

Read-back description of an opened video, filled by
[`chd_video_get_info`](#chd_video_get_info). Superset of the params struct:
adds the derived `fsc_hz` subcarrier frequency, the decodable `num_frames`
count, and `samples_per_frame`, the standard's native frame total (PAL
709,379; NTSC 477,750; PAL-M 477,225) regardless of container layout, while
`field_width` and `field_height` describe the served field raster. `layout`
is always the resolved value, never `CHD_FRAME_LAYOUT_UNKNOWN`. For
frame-native sources, `first_active_sample` / `last_active_sample` follow the
[horizontal alignment](file-formats.md#container-layouts) measured from the
capture's own sync at open time.

All four crop bounds are **inclusive**, and they name exactly the quantities the
[`CHD_OPT_*_ACTIVE_*` options](#option-registry) set, so a bound read from this
struct can be adjusted and set straight back with no off-by-one. Frame lines are
**0-indexed** lines of the woven interlaced frame (field 1 on the even lines,
field 2 on the odd); convert to and from the analogue standards' field-sequential
signal line numbers with `chd_video_frame_line_to_signal_line` /
`chd_video_signal_line_to_frame_line` (see [Video sources](#video-sources)).
Sample bounds are **0-indexed** row positions in the source's own horizontal
alignment, so they need not match the sample numbering of EBU Tech 3280-E or
SMPTE ST 244 (see
[active samples and the 4fsc standards](#active-samples-and-the-4fsc-standards)).

```c
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
```

#### Sample numbering

The crop sample positions — `first_active_sample` / `last_active_sample` and
their [`CHD_OPT_*_ACTIVE_SAMPLE` options](#option-registry) — are row positions
that count from **sample 0 = the first sampling instant at or after 0H**, where
0H is the half-amplitude point of the falling edge of line sync. Sync and colour
burst occupy the start of the row and the picture follows. This is the
ld-decode/vhs-decode TBC convention, and every input the library reads uses it
except one:

- **Line-locked field rasters** (ld-decode/vhs-decode `.tbc`, and CVBS field
  rasters): every row starts at its own 0H.
- **Frame-native CVBS files**: 0H-aligned per the CVBS spec. For orthogonal
  systems (NTSC, PAL-M) every row starts at 0H; for PAL the alignment is exact
  only at the frame's first line, and 0H then drifts by 4/625 of a sample per
  line across the frame (the non-orthogonal 4fsc lattice, stored on a uniform
  1135-sample row). The reader measures 0H from the sync edges at open and
  resolves such a capture to the same sync-start origin.
- **Blanking-start (subcarrier-locked) cuts**, such as ld-chroma-encoder's
  sc-locked output, are the one exception: their rows begin at the first
  digital blanking sample instead of 0H, so sample 0 sits a few samples ahead
  of 0H. A subcarrier-locked `.tbc` pair declares this
  itself, since the sidecar's burst and crop windows already sit in its row
  coordinates and are used as-is; a frame-native CVBS file with such rows is
  caught by the open-time 0H measurement. Either way `chd_video_get_info`
  reports the crop in the file's own row coordinates.

This origin **differs from the 4fsc interface standards.** SMPTE ST 244 and EBU
Tech 3280-E number a row from the start of the digital active line, with 0H
falling *late* in the row (between samples 784 and 785 for NTSC; 957 and 958 for
PAL). Translating a standard's sample number to ours is a fixed rotation — add
125 (NTSC), 177 (PAL), or 124 (PAL-M) and wrap at the row width — so a region ST
244 calls samples 0..767 is samples 125..892 here. Those constants are for
sync-start rows; on a blanking-start raster the rotation is 142 (NTSC), 187
(PAL), or 141 (PAL-M), putting the same region at samples 142..909.
[`chd_video_standard_sample_to_row_sample`](#chd_video_standard_sample_to_row_sample-chd_video_row_sample_to_standard_sample)
performs the right rotation for the opened source, so callers placing a
standards-referenced window need not track the alignment themselves. Do not
read a `first_active_sample` value as an ST 244 / EBU 3280 sample index.

This origin describes source-row positions only. The `x_start` / `x_end` of a
[`chd_dropout_span_t`](#chd_decoder_get_dropout_spans) are a different
coordinate: columns within the emitted frame, where `0` is the frame's left
edge (the crop's `first_active_sample`, plus any left padding), not 0H.

#### Active samples and the 4fsc standards

For a CVBS-native capture the default active-video crop is the **digital active
line** of the interface standard: EBU Tech 3280-E's 948 samples for 625-line
(PAL), SMPTE ST 244's 768 samples (its samples 0 to 767) for 525-line
(NTSC/PAL-M). Those standards number a row from the start of the digital active
line with 0H late in the row, while our rows start at 0H (line-locked) or at the
first digital blanking sample (blanking-start), so the digital active line is
walked round to where each alignment puts it. The synthesized default lands
exactly on it:

|System|Alignment|Default crop = digital active line|Width|
|-|-|-|-|
|PAL|sync-start|177..1124|948|
|PAL|blanking-start|187..1134|948|
|NTSC|sync-start|125..892|768|
|NTSC|blanking-start|142..909|768|
|PAL-M|sync-start|124..891|768|
|PAL-M|blanking-start|141..908|768|

This is wider than the analogue picture; it includes the blanking transition on
each side. Two common tighter crops, set via `CHD_OPT_FIRST_ACTIVE_SAMPLE` /
`CHD_OPT_LAST_ACTIVE_SAMPLE` (inclusive, in these same stored row coordinates):

- **Analogue active line** (~52.0 µs PAL, ~52.66 µs NTSC/PAL-M): line-locked PAL
  `184`/`1105` (922); line-locked NTSC `134`/`887` (754). These are nominal
  (line-blanking tolerance is a few samples).
- **ld-chroma-decoder's picture crop**: line-locked NTSC `134`/`893` (760), PAL
  `185`/`1106` (922). This is what a `.tbc` sidecar carries, so `.tbc` inputs
  already decode to it by default; these values reproduce it on a CVBS capture.

`.tbc` inputs take their crop from the sidecar and are not affected by this
default; `chd_video_get_info` reports whichever crop is in effect.

### chd_frame_info_t

```c
typedef struct chd_frame_info {
    chd_pixel_format_t format;
    int32_t  width;
    int32_t  height;
    int32_t  num_planes;
    int64_t  frame_index;
} chd_frame_info_t;
```

### chd_plane_info_t

```c
typedef struct chd_plane_info {
    int32_t width;
    int32_t height;
    int32_t first_frame_row;
} chd_plane_info_t;
```

Per-plane geometry, filled by
[`chd_frame_get_plane_info`](#chd_frame_get_plane_info). Full-height planes
report the frame dimensions and `first_frame_row` 0. The 4:4:0 chroma planes
weave the two fields by row parity and report their subsampled height and the
output frame row plane row 0 was decoded from (which is not always the
plane's topmost line); see [4:4:0 output](#440-output).

### chd_chroma_ident_report_t

```c
typedef struct chd_chroma_ident_report {
    chd_chroma_ident_mechanism_t mechanism;
    double confidence;
    double field_confidence[2];
    chd_chroma_row_component_t first_row_component;
} chd_chroma_ident_report_t;
```

Per-frame Db/Dr ident summary for line-sequential (SECAM) decodes, filled by
[`chd_frame_get_chroma_ident`](#chd_frame_get_chroma_ident): which mechanism
decided the per-line component identity, the fraction of measured lines that
agreed with the majority lattice (overall and per field, in frame order;
`1.0` for `manual`), and the component of the frame's first output row.

### chd_output_info_t

```c
typedef struct chd_output_info {
    chd_pixel_format_t format;
    int32_t  width;
    int32_t  height;
    int32_t  num_planes;
    int64_t  num_frames;
} chd_output_info_t;
```

The committed output framing, filled by
[`chd_decoder_get_output_info`](#chd_decoder_get_output_info): the `width` and
`height` of an emitted frame (the dimensions a decoded frame or a dropout mask
fills), the pixel `format` and its `num_planes`, and the source `num_frames`.

A frame is the active-picture crop surrounded by whatever black border
[`CHD_OPT_PADDING_MULTIPLE`](#option-registry) adds. The two knobs are
independent, and each does one thing: the crop chooses which signal samples and
lines become picture, and padding grows the frame around that picture without
moving or altering it. To bring in real signal from outside the default active
window, widen the crop; padding will never do it for you.

---

## Video sources

Declared in `<chromadec/video.h>`. A `chd_video_t` is an opened input plus its
metadata, read from a **metadata sidecar file** next to the data; it is the
argument you pass to [`chd_decoder_create`](#chd_decoder_create).

### chd_video_open_composite

```c
chd_status_t chd_video_open_composite(const char *path,
                                      const char *metadata_path_or_null,
                                      const chd_video_params_t *override_or_null,
                                      chd_video_t **out);
```

Open a single-file composite capture: an ld-decode `.tbc` or a CVBS
`.cvbs`. The sidecar flavour is detected automatically.

- `metadata_path_or_null`: `NULL` auto-locates the sidecar next to the data
  file: an ld-decode `<path>.db` (SQLite) / `<path>.json`, else a CVBS
  `<basename>.meta`. A decode-orc `<base>.tbcy` / `<base>.tbcc` plane looks
  for `<base>.tbc.db` / `<base>.tbc.json`. A `const char *` is an explicit
  path to a `.db`, `.json`, or `.meta` sidecar.
- `override_or_null`: `NULL` means all parameters come from the sidecar. When
  no sidecar is found, the override is mandatory and must set `standard`,
  `encoding`, and `signal_state`; the open fails otherwise. When a sidecar is
  present, `layout`, `is_subcarrier_locked`, and `is_second_field_first` are
  read from the override, and a non-zero `standard` re-declares the colour
  standard (see [chd_video_params_t](#chd_video_params_t)). See the
  [which-fields-to-set matrix](integration-guide.md#which-fields-to-set).

CVBS opens also measure signal properties the `.meta` sidecar cannot express:
the horizontal alignment of frame-native rows, and for NTSC each field's
four-field sequence position from its colour burst. See
[file formats](file-formats.md#container-layouts) for both.

### chd_video_open_yc

```c
chd_status_t chd_video_open_yc(const char *luma_path,
                               const char *chroma_path,
                               const char *metadata_path_or_null,
                               const chd_video_params_t *override_or_null,
                               chd_video_t **out);
```

Open a dual-file Y/C capture: a CVBS `.cvbsy` + `.cvbsc` pair, or a luma `.tbc` +
chroma `.tbc` pair. Sidecar resolution and flavour detection follow
[`chd_video_open_composite`](#chd_video_open_composite); the
`metadata_path_or_null` applies to the luma plane, and the chroma plane uses its
own sidecar if present, else falls back to the luma sidecar (some producers
write a single shared sidecar for the pair).

The two planes are decoded separately and merged: the luma plane is decoded
with the Mono kind for Y and the chroma plane with the configured colour kind
for Cb/Cr.

### chd_video_get_info

```c
chd_status_t chd_video_get_info(const chd_video_t *v, chd_video_info_t *out);
```

Fill `*out` with the opened video's [info](#chd_video_info_t). Note that
`num_frames` and `is_first_field_first` reflect the field order in effect; a
decoder's `reverse_field_order` option can change the decodable frame count
(see the [option registry](#option-registry)).

### chd_video_frame_line_to_signal_line / chd_video_signal_line_to_frame_line

```c
chd_status_t chd_video_frame_line_to_signal_line(const chd_video_t *v,
                                                 int32_t frame_line, int32_t *out);
chd_status_t chd_video_signal_line_to_frame_line(const chd_video_t *v,
                                                 int32_t signal_line, int32_t *out);
```

Convert between a **frame line** — the 0-indexed inclusive woven-raster line
that `first_active_frame_line` / `last_active_frame_line` and the
`CHD_OPT_*_ACTIVE_FRAME_LINE` options use — and the **field-sequential signal
line number** the analogue standards use (SMPTE ST 170 / ST 244, ITU-R
BT.470 / BT.1700, EBU Tech 3280). Field 1 (the top field) is signal lines
`1..field_height` on the even frame lines; field 2 is `field_height+1 ..
(2*field_height)-1` on the odd frame lines — a bijection over the
`(2*field_height)-1` raster lines.

The numbering is **not monotonic** down the raster: adjacent frame lines belong
to different fields and differ in signal number by about `field_height`, so a
crop's `first_active_frame_line` can convert to a *higher* signal number than
its `last` (the 525-line default region, frame lines 39..524, is signal lines
283..263).

`field_height` comes from `v`. Out-of-range input returns
`CHD_E_OUT_OF_RANGE`: frame lines run `0..(2*field_height)-2`, signal lines
`1..(2*field_height)-1`. These map the **source raster only** —
[`chd_plane_info`](#chd_plane_info_t)'s `first_frame_row` and
[`chd_dropout_span`](#dropout-correction)'s `y` are output rows in the cropped,
padded frame and need the crop origin applied before they relate.

### chd_video_standard_sample_to_row_sample / chd_video_row_sample_to_standard_sample

```c
chd_status_t chd_video_standard_sample_to_row_sample(const chd_video_t *v,
                                                     int32_t standard_sample, int32_t *out);
chd_status_t chd_video_row_sample_to_standard_sample(const chd_video_t *v,
                                                     int32_t row_sample, int32_t *out);
```

The horizontal twin of the line converters above: convert between a **row
sample**, the 0-indexed stored-row position that `first_active_sample` /
`last_active_sample` and the `CHD_OPT_*_ACTIVE_SAMPLE` options use, and the
**standard sample** numbering of the 4fsc interface standards (SMPTE ST 244
for 525-line, EBU Tech 3280-E for 625-line), where sample 0 is the first
sample of the digital active line and 0H falls late in the row. SECAM sources
convert with the 625/50 (EBU 3280) geometry.

The two numberings differ by a rotation that wraps at `field_width` and
follows the source's [horizontal alignment](#sample-numbering): sync-start
rows rotate standard sample 0 to row sample 125 (NTSC), 177 (PAL), or 124
(PAL-M); blanking-start rows to 142, 187, or 141. The alignment is read from
`v`, so the result matches the raster being decoded. Asking for ST 244's
digital active line (standard samples 0..767) yields row samples 125..892 on
a line-locked NTSC source and 142..909 on a subcarrier-locked one, ready to
feed `CHD_OPT_FIRST/LAST_ACTIVE_SAMPLE` without caring which cut the file
uses.

The rotation uses the uniform-row convention. On subcarrier-locked PAL, 0H
drifts by 4/625 of a sample per line across the frame, so a converted window
is exact at the frame's first line and sub-sample off elsewhere (the
standards' own uniform-row storage shares this property).

Both numberings run `0..field_width-1`; out-of-range input returns
`CHD_E_OUT_OF_RANGE`, and a source whose rows are not the standard 4fsc line
width (910 / 1135 / 909 samples) returns `CHD_E_UNSUPPORTED`. Like the line
converters, these map the **source raster only** —
[`chd_dropout_span`](#dropout-correction)'s `x_start` / `x_end` are output
columns.

### Extra sources for multi-source dropout

```c
chd_status_t chd_video_add_extra_source_composite(chd_video_t *v,
                                                  const char *path,
                                                  const char *metadata_path_or_null);
chd_status_t chd_video_add_extra_source_yc(chd_video_t *v,
                                           const char *luma_path,
                                           const char *chroma_path,
                                           const char *metadata_path_or_null);
```

Register additional captures of the same content as replacement candidates for
[dropout correction](#dropout-correction). Add them to the primary
`chd_video_t` before creating a decoder.

### chd_video_free

```c
void chd_video_free(chd_video_t *v);
```

Release a video handle. Decoders created from it must be freed first.

---

## Decoder

Declared in `<chromadec/decoder.h>`.

### chd_decoder_create / chd_decoder_free { #chd_decoder_create }

```c
chd_status_t chd_decoder_create(chd_video_t *v, chd_decoder_kind_t kind, chd_decoder_t **out);
void         chd_decoder_free(chd_decoder_t *d);
```

Create a decoder of `kind` bound to an opened video. `CHD_DEC_AUTO` selects a
default appropriate to the video standard.

| `chd_decoder_kind_t`                               |                                             |
|----------------------------------------------------|---------------------------------------------|
| `CHD_DEC_AUTO`                                     | Pick a sensible default for the standard.   |
| `CHD_DEC_MONO`                                     | Luma only.                                  |
| `CHD_DEC_NTSC_1D` / `_2D` / `_3D` / `_3D_NO_ADAPT` | NTSC comb decoders.                         |
| `CHD_DEC_PAL_2D`                                   | PAL 2D comb.                                |
| `CHD_DEC_TRANSFORM_2D` / `_3D`                     | Transform-domain decoders.                  |
| `CHD_DEC_NN_TRANSFORM3D`                           | Neural 3D transform (requires an NN model). |
| `CHD_DEC_LDZEUG_COLOR_CNN`                         | Neural colour CNN.                          |
| `CHD_DEC_LDZEUG_LUMA_SEP` / `_FRAME`               | Neural luma separation (field / frame).     |
| `CHD_DEC_NONE`                                     | Geometry/metadata only — no chroma decode.  |
| `CHD_DEC_SECAM`                                    | SECAM line-sequential FM chroma (4:4:0 output). |

`CHD_DEC_NONE` builds no chroma-decoding engine. Commit still resolves the
output framing, so [`chd_decoder_get_output_info`](#chd_decoder_get_output_info)
and the decode-free dropout queries
([`chd_decoder_get_dropout_spans`](#chd_decoder_get_dropout_spans),
[`chd_decode_dropout_mask`](#chd_decode_dropout_mask)) work — but
[`chd_decode_frame`](#chd_decode_frame) and
[`chd_decode_frames_async`](#chd_decode_frames_async) return
`CHD_E_DECODER_INCOMPATIBLE`. Use it to read dropout regions or output geometry
without paying for chroma decoding.

### Setting options

```c
chd_status_t chd_decoder_set_option_f64(chd_decoder_t *d, const char *name, double v);
chd_status_t chd_decoder_set_option_i32(chd_decoder_t *d, const char *name, int32_t v);
chd_status_t chd_decoder_set_option_bool(chd_decoder_t *d, const char *name, int v);
chd_status_t chd_decoder_set_option_str(chd_decoder_t *d, const char *name, const char *v);
chd_status_t chd_decoder_has_option(const chd_decoder_t *d, const char *name);
```

Strongly-typed setters keyed by the option-name macros in the
[registry](#option-registry). Setting an option that is not meaningful for the
decoder kind returns `CHD_E_INVALID_ARG`. `chd_decoder_has_option` reports
whether a name applies to this decoder.

!!! warning "Not concurrent with decode"
    Option setters and [`chd_decoder_commit`](#chd_decoder_commit) must not run
    concurrently with [`chd_decode_frame`](#chd_decode_frame) on the same
    decoder.

### chd_decoder_set_nn_model

```c
chd_status_t chd_decoder_set_nn_model(chd_decoder_t *d, chd_nn_model_t *m);
```

Attach a loaded [NN model](#neural-network-models) to a neural decoder kind.
The model must outlive the decoder; the decoder does not take ownership.

### chd_decoder_commit

```c
chd_status_t chd_decoder_commit(chd_decoder_t *d);
```

Apply pending options and prepare the decoder for use. **Required before the
first [`chd_decode_frame`](#chd_decode_frame).** Cheap to call repeatedly.
Call it again after changing options.

### chd_decoder_get_output_info

```c
chd_status_t chd_decoder_get_output_info(const chd_decoder_t *d,
                                         chd_output_info_t *out);
```

Fill `*out` with the committed [output framing](#chd_output_info_t) — the
post-crop/padding dimensions, pixel format, plane count, and frame count.
Requires a prior [`chd_decoder_commit`](#chd_decoder_commit); returns
`CHD_E_INVALID_ARG` otherwise. This is the only way to learn the output
dimensions without decoding a frame, which the decode-free
[dropout-detection](#dropout-detection) paths rely on.

### Option registry

Stable option names (string macros). The comment column gives the value type
and any decoder-kind restriction.

| Macro / name                        | Type | Notes                                                                                                                        |
|-------------------------------------|------|------------------------------------------------------------------------------------------------------------------------------|
| `CHD_OPT_CHROMA_GAIN`               | f64  | Chroma gain.                                                                                                                 |
| `CHD_OPT_CHROMA_PHASE_DEG`          | f64  | Chroma phase, degrees.                                                                                                       |
| `CHD_OPT_CHROMA_NR_LEVEL`           | f64  | Chroma noise reduction.                                                                                                      |
| `CHD_OPT_LUMA_NR_LEVEL`             | f64  | Luma noise reduction.                                                                                                        |
| `CHD_OPT_PADDING_MULTIPLE`          | i32  | Round both frame axes up to this multiple with a black border (default `1` = no padding). Never moves the crop.               |
| `CHD_OPT_REVERSE_FIELD_ORDER`       | bool | Swap field order (matches `ld-chroma-decoder -r`).                                                                           |
| `CHD_OPT_PHASE_COMPENSATION`        | bool | NTSC phase compensation; comb and ldzeug2 kinds. See [Phase compensation](#phase-compensation).                              |
| `CHD_OPT_COMB_ADAPT_THRESHOLD`      | f64  | Adaptive 3D candidate threshold; `CHD_DEC_NTSC_3D` only.                                                                     |
| `CHD_OPT_COMB_CHROMA_WEIGHT`        | f64  | Adaptive 3D chroma penalty weight; `CHD_DEC_NTSC_3D` only.                                                                   |
| `CHD_OPT_COMB_SHOW_MAP`             | bool | Overlay the adaptive 3D decision map; `CHD_DEC_NTSC_3D` only.                                                                |
| `CHD_OPT_CHROMA_IDENT_MODE`         | str  | `"auto"` (default), `"porch"`, `"bottles"`, or `"manual"`; `CHD_DEC_SECAM` only. See [SECAM line identification](#secam-line-identification). |
| `CHD_OPT_CHROMA_IDENT_MANUAL`       | str  | `"db_first"` or `"dr_first"`; required iff `chroma_ident_mode` is `"manual"`.                                                |
| `CHD_OPT_CHROMA_CLICK_NR_LEVEL`     | f64  | SECAM FM click concealment, `0.0`–`1.0` (default `1.0`; `0.0` bypasses the stage). See [SECAM click concealment](#secam-click-concealment). |
| `CHD_OPT_CHROMA_CLICK_ENV_DIP_DB`   | f64  | Expert absolute override of the adaptive envelope-dip threshold (dB); needs `chroma_click_nr_level` > 0.                     |
| `CHD_OPT_CHROMA_CLICK_FREQ_OVERSHOOT` | f64 | Expert absolute override of the adaptive deviation-overshoot threshold (max-deviation multiples); needs `chroma_click_nr_level` > 0. |
| `CHD_OPT_TRANSFORM_THRESHOLD`       | f64  | Transform-decoder threshold.                                                                                                 |
| `CHD_OPT_TRANSFORM_THRESHOLDS_FILE` | str  | Per-bin thresholds file.                                                                                                     |
| `CHD_OPT_FIRST_ACTIVE_SAMPLE`       | i32  | First active sample (inclusive, 0-indexed).                                                                                  |
| `CHD_OPT_LAST_ACTIVE_SAMPLE`        | i32  | Last active sample (inclusive, 0-indexed).                                                                                   |
| `CHD_OPT_FIRST_ACTIVE_FRAME_LINE`   | i32  | First active frame line (inclusive, 0-indexed woven frame line).                                                             |
| `CHD_OPT_LAST_ACTIVE_FRAME_LINE`    | i32  | Last active frame line (inclusive, 0-indexed woven — the line is included in the output).                                    |
| `CHD_OPT_NN_INPUT_MAGNITUDE_SCALE`  | f64  | nnTransform3D input magnitude scale.                                                                                         |
| `CHD_OPT_NN_CHROMA_BANDPASS`        | bool | ldzeug2 luma-sep chroma bandpass.                                                                                            |
| `CHD_OPT_OUTPUT_FORMAT`             | str  | `"yuv444p16"`, `"yuv444ps"`, `"rgb48"`, `"rgbs"`, `"gray16"`, `"grays"`, `"yuv440p16"`, or `"yuv440ps"` (the 4:4:0 pair is SECAM-only; see [4:4:0 output](#440-output)). |
| `CHD_OPT_OUTPUT_CLAMP`              | str  | `"none"` (default), `"legal_rgb_sdr"`, `"legal_rgb_hdr"`, or `"legal_ycbcr_bt601"`. See [Output clamping](#output-clamping). |
| `CHD_OPT_COLOR_DIFFERENCE_PRECISION` | str | `"classic"` or `"modern"` (default). Precision of the luma matrix coefficients. See [Colour conversion precision](#colour-conversion-precision).   |
| `CHD_OPT_BROADCAST_SCALING_PRECISION` | str | `"classic"`, `"modern"`, or `"scientific"` (default). Precision of the U/V reduction factors. See [Colour conversion precision](#colour-conversion-precision). |
| `CHD_OPT_OUTPUT_Y4M_HEADERS`        | bool | Emit Y4M stream headers.                                                                                                     |
| `CHD_OPT_THREAD_COUNT`              | i32  | Worker threads (`0` = auto).                                                                                                 |

### SECAM line identification { #secam-line-identification }

`CHD_OPT_CHROMA_IDENT_MODE` selects how the SECAM decoder resolves each
line's Db/Dr identity. Whatever the mode, per-line decisions feed a
strict-alternation majority fit per field, so single-line measurement errors
self-heal, and the result is reported per frame through
[`chd_frame_get_chroma_ident`](#chd_frame_get_chroma_ident).

- `"auto"` (default): back-porch reference-carrier measurement, preferred
  when enough lines measure cleanly; field-ident bottles cross-check it when
  present, take over when the porch is blanked, and content statistics are
  the last fallback.
- `"porch"`: line identification only, no fallback. The reported confidence
  still shows when this was a bad idea.
- `"bottles"`: the vertical-interval ident trapezoids only, for sources with
  blanked porches but intact vertical intervals.
- `"manual"`: no measurement; a fixed lattice anchored by
  `CHD_OPT_CHROMA_IDENT_MANUAL`, which names the component of the first
  active line of the first field of frame 0. The deterministic four-field
  alternation (Rec. ITU-R BR.469) extends it across the capture. For
  pathological captures and deterministic re-decodes.

The porch measurement doubles as per-field carrier calibration: the decoder
clusters the measured per-line reference carriers into the two undeviated
subcarriers and discriminates against the measured pair, absorbing converter
offsets (an ME-SECAM deck's free-running conversion arithmetic) without
assuming absolute carrier positions. The reference pair is measured on a
bell-free band response: the receiver's bell (cloche) network shapes noise
asymmetrically around its centre and would bias the clustered medians toward
each other. The calibration also recentres the chroma band and the bell, the
inverse of the encoder's HF pre-correction (anti-bell), on the measured pair:
a converter offset arises after encoding and translates the whole FM block,
anti-bell shaping included, so a bell left at nominal would sit on the wrong
centre (measured on an ME-SECAM capture with carriers +108 kHz off nominal:
colour-difference overshoot at large bar transitions drops from roughly
twice the step to a few percent once the bell follows the block). A field
whose porch pair is unmeasurable reuses the last measured pair, since a
converter offset is a property of the capture rather than the field; the
nominal 4.25/4.40625 MHz subcarriers apply only until some field has
measured. Opening a 625-line capture
whose measured porch signature contradicts its declared standard (a PAL
declaration over an alternating SECAM carrier pair, or the reverse) logs a
warning; the declaration always wins.

### SECAM click concealment { #secam-click-concealment }

FM clicks ("SECAM fire") on low-SNR tape are not fixable by a better
discriminator formula, so the decoder conceals them after demodulation,
enabled by default at full level. Detection flags discriminator samples
where the analytic envelope collapses below a threshold or the instantaneous
deviation exits the BT.1700 Part C Table 4 maxima; concealment interpolates
across narrow spans and substitutes the previous same-component line for
spans too wide to interpolate, before de-emphasis. `chroma_click_nr_level = 0`
bypasses the stage entirely.

Independent of the concealment stage, the demodulated deviation is always
clamped to the Table 4 maxima (D'B −350/+506 kHz, D'R −506/+350 kHz) before
de-emphasis. The transmitter clips the pre-corrected signal to those bounds,
so nothing beyond them is signal; the rail turns any click the concealment
stage left (or all of them, when bypassed) into a bounded flat-top instead
of an unbounded spike.

The thresholds come from a frozen formula composing the level with a
per-field chroma noise-floor estimate, measured deterministically from the
same back-porch windows used for ident and calibration (the median absolute
deviation of the per-line porch frequencies about their component's
carrier). With `level` in `0.0`–`1.0` and `noise` in Hz:

- envelope dip: `12 - 6*level` dB below the row's median analytic envelope;
- deviation overshoot: `max(2.6 - 1.6*level, 1.15) + 6*noise/506000` in
  multiples of the per-component maximum deviation. The floor keeps a
  transmitter limiter flat-top riding exactly at the Table 4 bounds from
  flagging on its own ripple; the deviation rail already bounds everything
  beneath the detection threshold.

Same capture and same level give bit-identical output; across captures the
effective thresholds adapt through the noise term. The endpoints were
calibrated with a swept-level study on the synthetic Table 4 generator. The
expert overrides replace either threshold with an absolute value for
batch-comparable decodes; `chd_decoder_get_chroma_click_thresholds` reports
the values actually applied to the most recent decode, so any decode is
auditable after the fact. Concealed spans are reported through
[`chd_decoder_get_dropout_spans`](#chd_decoder_get_dropout_spans) with
`CHD_DROPOUT_ORIGIN_DECODER_CONCEALMENT`, consistent with 4:4:0's
every-row-is-real honesty contract: consumers see exactly which chroma
samples are concealed rather than genuine.

### Phase compensation { #phase-compensation }

`CHD_OPT_PHASE_COMPENSATION` (NTSC; off by default) derives each line's
subcarrier phase from its own measured colour burst rather than from the
nominal field-phase table alone. Without it, a decoder assumes every line sits
at the phase RS-170 prescribes for its field and line parity, so any real phase
error in the capture lands directly in the decoded hue. It applies to the comb
kinds (`CHD_DEC_NTSC_1D` through `CHD_DEC_NTSC_3D_NO_ADAPT` and
`CHD_DEC_NN_TRANSFORM3D`) and to the three ldzeug2 kinds.

Burst-phase error is negligible on scLocked LaserDisc captures and other
sources locked to a stable reference, which is why it stays off by default;
it earns its cost on tape and off-air material where the phase drifts.

Each decoder applies the measurement where its pipeline allows:

| Kind | Where the correction lands |
|---|---|
| Comb kinds | Demodulation runs on the burst-locked axes directly (`splitIQlocked`). |
| `CHD_DEC_LDZEUG_LUMA_SEP`, `..._FRAME` | The chroma demodulation is ours, after the network's luma; the demodulated I/Q are rotated onto the measured phase before the bandpass. |
| `CHD_DEC_LDZEUG_COLOR_CNN` | The network demodulates against the nominal carrier planes; its I/Q output is rotated onto the measured phase. |

The measurement is per line from a single burst, so it corrects line-to-line
phase error but not drift within a line. A line whose burst is too weak to
measure keeps the nominal phase for that line.

### chd_decode_frame

```c
chd_status_t chd_decode_frame(chd_decoder_t *d, int64_t frame_index, chd_frame_t **out);
```

Decode a single frame by index (random access). On `CHD_OK`, `*out` is a new
[frame](#frames) the caller must free with [`chd_frame_free`](#chd_frame_free).
An out-of-range index returns `CHD_E_OUT_OF_RANGE`. Requires a prior
[`chd_decoder_commit`](#chd_decoder_commit).

### chd_decode_frames_async

```c
typedef void (*chd_frame_done_cb)(void *user, chd_status_t s, int64_t idx, chd_frame_t *f);

chd_status_t chd_decode_frames_async(chd_decoder_t *d,
                                     const int64_t *indices, size_t n,
                                     chd_frame_done_cb cb, void *user,
                                     chd_cancel_t *cancel_or_null);
```

Decode `n` frames identified by `indices`, fanning the work across the
decoder's worker pool. For each index the callback fires with the resulting
status, the index, and the frame.

!!! warning "Blocking, multi-threaded, and you own the frame"
    Despite the name, this call **blocks until every frame has been delivered**.
    It joins all workers before returning. The callback runs on **worker
    threads**, possibly concurrently, so it must be thread-safe. Each delivered
    frame is owned by your callback: call [`chd_frame_free`](#chd_frame_free)
    when done with it. On cancellation the callback receives `CHD_E_CANCELLED`
    with a `NULL` frame (nothing to free). Pass `NULL` for `cancel_or_null` to
    disable cancellation.

Requires a prior [`chd_decoder_commit`](#chd_decoder_commit).

---

## Frames

Declared in `<chromadec/frame.h>`. A `chd_frame_t` holds one decoded frame's
planes in the decoder's configured output format.

### chd_frame_get_info

```c
chd_status_t chd_frame_get_info(const chd_frame_t *f, chd_frame_info_t *out);
```

Fill `*out` with the frame's [format, dimensions, plane count, and
index](#chd_frame_info_t).

### 4:4:0 output

The `yuv440p16` / `yuv440ps` output formats carry line-sequential (SECAM)
chroma honestly: the Cb and Cr planes are full width but hold only the rows
that were really decoded, one plane row per decoded line, with no vertical
interpolation. SECAM transmits one colour-difference component per line, so
each field contributes every other line to each plane.

The chroma planes weave the two fields the same way the emitted luma plane
does: a plane row's parity matches the parity of the output frame row it was
decoded from. Plane row `2j` is the component's `j`-th line on even output
rows, plane row `2j+1` its `j`-th line on odd output rows. Consequences:

- The two chroma planes have equal heights (half the active height each) and
  together cover every active row.
- Separating fields by row parity works identically on the luma plane and
  both chroma planes.
- A chroma plane is not always in top-to-bottom picture order. The second
  field of a 625-line frame sits an odd line count after the first, so on any
  given frame one of the two planes carries each adjacent row pair spatially
  swapped, and which plane that is alternates frame to frame. Use
  [`chd_frame_chroma_row_component`](#chd_frame_chroma_row_component) to map
  frame rows to components (and therefore plane rows to frame rows), and
  [`chd_frame_get_plane_info`](#chd_frame_get_plane_info) for each plane's
  height and the frame row of its plane row `0` (its even-parity line, not
  always its topmost).
- The mapping is per-frame, not per-format: a given frame row's component
  flips frame to frame (the 625-line count is odd, giving the four-field
  ident cycle of Rec. ITU-R BR.469).

The weave interleaves both fields' half-rate chroma lattices, which tiles
only over whole two-line pairs of each field: `chd_decoder_commit` rejects
4:4:0 output unless the active frame-line crop spans a multiple of 4 lines.
The default SECAM crop (576 active lines) satisfies this.

SECAM sources decode only to these formats or to luma-only `gray16`/`grays`;
`chd_decoder_commit` rejects full-height chroma and RGB formats because any
line-repeat or resample decision belongs to the consuming application.
`output_y4m_headers` is likewise rejected for 4:4:0 output.

`padding_multiple` pads 4:4:0 frames the same way it pads every other format:
the frame and its full-height Y plane round up on both axes with a black
border. The chroma planes take the side border (neutral chroma) so all three
planes share the frame width, but they never gain rows. Every chroma plane row
stays a real decoded line, and the weave follows the output frame rows, so an
odd top border shifts which field sits on even plane rows together with the
luma weave. Each plane's `first_frame_row` shifts down with the top border,
and border rows report no component from
[`chd_frame_chroma_row_component`](#chd_frame_chroma_row_component).

### chd_frame_get_plane_info

```c
chd_status_t chd_frame_get_plane_info(const chd_frame_t *f, chd_plane_t p,
                                      chd_plane_info_t *out);
```

Fill `*out` with plane `p`'s [geometry](#chd_plane_info_t). Valid for every
pixel format, so consumers can size per-plane buffers unconditionally; the
4:4:0 formats are the reason to call it.

### chd_frame_chroma_row_component

```c
chd_status_t chd_frame_chroma_row_component(const chd_frame_t *f, int32_t frame_row,
                                            chd_chroma_row_component_t *out);
```

Report which colour-difference component the chroma decoded at output frame
row `frame_row` carries (`CHD_CHROMA_ROW_DB` or `CHD_CHROMA_ROW_DR`).
Line-sequential (4:4:0) frames only: returns `CHD_E_UNSUPPORTED` for other
frames and `CHD_E_OUT_OF_RANGE` for rows outside the output frame.

### chd_frame_get_chroma_ident

```c
chd_status_t chd_frame_get_chroma_ident(const chd_frame_t *f,
                                        chd_chroma_ident_report_t *out);
```

Fill `*out` with the frame's [Db/Dr ident summary](#chd_chroma_ident_report_t).
Line-sequential (4:4:0) frames only: returns `CHD_E_UNSUPPORTED` otherwise.
Archival consumers can use the confidence fraction to flag suspect colour
framing without touching the generic frame info.

### chd_frame_get_plane

```c
chd_status_t chd_frame_get_plane(const chd_frame_t *f, chd_plane_t p,
                                 const void **out_data,
                                 ptrdiff_t *out_stride_bytes);
```

Zero-copy borrow of a read-only pointer to a 16-bit plane `p` and its row stride
in **bytes**. The pointer is owned by the frame. Do not free it, and do not use
it after [`chd_frame_free`](#chd_frame_free). Valid for the integer pixel formats;
which planes are valid depends on the frame's [pixel format](#chd_frame_info_t)
(Y/Cb/Cr for `CHD_PIXEL_YUV444P16` and `CHD_PIXEL_YUV440P16`, R/G/B for
`CHD_PIXEL_RGB48`, or a single Y plane for `CHD_PIXEL_GRAY16`). For float
frames use [`chd_frame_get_plane_float`](#chd_frame_get_plane_float).

### chd_frame_get_plane_float

```c
chd_status_t chd_frame_get_plane_float(const chd_frame_t *f, chd_plane_t p,
                                       const float **out_data,
                                       ptrdiff_t *out_stride_bytes);
```

Zero-copy borrow of a `float` plane — same borrowing mechanism and
ownership/lifetime rules as [`chd_frame_get_plane`](#chd_frame_get_plane), over
the frame's float storage. Valid for the float pixel formats:

- `CHD_PIXEL_YUV444PS` exposes `E′Y` (plane Y, `0.0` = black, `1.0` = white)
  and `E′Cb`/`E′Cr` (planes Cb/Cr, centred at `0.0` with a `±0.5` range).
- `CHD_PIXEL_GRAYS` exposes `E′Y` only (plane Y).
- `CHD_PIXEL_RGBS` exposes `E′R`/`E′G`/`E′B` (planes R/G/B, `0.0` = black, `1.0`
  = white). Computed directly from the decoder's component signals via the
  BT.601/H.273 MatrixCoefficients=5/6 Y′CbCr → R′G′B′ matrix; no intermediate
  Y′CbCr integer quantization.
- `CHD_PIXEL_YUV440PS` exposes the same signals as `CHD_PIXEL_YUV444PS` with
  subsampled Cb/Cr planes; see [4:4:0 output](#440-output).

For `CHD_PIXEL_YUV444PS` and `CHD_PIXEL_GRAYS` these are the normalized
colour-difference signals `E′Y E′Cb E′Cr` of ITU-R BT.601 / ITU-T H.273; the
integer formats are narrow-range quantizations of the same signals, so float
output preserves full precision before quantization.

### chd_frame_free

```c
void chd_frame_free(chd_frame_t *f);
```

Release a frame and invalidate every plane pointer previously borrowed from
it. Safe on `NULL`.

## Output clamping

`CHD_OPT_OUTPUT_CLAMP` controls how out-of-range or sync-reserved sample codes
are handled.

| Token               | Meaning                                                                                                                                                                                                                                                                                                                                                                                       |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `none` (default)    | No signal-domain clamp. Integer formats still saturate at the bit-depth limits per ITU-T H.273 `Clip1` (`[0, 65535]` in 16-bit). Float formats emit raw values which may fall below `0` or above `1`. Most faithful to the decoded signal.                                                                                                                                                    |
| `legal_rgb_sdr`     | Limit output to values that map between R′G′B′ black and white. Y′CbCr formats: `Y′` stays between black (`16·256`) and white (`235·256`); `Cb`/`Cr` stays within the standard `±112·256` excursion around neutral gray (`128·256`). RGB formats: components are clamped to `[black, white]` = `[0, 1]`. Suppresses super-white, sub-black, and over-saturated chroma.                        |
| `legal_rgb_hdr`     | Map to positive-only R′G′B′ with unconstrained headroom past SDR white: components are floored at black (`0`) with no ceiling, preserving HDR highlights and removing negative excursion. Affects RGB formats (`rgbs`; `rgb48` already saturates at `0`); a **no-op** for Y′CbCr / GRAY formats, which have no clean per-component box for the positive-R′G′B′ region.                        |
| `legal_ycbcr_bt601` | Clamp to values that map to ITU-R BT.601-7 §2.5.3 video-allowed codes `[1.00d, 254.75d]` scaled to output bit-depth. Useful before quantizing for downstream SD-SDI or DV transport. In RGBS, it clamps components to bounds achievable from BT.601-legal Y′CbCr (`R′∈[-0.863, +1.884]`, `G′∈[-0.667, +1.690]`, `B′∈[-1.073, +2.093]` at the default `modern` colour-difference precision; the box is projected through the selected matrix, so it shifts slightly under `classic`). Doesn't affect integer RGB which is already stricter. |

For the GRAY formats (`gray16`, `grays`) the RGB-domain modes act on the luma
axis only, because a luma sample alone cannot carry R′G′B′ legality — `grays`
may be the `E′Y` of a split decode (for example luma from one decoder and
`E′Cb`/`E′Cr` from another), in which case the eventual R′G′B′ depends on
chroma that is not present in this frame. Accordingly, `legal_rgb_sdr` clamps
`Y′`/`E′Y` to the nominal narrow-range luma extent (`[16·256, 235·256]` /
`[0, 1]`), which doubles as a sensible luma clamp, while `legal_rgb_hdr` is a
no-op (flooring luma at black would be an R′G′B′ assumption with no luma-domain
justification). Use `legal_ycbcr_bt601` for the BT.601 §2.5.3 sync-safe luma
range, or `none` to preserve all headroom for later recombination.

The scaling relationship between "limited-range" matrix coefficient values like
Y′CbCr and their corresponding R′G′B′ values is unaffected by clamp options.
For example, `CHD_OPT_OUTPUT_CLAMP=none` with "yuv444p16" will *not* result in
"full-range" JFIF Y′CbCr.

The `legal_ycbcr_bt601` clamp matches the default behavior of tools like
decode-orc and `ld-chroma-decoder`.

---

## Colour conversion precision { #colour-conversion-precision }

Getting from a decoder's composite-domain Y/U/V to R′G′B′ or E′Y/E′Cb/E′Cr
takes two independent sets of constants. They come from different standards,
and both have been published at more than one precision, so each gets its own
option. The defaults reproduce `ld-chroma-decoder`, so you only need these if
you are chasing a specific standard's arithmetic.

Careful: the literature spells both families with a K subscripted R and B. They
are not the same constants.

### Colour-difference precision

`CHD_OPT_COLOR_DIFFERENCE_PRECISION` picks the precision of the luma matrix
coefficients, the ones that say what fraction of E′Y each primary carries.

| Token              | Luma equation                             | IEC 23091-2/ITU-T H.273 MatrixCoefficients |
|--------------------|-------------------------------------------|--------------------------------------------|
| `classic`          | `E′Y = 0.30 E′R + 0.59 E′G + 0.11 E′B`    | `4` (NTSC-1953, FCC 47 §73.682)            |
| `modern` (default) | `E′Y = 0.299 E′R + 0.587 E′G + 0.114 E′B` | `5` (625-line), `6` (525-line)             |

`classic` is the NTSC-1953 equation. `modern` from ITU-R BT.470, SMPTE ST 170,
and ITU-R BT.1700.

If you decode with `classic`, tag E′Y/E′Cb/E′Cr and Y′Cb′Cr′ results with
`MatrixCoefficients` code point `4` instead of `5`or `6`.

These coefficients reach the output in two places:

- The **E′Cb / E′Cr** scaling, through the full-excursion colour-difference
  spans `2(1 - KB)` and `2(1 - KR)`. Under `classic` those are `1.78` and
  `1.40` rather than `1.772` and `1.402`.
- The **green** row of the U/V → R′G′B′ matrix.

### Broadcast scaling precision

`CHD_OPT_BROADCAST_SCALING_PRECISION` picks the precision of reduction factors
applied to the two color differences before they modulate the subcarrier,
keeping the composite signal's excursion inside broadcast-safe transmission
range.

```
U = uReduction × (E′B - E′Y)
V = vReduction × (E′R - E′Y)
```

| Token                  | `uReduction`         | `vReduction`         | Source                                                |
|------------------------|----------------------|----------------------|-------------------------------------------------------|
| `classic`              | `0.493`              | `0.877`              | ITU-R BT.470-6 §2.5, ITU-R BT.1700 item 9             |
| `modern`               | `0.492111`           | `0.877283`           | SMPTE ST 170 Annex A.3, eq 4 and 5                    |
| `scientific` (default) | `0.4921110411224836` | `0.8772832199381787` | The closed forms ST 170's trailing ellipses stand for |

`classic` and `modern` are not roundings of the same number. SMPTE ST 170 Annex
A.3 records that the 1953 derivation of the reduction factors used a blue luma
coefficient of `0.115` instead of the correct `0.114`, and `0.493`/`0.877` are
the result. ST 170 redoes the derivation from the correct luma matrix, which is
where `0.492111`/`0.877283` come from. So `classic` is a real difference: about
`0.18%` on `uReduction`. `modern` and `scientific` agree to roughly `1e-7`, far
below a 16-bit quantum, so that pair is a numerical no-op in most cases.

Unlike color-difference precision, these factors reach *every* row of the
U/V → R′G′B′ matrix, plus both chroma scalings.

For SECAM this option is inert by construction. SECAM scaling follows
BT.470/BT.1700 standards:
`D′B = 1.505 (E′B - E′Y)`  
`D′R = -1.902 (E′R - E′Y)`

---

## Dropout correction

Declared in `<chromadec/dropout.h>`. Dropout correction replaces damaged
samples, optionally drawing on the [extra sources](#extra-sources-for-multi-source-dropout)
registered on the video.

```c
typedef struct chd_dropout_opts {
    int enabled;
    int overcorrect;        /* extend dropout boundaries by ±24 samples */
    int intra_field_only;   /* skip cross-field replacement candidates */
    void *reserved[4];      /* zero-initialised; do not repurpose */
} chd_dropout_opts_t;

typedef struct chd_dropout_stats {
    int32_t corrected;
    int32_t failed;
    int64_t total_distance;
} chd_dropout_stats_t;
```

### chd_decoder_set_dropout

```c
chd_status_t chd_decoder_set_dropout(chd_decoder_t *d, const chd_dropout_opts_t *opts);
```

Configure dropout correction on a decoder. Takes effect at the next
[`chd_decoder_commit`](#chd_decoder_commit).

### chd_decoder_get_last_dropout_stats

```c
chd_status_t chd_decoder_get_last_dropout_stats(const chd_decoder_t *d,
                                                chd_dropout_stats_t *out);
```

Return the correction counters from the **most recent**
[`chd_decode_frame`](#chd_decode_frame) on this decoder.

### chd_decoder_get_chroma_click_thresholds

```c
chd_status_t chd_decoder_get_chroma_click_thresholds(const chd_decoder_t *d,
                                                     double *env_dip_db,
                                                     double *freq_overshoot);
```

Return the effective [SECAM click-concealment](#secam-click-concealment)
thresholds applied to the **most recent** decode on this decoder: the
envelope-dip depth in dB and the deviation-overshoot multiple, after the
adaptive formula or the expert overrides. `CHD_E_UNSUPPORTED` before any
decode has run with `chroma_click_nr_level` > 0.

## Dropout detection

The functions above *conceal* dropouts during a decode; these *expose* the
flagged regions without running the chroma decoder, so a consumer can build its
own visualisation (for example a mask clip marking damaged areas). Dropouts are
detected upstream and stored in the source metadata, so these work regardless of
whether concealment is enabled — and pair naturally with a
[`CHD_DEC_NONE`](#chd_decoder_create) decoder to skip chroma decoding entirely.

```c
typedef CHD_ENUM(chd_dropout_origin) {
    CHD_DROPOUT_ORIGIN_SOURCE_METADATA     = 0,
    CHD_DROPOUT_ORIGIN_DECODER_CONCEALMENT = 8
} chd_dropout_origin_t;

typedef struct chd_dropout_span {
    int32_t y;        /* active-output row */
    int32_t x_start;  /* half-open [x_start, x_end) within the active width */
    int32_t x_end;
    chd_dropout_origin_t origin;
} chd_dropout_span_t;
```

`origin` distinguishes upstream-flagged regions (source metadata) from
samples the decoder itself detected and concealed (SECAM
[click concealment](#secam-click-concealment)). One enumeration path,
filterable by origin; the enum is numeric-gapped per family so future
origins can slot in.

### chd_dropout_detect_mode_t

```c
typedef CHD_ENUM(chd_dropout_detect_mode) {
    CHD_DROPOUT_DETECTED    = 0,
    CHD_DROPOUT_OVERCORRECT = 1
} chd_dropout_detect_mode_t;
```

Selects which regions the queries below report (mutually exclusive):

| Value | Reports |
|-------|---------|
| `CHD_DROPOUT_DETECTED` | The raw regions flagged in the source metadata. |
| `CHD_DROPOUT_OVERCORRECT` | The detected regions widened by the overcorrect margin (±24 samples, clamped to the active picture) — the footprint that overcorrect-mode concealment would touch. Independent of the configured [dropout options](#chd_decoder_set_dropout); reporting this footprint does not require `overcorrect` to be enabled. |

Both modes read source metadata only — no replacement search, no chroma
decode. Decoder-detected concealment spans are the exception: they exist only
once a frame has been decoded with `chroma_click_nr_level` > 0, after which
both modes include them for that frame.

### chd_decoder_get_dropout_spans

```c
chd_status_t chd_decoder_get_dropout_spans(chd_decoder_t *d, int64_t frame_index,
                                           chd_dropout_detect_mode_t mode,
                                           chd_dropout_span_t **out_spans,
                                           size_t *out_count);
void         chd_dropout_spans_free(chd_dropout_span_t *spans);
```

Return the dropout regions for one frame selected by `mode`, mapped into the
committed [output framing](#chd_output_info_t): each span's `y`, `x_start`, and
`x_end` index the same coordinate space as
[`chd_frame_get_plane`](#chd_frame_get_plane) for a frame from the same committed
decoder (interlace weave, crop, padding, and field order applied; spans clipped
to the active picture, sorted by `y` then `x_start`). On `CHD_OK`, `*out_spans`
is a newly-allocated array of `*out_count` spans the caller releases with
`chd_dropout_spans_free`; a frame with no dropouts yields `*out_count == 0` and
`*out_spans == NULL`. An out-of-range index returns `CHD_E_OUT_OF_RANGE`; an
unknown `mode` returns `CHD_E_INVALID_ARG`. Requires a prior
[`chd_decoder_commit`](#chd_decoder_commit).

### chd_decode_dropout_mask

```c
chd_status_t chd_decode_dropout_mask(chd_decoder_t *d, int64_t frame_index,
                                     chd_dropout_detect_mode_t mode,
                                     chd_frame_t **out);
```

Rasterise the `mode`-selected regions into a single-plane [frame](#frames)
matching the output framing: `0` for clean samples, set for dropped ones. The
mask format follows the committed output format's precision domain: a float
output format (`yuv444ps`, `rgbs`, `grays`) yields a
[`CHD_PIXEL_GRAYS`](#enumerations) mask (`1.0` dropped), any integer format
yields a [`CHD_PIXEL_GRAY16`](#enumerations) mask (`0xFFFF` dropped). The mask
clip pairs with the decode clip's sample type. Read it with
[`chd_frame_get_info`](#chd_frame_get_info) /
[`chd_frame_get_plane`](#chd_frame_get_plane) (or
[`chd_frame_get_plane_float`](#chd_frame_get_plane_float) for a `GRAYS` mask) and
free it with [`chd_frame_free`](#chd_frame_free). Does not run the chroma
decoder.

---

## Neural-network models

Declared in `<chromadec/nn.h>`. Available only when the library was built with
NN support (`chd_has_feature("nn")`). A `chd_nn_model_t` wraps a loaded ONNX
model and its execution-provider session.

### Backends

```c
typedef CHD_ENUM(chd_nn_backend) {
    CHD_NN_BACKEND_AUTO  = 0,   /* best across all backends; inferred from artifact */

    CHD_NN_ORT_AUTO      = 10,  /* ONNX Runtime, per-OS EP fallback chain */
    CHD_NN_ORT_CPU       = 11,
    CHD_NN_ORT_CUDA      = 12,
    CHD_NN_ORT_TENSORRT  = 13,
    CHD_NN_ORT_COREML    = 14,  /* ONNX Runtime CoreML execution provider */
    CHD_NN_ORT_DIRECTML  = 15,
    CHD_NN_ORT_MIGRAPHX  = 16,

    CHD_NN_COREML        = 20   /* native CoreML .mlpackage via MLModel */
} chd_nn_backend_t;
```

A backend names the runtime that runs inference. The family is encoded in the
name: the `CHD_NN_ORT_*` values select an ONNX Runtime execution provider;
values with no `ORT_` infix (e.g. `CHD_NN_COREML`) are native, non-ORT backends.

- `CHD_NN_BACKEND_AUTO` (the default) picks the best backend for the model
  *artifact*: a `.onnx` loads through ONNX Runtime with the per-OS EP auto chain
  (`CHD_NN_ORT_AUTO`); a `.mlpackage`/`.mlmodelc` loads through `CHD_NN_COREML`.
- `CHD_NN_ORT_AUTO` forces ONNX Runtime and walks its per-OS provider chain
  (Windows: TensorRT → CUDA → DirectML → CPU; Linux: TensorRT → CUDA →
  MIGraphX → CPU; macOS: CoreML → CPU), attaching the first that succeeds.
- A specific `CHD_NN_ORT_*` value pins that one EP (no fallback; load fails with
  `CHD_E_NN_BACKEND_UNAVAILABLE` if it can't attach).
- `CHD_NN_COREML` forces the native CoreML backend (requires a `.mlpackage`).

`CHD_NN_ORT_COREML` (the ORT CoreML EP) and `CHD_NN_COREML` (native) are distinct
and distinguishable after load via
[`chd_nn_model_get_active_backend`](#chd_nn_model_get_active_backend).

### CoreML compute units

```c
typedef CHD_ENUM(chd_nn_coreml_compute) {
    CHD_NN_COREML_CPU_AND_GPU = 0,  /* default: CPU + GPU, no ANE */
    CHD_NN_COREML_ALL         = 1,  /* CPU + GPU + Apple Neural Engine */
    CHD_NN_COREML_CPU_ONLY    = 2   /* CPU only */
} chd_nn_coreml_compute_t;
```

Applies to the native `CHD_NN_COREML` backend only (ignored by every ORT
backend). `CHD_NN_COREML_ALL` adds the Apple Neural Engine, which executes
fp16 only — an fp32-precision `.mlpackage` is ineligible for it wholesale, so
`ALL` changes nothing for fp32 packages. For an fp16-converted package it is
the fastest configuration for nnTransform3D `chroma_net` v2; see
[NN models](nn-models.md#native-coreml-macos).

### Session options

```c
typedef struct chd_nn_session_opts {
    chd_nn_backend_t backend;            /* AUTO unless caller pins */
    int32_t device_id;                   /* 0 unless multi-GPU */
    int     enable_graph_optim;          /* default 1 */
    int     enable_mem_pattern;          /* default 1 */
    int32_t inter_op_threads;            /* default 1 */
    int32_t intra_op_threads;            /* default 1 */
    const char *engine_cache_dir;        /* see below */
    chd_nn_coreml_compute_t coreml_compute; /* native CoreML only */
    chd_nn_compute_precision_t precision;   /* see below */
    void *reserved[4];                   /* zero-initialised; do not repurpose */
} chd_nn_session_opts_t;
```

### Compute precision

```c
typedef enum chd_nn_compute_precision {
    CHD_NN_PRECISION_FP32         = 0,  /* default */
    CHD_NN_PRECISION_FP16_ALLOWED = 1
} chd_nn_compute_precision_t;
```

`CHD_NN_PRECISION_FP16_ALLOWED` is a permission, not a mandate: a backend that
compiles the model into a device engine may build that engine with mixed
fp16/fp32 kernels (fp16 where it wins, fp32 where it doesn't). Backends
without such an engine mode ignore the field and run the model at its stored
precision — it never causes a load failure.

The TensorRT EP honours it (`trt_fp16_enable`), keeping fp16 engines in an
`fp16/` subdirectory of `engine_cache_dir` so precision modes never share a
cached engine. The CUDA EP has no reduced-precision engine mode — it executes
the model at its stored dtype — so fp16 through the CUDA EP would require an
fp16-converted `.onnx` artifact instead of this field; the field is simply
ignored there, as it is by every other EP without an engine mode.

Only enable it for weights whose input contract keeps every tensor inside
fp16 range. For nnTransform3D that means the v2 (÷128-scale) `chroma_net`
weights; a v1-series model's unscaled magnitudes overflow fp16 and the output
is unusable. See [model series and the magnitude
scale](nn-models.md#model-series-and-the-magnitude-scale).

Precision for the native CoreML backend is not selected here: it is baked
into the `.mlpackage` at conversion time
([details](nn-models.md#native-coreml-macos)).

`engine_cache_dir` controls caching of compiled EP engines (TensorRT plans,
MIGraphX binaries), which otherwise recompile on the first inference (a
15–30 s cost):

- `NULL`: auto-pick a per-user cache directory (created if absent):
  `$XDG_CACHE_HOME/chromadec` (Linux), `$HOME/Library/Caches/chromadec`
  (macOS), `%LOCALAPPDATA%/chromadec` (Windows).
- `""`: caching disabled (recompile every load; useful for CI and cold-start
  benchmarking).
- *path*: use this absolute directory (created if it doesn't exist).

Honoured by the TensorRT and MIGraphX EPs; the CUDA EP uses its own internal
PTX cache that is not configurable here.

!!! note "Why threads default to 1"
    The decoder pool already parallelises across frames; raising the intra-op
    thread count oversubscribes the CPU. That reasoning assumes the pool's
    frame-parallelism is actually in play: a consumer that pulls frames
    **serially** (a plugin host that permits one in-flight request, for
    instance) gets no parallelism from the pool, and a single-threaded CPU-EP
    inference then costs up to several times what it should — measured 3.8-4.9×
    on `chroma_net` and 2.2-2.8× on the ldzeug2 models on a 16-core machine.
    Such consumers should raise `intra_op_threads` (or pass `0` for the ORT
    default) alongside their `CHD_OPT_THREAD_COUNT` choice.

The `reserved` array exists so future minor versions can add fields without
breaking source compatibility; always leave it zeroed.
[`chd_nn_session_opts_default`](#chd_nn_session_opts_default) does this for you.

### chd_nn_session_opts_default

```c
void chd_nn_session_opts_default(chd_nn_session_opts_t *out);
```

Fill `*out` with default session options (`CHD_NN_BACKEND_AUTO`, the defaults
noted above, `CHD_NN_COREML_CPU_AND_GPU`, `CHD_NN_PRECISION_FP32`, zeroed
reserved fields). Always initialise via this function rather than by hand, so
new fields pick up correct defaults.

### chd_nn_model_load_from_file / chd_nn_model_load_from_memory / chd_nn_model_free { #chd_nn_model_load_from_file }

```c
chd_status_t chd_nn_model_load_from_file(const char *model_path,
                                         const chd_nn_session_opts_t *opts_or_null,
                                         chd_nn_model_t **out);
chd_status_t chd_nn_model_load_from_memory(const void *model_data,
                                           size_t model_size,
                                           const chd_nn_session_opts_t *opts_or_null,
                                           chd_nn_model_t **out);
void chd_nn_model_free(chd_nn_model_t *m);
```

Load a model from a file on disk (`chd_nn_model_load_from_file`) or from an
in-memory buffer (`chd_nn_model_load_from_memory`), for callers that embed the
model as a compiled-in byte array and want no filesystem dependency. The buffer
is consumed during the call and need not outlive it.

`opts.backend` selects the runtime (see [Backends](#backends)). The default
`CHD_NN_BACKEND_AUTO` infers it from the artifact: a `.onnx` loads through ONNX
Runtime; a `.mlpackage`/`.mlmodelc` loads through the native CoreML backend. A
pinned backend forces that path and requires the matching artifact (a native
backend pinned against a `.onnx` fails to load, and vice versa).

The in-memory loader works with any backend that can ingest a serialized model
buffer. ONNX Runtime can, so the `CHD_NN_ORT_*` backends and
`CHD_NN_BACKEND_AUTO` (which resolves to ONNX Runtime here) all load from
memory. Pinning `CHD_NN_COREML` returns `CHD_E_INVALID_ARG`: a `.mlpackage` is
a multi-file on-disk bundle with no in-memory load API, so use
[`chd_nn_model_load_from_file`](#chd_nn_model_load_from_file) instead. This is
a per-backend limitation.

`opts_or_null` of `NULL` uses [defaults](#chd_nn_session_opts_default). On
`CHD_E_NN_BACKEND_UNAVAILABLE` the pinned backend isn't available in this
build/host (e.g. `CHD_NN_COREML` on a non-Apple build, or one configured with
`-Dwith_coreml=disabled` — detect at runtime with `chd_has_feature("coreml")`);
`CHD_E_NN_MODEL_LOAD` indicates a bad or unreadable model. The native CoreML
`.mlpackage` is produced offline from the ONNX model with `coremltools` (see
`scripts/convert_coreml.py`) and is not shipped with the library. Remember
[`chd_shutdown`](#chd_shutdown) before exit once any model has been loaded.

### chd_nn_model_get_active_backend

```c
chd_status_t chd_nn_model_get_active_backend(const chd_nn_model_t *m,
                                             chd_nn_backend_t *out);
```

Report the backend actually selected for a loaded model. Useful after
`CHD_NN_BACKEND_AUTO` / `CHD_NN_ORT_AUTO` (which resolve to a concrete value),
and to distinguish the native `CHD_NN_COREML` backend from the ORT CoreML EP
(`CHD_NN_ORT_COREML`).

### chd_nn_backend_is_available

```c
int chd_nn_backend_is_available(chd_nn_backend_t b);   /* 1 = yes, 0 = no */
```

Query whether a backend can be used on this build and host without attempting a
model load. The AUTO sentinels report available (they always resolve to at least
CPU); `CHD_NN_COREML` tracks the `coreml` build feature.

---

## Cancellation

Declared in `<chromadec/pipeline.h>`. A `chd_cancel_t` is an optional
cooperative-cancellation token for [`chd_decode_frames_async`](#chd_decode_frames_async).

```c
chd_status_t chd_cancel_create(chd_cancel_t **out);
void         chd_cancel_request(chd_cancel_t *c);
int          chd_cancel_is_requested(const chd_cancel_t *c);
void         chd_cancel_free(chd_cancel_t *c);
```

Create a token, pass it to an async decode, and call `chd_cancel_request` from
any thread to ask the in-flight workers to stop. Remaining indices are
delivered to the callback as `CHD_E_CANCELLED` with a `NULL` frame.
`chd_cancel_is_requested` polls the flag. Free the token after the async call
returns.

!!! note "Threading model"
    The only thread-count knob is the `CHD_OPT_THREAD_COUNT` decoder option.
    Pools are **per-decoder**: `N` decoders running `T` threads each can spawn
    up to `N × T` workers in your process.
