# Integration guide

How to consume libchromadec from your application: opening a source, decoding
frames, and handling the output. This guide is for **any** consumer, whether
you are starting from scratch or replacing an existing `ld-chroma-decoder`
integration. If you are doing the latter, jump to
[Migrating from ld-chroma-decoder](#migrating-from-ld-chroma-decoder) once
you have skimmed the basics.

For the exhaustive per-function contract, see the
[C API reference](api-reference.md).

## Linking the library

libchromadec installs both a CMake package config and a pkg-config file, so a
downstream project links it idiomatically from either build system.

=== "CMake"

    ```cmake
    find_package(chromadec CONFIG REQUIRED)
    target_link_libraries(my_consumer PRIVATE chromadec::chromadec)
    ```

=== "Meson"

    ```meson
    chromadec_dep = dependency('chromadec', version: '>= 0.1')
    executable('my_consumer', 'main.c', dependencies: chromadec_dep)
    ```

=== "pkg-config"

    ```sh
    cc main.c $(pkg-config --cflags --libs chromadec)
    ```

=== "Autotools"

    In `configure.ac`:

    ```m4
    PKG_CHECK_MODULES([CHROMADEC], [chromadec >= 0.1])
    ```

    Then apply the flags to the consuming target in `Makefile.am`:

    ```makefile
    my_consumer_CFLAGS = $(CHROMADEC_CFLAGS)
    my_consumer_LDADD  = $(CHROMADEC_LIBS)
    ```

    `PKG_CHECK_MODULES` requires `pkg.m4` (from pkg-config) on your Autoconf
    macro path; most pkg-config installs provide it.

Then include the umbrella header (or the individual headers you need):

```c
#include <chromadec/chromadec.h>
```

The public surface is pure C, so the headers are consumable from both C and
C++ without a C++ runtime requirement at the boundary.

## Static linking

Most consumers link libchromadec dynamically, and the shared object resolves its
own dependencies. Load-time needs are recorded for the dynamic loader to pull
in: the C++ runtime library, FFTW, ONNX Runtime, and SQLite (if not embedded).
Either way, linking `-lchromadec` is all your link step does.

The static archive (`libchromadec.a`) is different. It doesn't convey
dependencies the way the shared object does, so your final link has to supply
them. Ask pkg-config for the **static** view and it expands the full private
link line for you from metadata the build generates. Example:

```sh
cc app.c $(pkg-config --static --cflags --libs chromadec)
```

The `--static` flag is the opt-in. Without it, pkg-config emits only
`-lchromadec`; with it, it appends `Libs.private` and recurses into
`Requires.private`, so you never list the C++ runtime or the optional deps by
hand. Meson's `dependency('chromadec', static: true)` does the same.

CMake has no equivalent switch. Our package config resolves through
`pkg_check_modules`, whose imported target (`chromadec::chromadec`) is always
built from the dynamic link flags; there is no argument that makes it static. A
CMake consumer that must link statically bypasses the imported target and uses
the static flags pkg-config exposes as variables:

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(CHROMADEC REQUIRED chromadec)
target_link_libraries(my_consumer PRIVATE ${CHROMADEC_STATIC_LDFLAGS})
```

These must be resolvable at that final link:

- **The C++ runtime library.** libchromadec is implemented in C++ behind its C
  ABI; the `.pc` conveys the standard library (`-lc++` or `-lstdc++`) detected
  at build time in `Libs.private`, so `--static` adds it for you, nothing to
  install.
- **FFTW3 and ONNX Runtime**, when those features were enabled. They're always
  external; their pkg-config files (`fftw3`, `libonnxruntime`) must be on your
  `PKG_CONFIG_PATH`, or `pkg-config --static` errors.
- **SQLite3, only if libchromadec was built against a shared/system SQLite.**
  It then stays external and you provide `sqlite3` the same way. If
  libchromadec was built with bundled SQLite (no system copy at build time),
  that SQLite is folded into `libchromadec.a` and you need nothing for
  it.

When SQLite is folded in like that, its code is redistributed as part of your
binary. The install records what went in: `share/licenses/chromadec/` carries
the licence texts plus a `depmf.json` manifest naming every subproject absorbed
into the archive and its licence. Point `-Dlicensedir=` somewhere else, or set
it empty to skip installing them.

## Relocating an install tree

`chromadec.pc` is generated relocatable: its `prefix` is derived from the
`.pc` file's own location rather than baked in as an absolute path. Moving or
vendoring a built install tree therefore keeps it working, with no rewriting of
`prefix=` by hand, as long as the internal layout is preserved.

## Shipping on Windows

libchromadec does not link ONNX Runtime statically, so a Windows application
that uses the neural decoders has to ship it, and where those DLLs sit decides
whether the build works at all.

**Put `onnxruntime.dll` in the same directory as the executable that loads
it.** Windows searches `System32` ahead of every `PATH` entry, and Windows ML
installs its own copy of `onnxruntime.dll` there (v1.17 on Windows Server 2025,
among others). A copy reachable only through `PATH` therefore loses to it, and
that older runtime is too old for the API version these headers ask for: its
`GetApi` returns a null `OrtApi` that faults on first use, which surfaces as
`STATUS_ACCESS_VIOLATION` (0xC0000005) at startup rather than as a version
error. The executable's own directory is the one location searched before
`System32`.

**Ship the execution provider alongside it.** ONNX Runtime resolves its
providers relative to the `onnxruntime.dll` that loaded, so they travel as a
set:

| Provider | Ships as | Where it comes from |
|---|---|---|
| CPU | built into `onnxruntime.dll` | any package |
| DirectML | `DirectML.dll`, delay-loaded by name | bundled in the Windows ML package's `runtimes/<rid>/native`; with the classic `Microsoft.ML.OnnxRuntime.DirectML` package it comes from `Microsoft.AI.DirectML`, restored as a dependency, not from the ONNX Runtime package itself |
| CUDA / TensorRT | `onnxruntime_providers_cuda.dll`, `onnxruntime_providers_tensorrt.dll`, `onnxruntime_providers_shared.dll` | the `onnxruntime-win-x64-gpu` release archive; the CUDA and TensorRT runtimes themselves are the user's to install |

No published Windows package carries both CUDA and DirectML, so the auto chain
a build can actually reach is bounded by the package it linked. Pick the
package for the provider you intend to ship. The two also run on different
version streams: the classic DirectML builds of ONNX Runtime end at 1.24.4,
the `Microsoft.ML.OnnxRuntime.DirectML` NuGet package and the PyPI
`onnxruntime-directml` wheel alike, and newer DirectML-capable runtimes ship
only inside the Windows ML package, while the release archives continue on
their own cadence.

!!! warning "Bundlers: what not to touch"
    Tools that vendor a binary's dependencies (delvewheel, PyInstaller and the
    like) will break an otherwise working ONNX Runtime unless told to leave
    parts of it alone.

    - **Providers are `LoadLibrary`'d by exact filename**, not linked, so a
      bundler neither finds them by walking imports nor may rename them. Add
      them by hand and exclude them from name mangling.
    - **`D3D12.dll`, `DXCore.dll` and `dxgi.dll` are OS components.** Never
      vendor them. A bundler that does will rename its copies and then patch
      `DirectML.dll`'s delay-import table in place to match, which corrupts
      `DirectML.dll` enough to fault the moment anything touches ONNX Runtime.
      The failure appears at `Ort::Env` construction, before any provider is
      selected, so it takes down a CPU-only decode too and reads as a bug
      anywhere but where it is.
    - **`ext-ms-win-gdi-internal-uap-init-l1-1-0.dll` is an API set**, a
      virtual name the loader resolves itself and never a file on disk.
      Dependency walkers cannot resolve it and have to be told to stop
      looking.

`.onnx` weights are data, not code: put them anywhere and pass the path to
[`chd_nn_model_load_from_file`](api-reference.md#chd_nn_model_load_from_file).

## Consuming as a Meson subproject

The linking examples above assume libchromadec is already installed to a prefix
your build can discover. A Meson project can instead build libchromadec **inline
as a subproject**, with no install step and no system copy: pin the source with a
wrap file and let Meson clone and build it as part of your configure.

Drop a `subprojects/chromadec.wrap` into your project:

```ini
[wrap-git]
url = https://github.com/JustinTArthur/libchromadec
revision = main
depth = 1
```

Then consume it with the **same line** you would use for an installed copy:

```meson
chromadec_dep = dependency('chromadec', version: '>= 0.1')
executable('my_consumer', 'main.c', dependencies: chromadec_dep)
```

libchromadec calls
[`meson.override_dependency`](https://mesonbuild.com/Reference-manual_builtin_meson.html#mesonoverride_dependency),
so when no system install is found `dependency('chromadec')` transparently
resolves to the subproject. Vendored and installed consumption are then identical
at the call site — flip between them by adding or removing the wrap, with no edit
to your `meson.build`.

!!! note "Transitive dependencies"
    A wrap fetches libchromadec's source, not most of its dependencies.
    **SQLite3** (required) is the exception: libchromadec ships its own
    `sqlite3.wrap`, so when no system SQLite is found it is fetched and built
    from source as a nested subproject automatically — a consumer who wraps
    libchromadec inherits that fallback and needs nothing installed for it.
    **FFTW3** and **ONNX Runtime** (optional, feature-gating Transform-PAL and
    the neural decoders) have no bundled fallback and are still resolved from
    **your** environment; install them as you would for a standalone build. To
    force the SQLite fallback even when a system copy exists, configure with
    `--force-fallback-for=sqlite3`.

Every libchromadec option, and every Meson built-in, can be set for the
subproject alone by prefixing it with the subproject name, so a vendored build
is tuned without touching your own settings:

```sh
meson setup build -Dchromadec:with_onnxruntime=false -Dchromadec:optimization=3
```

That is also how you pick which half of the library a vendored build links.
libchromadec builds both a shared object and a static archive
(`default_library=both`), and Meson hands dependents the shared one by default.
`-Ddefault_both_libraries=static` selects the archive instead, which is the
usual choice when you are producing a single self-contained binary.

## Packaging recipes

The recipes below are what a packager, or an application bundling libchromadec
through a package manager, needs beyond the linking lines above. Each one has
been shaped so the consumer's own build system sees an ordinary installed copy:
`chromadec.pc` for pkg-config and Meson, `chromadecConfig.cmake` for CMake.

### Nix

The repository is a flake. Its `overlays.default` adds a `libchromadec`
attribute to **your** nixpkgs, so the library is built against your channel
and your dependency overrides rather than a second copy of nixpkgs:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    chromadec = {
      url = "github:JustinTArthur/libchromadec";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, chromadec }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ chromadec.overlays.default ];
      };
    in {
      packages.${system}.default = pkgs.stdenv.mkDerivation {
        name = "my-consumer";
        src = ./.;
        nativeBuildInputs = [ pkgs.cmake pkgs.ninja ];
        buildInputs = [ pkgs.libchromadec ];
      };
    };
}
```

The derivation is multi-output: headers, `chromadec.pc` and the CMake config
land in `dev`, the library in `out`. nixpkgs' CMake and pkg-config hooks find
both, so `find_package(chromadec CONFIG REQUIRED)` and `dependency('chromadec')`
work with no `_DIR` hint. On Linux the shared object carries an RPATH into the
store for its own dependencies, so a consumer adds nothing for them.

The inference backends are arguments on the derivation, so the same overlay
covers every configuration:

```nix
# Classic decoders only: no ONNX Runtime in the closure.
pkgs.libchromadec.override { onnxruntime = null; withCoreml = false; }

# NN decoders against a Microsoft prebuilt ORT with the CUDA or CoreML
# execution providers instead of nixpkgs' CPU-only build.
pkgs.libchromadec.override { onnxruntime = myPrebuiltOrt; }

# CUDA sideband decoding on x86_64-linux.
pkgs.libchromadec.override { cudaSupport = true; cudaPackages = pkgs.cudaPackages; }
```

`onnxruntime` accepts either the multi-output nixpkgs package (found through
pkg-config) or a single-prefix unpacked release archive (handed to
`onnxruntime_root`); pass `onnxruntimeRoot` directly for a layout that fits
neither. The flake also exports `packages.<system>.libchromadec`,
`libchromadec-noml` and, on x86_64-linux, `libchromadec-cuda`, built against
the flake's own nixpkgs pin, for a plain `nix build` without an overlay.

### Flatpak

A libchromadec module for an application manifest. Meson, CMake, Python and
pkg-config all come from the freedesktop SDK the KDE and GNOME SDKs are based
on, so the module needs no build tooling of its own:

```yaml
  - name: chromadec
    buildsystem: meson
    config-opts:
      - -Dwith_onnxruntime=false
      - -Dwith_coreml=disabled
      - -Dwith_cuda=disabled
      - -Dwith_rocm=disabled
      - -Dwith_tests=false
      - -Dwith_examples=false
      - -Dwith_debug_overlay=false
      - -Dpkgconfig.relocatable=false
    sources:
      - type: git
        url: https://github.com/JustinTArthur/libchromadec.git
        commit: <full commit sha, or a release tag once one exists>
```

Points that matter inside flatpak-builder's sandbox:

- The build has no network. `dependency('sqlite3')` must find a system copy
  (the freedesktop runtime ships `libsqlite3`, and many manifests build their
  own module anyway); the bundled `sqlite3.wrap` fallback would try to fetch
  and fail. **FFTW3** gates the Transform PAL decoders and has no fallback:
  if the runtime you target does not carry it, add an `fftw` module (autotools,
  `--enable-shared --enable-threads`) ahead of this one, or the decoders are
  compiled out with only a line in the configure summary to say so.
- The recipe above turns every inference backend off, which is what fits a
  Flatpak: CoreML, CUDA and ROCm are not reachable from the sandbox, and there
  is no established ONNX Runtime module. To ship the neural decoders, build ORT
  as an earlier module and flip `with_onnxruntime` on.
- The library installs to `/app/lib`, which the runtime's loader already
  searches; no RPATH step is needed for the application binary.

### vcpkg

There is no port in the vcpkg registry yet; publishing one is tracked in
[issue #21](https://github.com/JustinTArthur/libchromadec/issues/21). Until
then, a vcpkg-manifest project can build libchromadec from a system install or,
if it is a Meson project, as a subproject as described above.

## A minimal decode

The shape of every integration is the same: initialise, open a source, create
and configure a decoder, commit, decode frames, and free everything in reverse
order of creation.

```c
#include <chromadec/chromadec.h>
#include <stdio.h>

int main(void) {
    if (chd_init() != CHD_OK)
        return 1;

    /* 1. Open a source. NULL sidecar → auto-locate <path>.db or <path>.json. */
    chd_video_t *video = NULL;
    if (chd_video_open_composite("capture.tbc", NULL, NULL, &video) != CHD_OK) {
        fprintf(stderr, "open: %s\n", chd_last_error());
        return 1;
    }

    chd_video_info_t info;
    chd_video_get_info(video, &info);   /* info.num_frames, standard, etc. */

    /* 2. Create a decoder bound to the source. */
    chd_decoder_t *dec = NULL;
    if (chd_decoder_create(video, CHD_DEC_AUTO, &dec) != CHD_OK) {
        fprintf(stderr, "decoder: %s\n", chd_last_error());
        chd_video_free(video);
        return 1;
    }

    /* 3. Configure options, then commit before the first decode. */
    chd_decoder_set_option_str(dec, CHD_OPT_OUTPUT_FORMAT, "yuv444p16");
    chd_decoder_set_option_f64(dec, CHD_OPT_CHROMA_GAIN, 1.0);
    if (chd_decoder_commit(dec) != CHD_OK) {
        fprintf(stderr, "commit: %s\n", chd_last_error());
        chd_decoder_free(dec);
        chd_video_free(video);
        return 1;
    }

    /* 4. Decode frames by index (random access). */
    for (int64_t i = 0; i < info.num_frames; i++) {
        chd_frame_t *frame = NULL;
        chd_status_t rc = chd_decode_frame(dec, i, &frame);
        if (rc != CHD_OK) {
            fprintf(stderr, "frame %lld: %s\n", (long long)i, chd_last_error());
            break;
        }

        chd_frame_info_t fi;
        chd_frame_get_info(frame, &fi);

        const void *y = NULL;
        ptrdiff_t stride = 0;
        chd_frame_get_plane(frame, CHD_PLANE_Y, &y, &stride);
        /* ... consume the borrowed plane pointer here ... */

        chd_frame_free(frame);   /* you own each decoded frame */
    }

    /* 5. Free in reverse order: decoder borrows the video, so it goes first. */
    chd_decoder_free(dec);
    chd_video_free(video);
    chd_shutdown();              /* optional deterministic NN teardown */
    return 0;
}
```

!!! warning "Free order matters"
    A decoder holds a borrowed pointer to its video source. Freeing the video
    while a decoder still references it is a use-after-free. Always free the
    decoder (and any decoded frames) **before** the video.

## Opening other source types

[`chd_video_open_composite`](api-reference.md#chd_video_open_composite) opens a
single-file composite (an ld-decode `.tbc` or a CVBS `.cvbs`).
[`chd_video_open_yc`](api-reference.md#chd_video_open_yc) opens a dual-file Y/C
pair (luma + chroma `.tbc`, or CVBS `.cvbsy` + `.cvbsc`). Both accept an optional
[`chd_video_params_t`](api-reference.md#chd_video_params_t) override for when
metadata is absent or you need to force parameters.

### Which fields to set

Per input variant, the override fields that matter (every variant also wants
the matching open function and metadata sidecar file from the
[reader matrix](file-formats.md#reader-matrix)):

| Input                                                       | Sidecar                              | `chd_video_params_t` fields to set                                                                      |
|-------------------------------------------------------------|--------------------------------------|---------------------------------------------------------------------------------------------------------|
| ld-decode / vhs-decode / encode-orc composite `.tbc`        | `.tbc.db` / `.tbc.json`              | none (pass `NULL`)                                                                                      |
| vhs-decode luma + chroma `.tbc` pair                        | shared `.tbc.json`, or one per plane | none (pass `NULL`)                                                                                      |
| ld-chroma-encoder `.tbc` (line-locked or `--sc-locked` PAL) | `.tbc.db` / `.tbc.json`              | none (the sidecar carries the subcarrier lock)                                                          |
| CVBS field raster with `.meta` (`.cvbs` or `.cvbsy`/`.cvbsc`) | `.meta`                            | none, or `is_subcarrier_locked = 1` for a subcarrier-locked (blanking-start) raster                     |
| CVBS frame native with `.meta` (`.cvbs` or `.cvbsy`/`.cvbsc`) | `.meta`                            | none, or `layout = CHD_FRAME_LAYOUT_FRAME_NATIVE` at the ambiguous sizes noted below                    |
| CVBS without `.meta` (any layout)                           | none                                 | `standard`, `encoding`, `signal_state` (all required), plus `layout` / `is_subcarrier_locked` as needed |

Two rules govern the table. With a sidecar present, only `layout`,
`is_subcarrier_locked`, and `is_second_field_first` are read from the
override (the `.meta` schema carries none of them); the preset triple comes
from the sidecar. With no sidecar, the override is mandatory and must set
`standard`, `encoding`, and `signal_state`, or the open fails.
`is_second_field_first` applies to any CVBS row of the table: set it when the
capture's temporally-first field of each stored pair is not the interlace
first field.

Container layout is normally auto-detected, so the `layout` override is only
needed where detection cannot decide: NTSC/PAL-M files whose size is an exact
multiple of the 526-native = 525-field-raster collision length, and truncated
files (both fall back to field raster with a warning). See
[container layouts](file-formats.md#container-layouts) for the detection
order and the byte totals.

## Configuring the decode

Options are set by name through the typed setters and take effect at the next
[`chd_decoder_commit`](api-reference.md#chd_decoder_commit). The full set lives
in the [option registry](api-reference.md#option-registry); a few common ones:

```c
chd_decoder_set_option_str (dec, CHD_OPT_OUTPUT_FORMAT, "yuv444p16");
chd_decoder_set_option_f64 (dec, CHD_OPT_CHROMA_GAIN, 1.2);
chd_decoder_set_option_bool(dec, CHD_OPT_REVERSE_FIELD_ORDER, 1);
chd_decoder_set_option_i32 (dec, CHD_OPT_THREAD_COUNT, 0);   /* 0 = auto */
chd_decoder_commit(dec);
```

Setting an option that does not apply to the decoder kind returns
`CHD_E_INVALID_ARG`; use
[`chd_decoder_has_option`](api-reference.md#setting-options) to probe.

Framing is two independent knobs, and each does one thing.

`CHD_OPT_FIRST_ACTIVE_SAMPLE` / `CHD_OPT_LAST_ACTIVE_SAMPLE` and the two
`*_ACTIVE_FRAME_LINE` options are the crop: they choose which signal samples and
lines become picture. All four are inclusive and **0-indexed** — samples are
stored-row positions, frame lines are lines of the woven interlaced frame — and
are named after the fields
[`chd_video_get_info`](api-reference.md#chd_video_info_t) reports, so you can
read a bound, adjust it, and set it straight back. Widen them to admit overscan
the default active window excludes. To work in the analogue standards' line
numbers instead, convert with `chd_video_frame_line_to_signal_line` /
`chd_video_signal_line_to_frame_line`.

`CHD_OPT_PADDING_MULTIPLE` rounds both frame axes up to a multiple your codec
wants, by adding a black border around the picture. It never moves or alters the
crop, so switching it on cannot change a single picture sample; it only decides
how much black surrounds them.

## Pixel formats and plane access

The output format is chosen with `CHD_OPT_OUTPUT_FORMAT`
(`"yuv444p16"`, `"yuv444ps"`, `"rgb48"`, `"rgbs"`, `"gray16"`, `"grays"`). For
each decoded frame:

- [`chd_frame_get_plane`](api-reference.md#chd_frame_get_plane) borrows a
  read-only pointer plus a **byte** stride into 16-bit plane data for the
  integer formats. The pointer belongs to the frame. Never free it, and never
  use it after the frame is freed.
- [`chd_frame_get_plane_float`](api-reference.md#chd_frame_get_plane_float)
  is the equivalent zero-copy borrow for the float formats. `"yuv444ps"` and
  `"grays"` expose the normalized `E′Y E′Cb E′Cr` signals of ITU-R
  BT.601 / ITU-T H.273 that the integer formats quantize. `"rgbs"` exposes
  normalized `E′R E′G E′B` planes computed direct from the decoder's
  component signals via the BT.601/H.273 Y′CbCr → R′G′B′ matrix, with no
  intermediate integer quantization.

Which planes are valid depends on the format: Y/Cb/Cr, R/G/B, or a single Y
plane for `"gray16"`.

## Neural decoders

The neural decoder kinds (`CHD_DEC_NN_TRANSFORM3D`, `CHD_DEC_LDZEUG_*`) need a
model attached before commit:

```c
chd_nn_session_opts_t opts;
chd_nn_session_opts_default(&opts);     /* CHD_NN_BACKEND_AUTO, sane defaults */

chd_nn_model_t *model = NULL;
if (chd_nn_model_load_from_file("chroma_net_v2.onnx", &opts, &model) != CHD_OK) {
    fprintf(stderr, "model: %s\n", chd_last_error());
    /* ... */
}
/* AUTO infers the backend from the artifact: a .onnx loads through ONNX
 * Runtime, a .mlpackage through the native CoreML backend. To force one,
 * set opts.backend (e.g. CHD_NN_COREML for native CoreML, CHD_NN_ORT_CPU for
 * CPU-only ONNX Runtime). Or load an embedded ONNX model (no file needed):
 *   chd_nn_model_load_from_memory(model_bytes, model_len, &opts, &model); */

chd_decoder_t *dec = NULL;
chd_decoder_create(video, CHD_DEC_NN_TRANSFORM3D, &dec);
chd_decoder_set_nn_model(dec, model);   /* model must outlive the decoder */
chd_decoder_commit(dec);
```

The model is **borrowed**, not owned. Keep it alive for the decoder's
lifetime, free the decoder first, then `chd_nn_model_free(model)`.

!!! note "Shutdown after NN use"
    [`chd_shutdown`](api-reference.md#chd_shutdown) tears down the ONNX
    Runtime environment deterministically; call it at most once, after every
    decoder and model is freed. It is optional: without it the environment is
    deliberately left alive through process exit, which is safe — a host that
    cannot control destructor ordering (a plugin, an interpreter) can simply
    never call it.

Backend availability can be probed up front with
[`chd_nn_backend_is_available`](api-reference.md#chd_nn_backend_is_available),
and the backend actually chosen by `CHD_NN_BACKEND_AUTO` read back with
[`chd_nn_model_get_active_backend`](api-reference.md#chd_nn_model_get_active_backend).
See [NN model conventions](nn-models.md) for model paths, magnitude scales, and
per-platform backend notes.

## Dropout correction

Configure correction on the decoder, optionally registering extra captures of
the same content as replacement sources **before** creating the decoder:

```c
chd_video_add_extra_source_composite(video, "second-pass.tbc", NULL);

chd_dropout_opts_t dz = { .enabled = 1, .overcorrect = 0, .intra_field_only = 0 };
chd_decoder_set_dropout(dec, &dz);
chd_decoder_commit(dec);
/* after each decode: */
chd_dropout_stats_t stats;
chd_decoder_get_last_dropout_stats(dec, &stats);
```

## Dropout detection

Concealment *hides* dropouts; sometimes you want to *see* where they are — for
example to produce a mask clip highlighting damaged regions alongside the
decoded video. The detected regions are source metadata, available without
running the chroma decoder, so extract them cheaply with a `CHD_DEC_NONE`
decoder. Configure it with the same geometry options (padding, output format,
line overrides, field order) as your video decoder so the two line up
pixel-for-pixel:

```c
chd_decoder_t *mask_dec = NULL;
chd_decoder_create(video, CHD_DEC_NONE, &mask_dec);
chd_decoder_set_option_i32(mask_dec, CHD_OPT_PADDING_MULTIPLE, 1);
chd_decoder_commit(mask_dec);

chd_output_info_t oi;
chd_decoder_get_output_info(mask_dec, &oi);   /* canvas size, no decode needed */
```

The simplest path is `chd_decode_dropout_mask`, which returns a single-plane
frame (`0` clean, set when dropped) sized to the output framing — ready to use as
a mask plane. Its format follows the committed output format's precision domain,
so the mask matches the sample type of the decode clip it accompanies: an integer
output format gives a `GRAY16` mask (`0xFFFF` dropped, read with
`chd_frame_get_plane`); a float output format (`grays`, `yuv444ps`, `rgbs`) gives
a `GRAYS` mask (`1.0` dropped, read with `chd_frame_get_plane_float`). The
example above commits the default `yuv444p16`, so the mask is `GRAY16`:

```c
chd_frame_t *mask = NULL;
if (chd_decode_dropout_mask(mask_dec, frame_index, CHD_DROPOUT_DETECTED, &mask) == CHD_OK) {
    const void *data; ptrdiff_t stride;
    chd_frame_get_plane(mask, CHD_PLANE_Y, &data, &stride);
    /* ... composite the mask against the decoded luma plane ... */
    chd_frame_free(mask);
}
```

The `mode` argument selects which regions to report. `CHD_DROPOUT_DETECTED` gives
the raw flagged regions; `CHD_DROPOUT_OVERCORRECT` widens them by the overcorrect
margin to show the footprint overcorrect-mode concealment would overwrite. Both
read metadata only — neither runs the corrector.

For finer control — feathered or coloured overlays, say — ask for the raw spans
and rasterise them yourself:

```c
chd_dropout_span_t *spans = NULL;
size_t count = 0;
chd_decoder_get_dropout_spans(mask_dec, frame_index, CHD_DROPOUT_DETECTED, &spans, &count);
for (size_t i = 0; i < count; i++) {
    /* mark columns [spans[i].x_start, spans[i].x_end) on row spans[i].y */
}
chd_dropout_spans_free(spans);
```

Both report the raw detected dropouts in the active-output coordinate space, so
they overlay frames from a video decoder committed with the same geometry. A
`CHD_DEC_NONE` decoder rejects [`chd_decode_frame`](api-reference.md#chd_decode_frame)
with `CHD_E_DECODER_INCOMPATIBLE` — it exists only to serve geometry and dropout
queries.

## Decoding many frames in parallel

[`chd_decode_frames_async`](api-reference.md#chd_decode_frames_async) fans a
list of indices across the decoder's worker pool and delivers each result to a
callback.

```c
static void on_frame(void *user, chd_status_t s, int64_t idx, chd_frame_t *f) {
    if (s == CHD_OK) {
        /* consume f from a worker thread; must be thread-safe */
        chd_frame_free(f);          /* the callback owns the frame */
    }
}

int64_t indices[] = {0, 1, 2, 3};
chd_decode_frames_async(dec, indices, 4, on_frame, NULL, NULL);
```

!!! warning "It blocks, runs on worker threads, and hands you ownership"
    Despite the name the call **blocks until every frame is delivered**. The
    callback runs on worker threads, possibly concurrently, so it must be
    thread-safe, and each delivered frame is yours to
    [`chd_frame_free`](api-reference.md#chd_frame_free). Pass a
    [`chd_cancel_t`](api-reference.md#cancellation) to stop early; cancelled
    indices arrive as `CHD_E_CANCELLED` with a `NULL` frame.

## Error handling pattern

Every fallible call returns [`chd_status_t`](api-reference.md#status-codes).
On failure, [`chd_last_error()`](api-reference.md#chd_last_error) gives a
thread-local detail string for that thread's most recent failing call:

```c
chd_status_t rc = chd_video_open_composite(path, NULL, NULL, &video);
if (rc != CHD_OK) {
    fprintf(stderr, "%s: %s\n", chd_status_str(rc), chd_last_error());
}
```

`chd_status_str` is a stable static code name; `chd_last_error` is the
context-rich, per-thread message. Check the status first and read the detail
only when it is not `CHD_OK`: the library records a detail wherever it detects
a failure, including ones it goes on to handle itself, so the string can be
non-empty after a call that succeeded.

## Diagnostics

Failures come back on the return path above; the commentary a decode produces
along the way has no return path, so it goes to a sink you install. The library
writes nothing to stderr or stdout on its own, which keeps it out of your
program's output unless you ask:

```c
static void log_sink(chd_log_level_t level, chd_log_flags_t flags,
                     const char *message, void *user_data)
{
    /* You are already printing this one from chd_last_error() above. */
    if (flags & CHD_LOG_F_RETURNED) return;

    fprintf((FILE *)user_data, "[chromadec/%s] %s\n",
            chd_log_level_str(level), message);
}

chd_set_log_callback(log_sink, stderr);
chd_set_log_level(CHD_LOG_DEBUG);   /* default is CHD_LOG_INFO */
```

If you report failures from the return path, as the pattern above does, drop
`CHD_LOG_F_RETURNED` messages: they are the same reason arriving a second time.
Keep them if the sink is your only output. Errors *without* the flag have no
return path of their own, so never filter on the level alone. See
[Message flags](api-reference.md#chd_log_f_returned).

For a command-line tool that just wants the messages on the terminal,
`chd_log_to_stderr()` installs a built-in sink that does exactly that. See
[Diagnostics](api-reference.md#diagnostics) for the sink contract, in
particular that it is called from decode worker threads.

Both calls may be made before `chd_init`, which is where you want them if you
would rather not miss anything early. If you are writing a plugin rather than
an application, note that the sink is process-wide: the last consumer of
libchromadec in the process to install one wins, so installing unconditionally
at load time can take the messages away from whoever asked for them first.

---

## Migrating from ld-chroma-decoder

If your project currently builds against `ld-chroma-decoder` (as a git
submodule, a vendored copy, or in-tree decoder classes), this section maps the
old surface onto libchromadec. The structural goal is to **drop the vendored
code and dynamically link `libchromadec` instead**.

### Structural changes

1. Remove the submodule / vendored `ld-chroma-decoder` (and any local patches).
2. Add `find_package(chromadec CONFIG REQUIRED)` and link
   `chromadec::chromadec`.
3. Replace direct use of the C++ decoder classes and the CLI-driven process
   model with the C ABI flow shown above: open → create → set options →
   commit → decode → free.

### CLI flags → ABI

The old command-line flags map onto the ABI in four ways: as the **decoder
kind** at create time, as **option-registry** entries, as **open arguments**,
or as part of **your decode loop**.

| `ld-chroma-decoder` flag | libchromadec equivalent |
|---|---|
| `-f, --decoder <name>` | `chd_decoder_kind_t` passed to [`chd_decoder_create`](api-reference.md#chd_decoder_create) (`ntsc2d` → `CHD_DEC_NTSC_2D`, `transform3d` → `CHD_DEC_TRANSFORM_3D`, …) |
| `-b, --blackandwhite` | decoder kind `CHD_DEC_MONO` |
| `--simple-pal` | decoder kind `CHD_DEC_PAL_2D` (vs the transform PAL kinds) |
| `-r, --reverse` | `CHD_OPT_REVERSE_FIELD_ORDER` (bool) |
| `--chroma-gain` | `CHD_OPT_CHROMA_GAIN` (f64) |
| `--chroma-phase` | `CHD_OPT_CHROMA_PHASE_DEG` (f64) |
| `--chroma-nr` | `CHD_OPT_CHROMA_NR_LEVEL` (f64) |
| `--luma-nr` | `CHD_OPT_LUMA_NR_LEVEL` (f64) |
| `--ntsc-phase-comp` | `CHD_OPT_PHASE_COMPENSATION` (bool) |
| `--adapt-threshold` | `CHD_OPT_COMB_ADAPT_THRESHOLD` (f64) |
| `--chroma-weight` | `CHD_OPT_COMB_CHROMA_WEIGHT` (f64) |
| `--transform-threshold` | `CHD_OPT_TRANSFORM_THRESHOLD` (f64) |
| `--transform-thresholds <file>` | `CHD_OPT_TRANSFORM_THRESHOLDS_FILE` (str) |
| `-o, --oftest` | `CHD_OPT_COMB_SHOW_MAP` (bool) |
| `--pad, --output-padding` | `CHD_OPT_PADDING_MULTIPLE` (i32; **defaults to `1` / no padding here**) |
| `-p, --output-format` | `CHD_OPT_OUTPUT_FORMAT` (str) + `CHD_OPT_OUTPUT_Y4M_HEADERS` (bool) |
| `-t, --threads` | `CHD_OPT_THREAD_COUNT` (i32, `0` = auto) |
| `--ffrl / --lfrl` (frame lines) | `CHD_OPT_FIRST_ACTIVE_FRAME_LINE` / `CHD_OPT_LAST_ACTIVE_FRAME_LINE` (i32; **inclusive** — last line is included) |
| `--input-metadata <file>` | the `metadata_path` argument to [`chd_video_open_composite`](api-reference.md#chd_video_open_composite) |
| `-s, --start` / `-l, --length` | the frame **indices** you pass to [`chd_decode_frame`](api-reference.md#chd_decode_frame). There is no global start/length; you drive the range |
| `--show-ffts` and other debug flags | not part of the stable ABI |

### Things that change in spirit, not just syntax

- **No process-per-decode.** Where you previously shelled out to the
  `ld-chroma-decoder` binary and piped its output, you now hold a decoder
  handle and pull frames directly: no subprocess, no stdout parsing.
- **Random access instead of a stream.** `chd_decode_frame` takes an index, so
  seeking is free; the old `--start`/`--length` windowing becomes a loop bound.
- **Output is in-memory planes**, not a Y4M byte stream; request Y4M framing
  with `CHD_OPT_OUTPUT_Y4M_HEADERS` only if you specifically need it.
- **Neural decoders are first-class** here (`CHD_DEC_NN_TRANSFORM3D`,
  `CHD_DEC_LDZEUG_*`) rather than out-of-tree patches.
- **Active frame lines are inclusive and 0-indexed woven-raster lines.**
  `CHD_OPT_FIRST_ACTIVE_FRAME_LINE` / `CHD_OPT_LAST_ACTIVE_FRAME_LINE` name the
  first and last lines that are *part of* the active region, 0-indexed into the
  woven interlaced frame (field 1 on the even lines, field 2 on the odd); the
  last line is included, not one-past-the-end. If you previously passed
  ld-chroma-decoder's exclusive `--lfrl` value through, subtract one. To set or
  read the crop in the analogue standards' field-sequential line numbers, convert
  with `chd_video_frame_line_to_signal_line` / `chd_video_signal_line_to_frame_line`.
  The active field-line range is derived from the frame-line crop, so there is no
  separate field-line option.
  **TBC sidecar metadata is handled for you:** tbc-tools writes
  `last_active_frame_line` as an exclusive bound, and the SQLite reader translates
  it to the inclusive scheme on ingest, so a v4 sidecar decodes to exactly the
  picture its author intended. With the default active region you get a full
  **486-line picture for 525-line (NTSC)** systems and **576 lines for 625-line
  (PAL)** systems.
- **No output padding by default.** `CHD_OPT_PADDING_MULTIPLE` defaults to `1`
  (emit the active region as-is). decode-orc / tbc-tools pad the frame to a
  multiple of 8 unless told otherwise — set `CHD_OPT_PADDING_MULTIPLE`
  explicitly if you need codec-friendly dimensions. Padding now adds a black
  border rather than widening the crop, and applies uniformly to every output
  format, including `yuv444ps`, `rgbs`, and the 4:4:0 SECAM formats (whose
  chroma planes take only the side border and never gain rows).
- **CVBS captures default to the full digital active line.** For `.tbc` inputs
  the active-video crop comes from the sidecar, exactly as ld-chroma-decoder
  used it, so the default width is unchanged (NTSC 760, PAL 922). For
  CVBS-native captures libchromadec synthesizes the crop, and it now defaults to
  the interface standard's digital active line (SMPTE ST 244: 768 samples for
  525-line; EBU Tech 3280-E: 948 for 625-line) rather than the tighter analogue
  picture crop. That is wider than ld-chroma-decoder's picture and deliberately
  includes the blanking transition on each side. To reproduce the narrower
  picture crop on a CVBS capture, set `CHD_OPT_FIRST_ACTIVE_SAMPLE` /
  `CHD_OPT_LAST_ACTIVE_SAMPLE` — for a line-locked (0H) raster, NTSC `134`/`893`
  and PAL `185`/`1106`.
- **The crop options override a sidecar-provided span, per bound.** When a `.tbc`
  sidecar spells out `active_video_start` / `active_video_end`, that is the
  starting point, but any of the six `CHD_OPT_*_ACTIVE_SAMPLE` /
  `CHD_OPT_*_ACTIVE_*_LINE` options you set replaces the corresponding bound;
  bounds you leave unset keep the sidecar value. This is the same behaviour for a
  synthesized CVBS default — by the time your options apply, sidecar-provided and
  synthesized crops are indistinguishable. Setting only one end (say
  `CHD_OPT_LAST_ACTIVE_SAMPLE`) is fine; the other keeps the sidecar's value.
  Every resulting bound is validated against the field, so an out-of-range
  override is reported at `chd_decoder_commit`, not silently dropped. To place
  a window given in SMPTE ST 244 / EBU Tech 3280-E sample numbers, convert
  each bound with `chd_video_standard_sample_to_row_sample` first; it applies
  the right rotation for the source's horizontal alignment.