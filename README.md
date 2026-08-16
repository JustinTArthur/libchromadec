# libchromadec

A library for decoding 4×fsc-sampled composite analog video (CVBS / TBC) into
component Y′ C′b C′r, E′Y E′Cb E′Cr or R′G′B′ video behind a stable C ABI. It
brings the classical decoders (1D/2D/3D comb, PALcolour, Transform-PAL 2D/3D,
SECAM) and the neural-network decoders (nnTransform3D, ldzeug2 color_cnn and
luma_sep) together in one place.

## Status

Near first 0.1.0 release. The C ABI is implemented and the library decodes.
The ABI is **not frozen until that tag**, so treat the current `main` as
movable. From 0.1.0 on, the rules in
[docs/abi-stability.md](docs/abi-stability.md) apply: pre-1.0 the ABI breaks at
the minor, and the soname moves with it.

## What it decodes

**Sources.** A single-file composite (ld-decode / vhs-decode / encode-orc
`.tbc`, or a CVBS `.cvbs`), or a dual-file Y/C pair (luma + chroma `.tbc`,
or CVBS `.cvbsy` + `.cvbsc`). Metadata comes from a `.tbc.json`, `.db`, or `.meta`
sidecar, or from an explicit parameter override where there is none.
[docs/file-formats.md](docs/file-formats.md) has the reader matrix.

**Decoders.** `chd_decoder_kind_t` selects one:

| Kind                                                         | Notes                                             |
|--------------------------------------------------------------|---------------------------------------------------|
| `CHD_DEC_MONO`                                               | luma only                                         |
| `CHD_DEC_NTSC_1D` / `_2D` / `_3D` / `_3D_NO_ADAPT`           | comb filters                                      |
| `CHD_DEC_PAL_2D`                                             | PALcolour                                         |
| `CHD_DEC_TRANSFORM_2D` / `_3D`                               | Transform PAL; needs FFTW                         |
| `CHD_DEC_SECAM`                                              | line-sequential FM chroma; 4:4:0 output           |
| `CHD_DEC_NN_TRANSFORM3D`                                     | needs an NN model                                 |
| `CHD_DEC_LDZEUG_COLOR_CNN` / `_LUMA_SEP` / `_LUMA_SEP_FRAME` | needs an NN model                                 |
| `CHD_DEC_NONE`                                               | geometry and dropout queries only, no chroma work |

**Output.** `yuv444p16`, `yuv444ps`, `yuv440p16`, `yuv440ps`, `rgb48`, `rgbs`,
`gray16`, `grays`. The float formats carry the canonical normalized BT.601 /
H.273 signals that the integer formats quantize. Planes are borrowed zero-copy
from the decoded frame.

Beyond decoding, the library conceals dropouts (optionally drawing replacement
data from extra captures of the same content) and can also *report* them,
handing back a mask plane or the raw spans without running a chroma decode at
all.

## Building

Needs Meson 1.8 or newer and a C++17 compiler.

```sh
python -m venv .venv && source .venv/bin/activate
pip install meson ninja
meson setup build
meson compile -C build
meson test -C build
```

**Dependencies.** SQLite3 is required, and is fetched and built as a subproject
when no system copy is found. ONNX Runtime is enabled by default and backs the
neural decoders; if it is not discoverable, either point the build at it with
`-Donnxruntime_root=/path/to/onnxruntime` (an unpacked release archive, or a
restored NuGet package directory: `Microsoft.ML.OnnxRuntime.DirectML` or the
Windows ML package, the two forms DirectML ships in) or turn it off with
`-Dwith_onnxruntime=false`. FFTW3 is optional and gates Transform-PAL. CUDA,
ROCm, and native CoreML are auto-detected and gate GPU inference. Run `meson
configure build` to see every option.

## Using it

libchromadec installs its public headers, a relocatable pkg-config file, and a
CMake package config, so it links idiomatically from either build system:

```meson
chromadec_dep = dependency('chromadec', version: '>= 0.1')
```

```cmake
find_package(chromadec CONFIG REQUIRED)
target_link_libraries(my_consumer PRIVATE chromadec::chromadec)
```

A Meson project can skip installing it altogether and build libchromadec inline
as a subproject via a wrap file; the `dependency('chromadec')` call above is the
same either way. Then include the umbrella header:

```c
#include <chromadec/chromadec.h>
```

Every integration has the same shape: initialise, open a source, create a
decoder, set options, commit, decode frames by index, free in reverse. The
[integration guide](docs/integration-guide.md) walks through that, covers static
linking and vendoring, and maps `ld-chroma-decoder`'s CLI flags onto the ABI for
anyone replacing an existing integration.
[docs/api-reference.md](docs/api-reference.md) is the exhaustive per-function
contract.

Rust bindings live under `rust/`; see
[docs/rust-bindings.md](docs/rust-bindings.md).

## Models

**No model weights ship with this library.** They are separately distributed
and sometimes large. The neural decoder kinds take a model you load yourself,
from a path or straight from memory, and the library never goes looking for
one. [models/README.md](models/README.md) covers the layout convention and the
per-model magnitude scales; [docs/nn-models.md](docs/nn-models.md) covers the
backends and execution providers.

## Documentation

Start at [docs/index.md](docs/index.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE). Inherited from `ld-decode`.

## Attribution

This library extracts and consolidates work originally authored upstream in
`ld-decode`, alongside neural decoders derived from their authors' original
harnesses. The git history preserves the authorship of every extracted commit.
See [docs/attribution.md](docs/attribution.md) for the contributors whose work
is preserved here.