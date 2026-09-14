# Changelog

## [0.2.0] - 2026-09-13

### Added

- **HVD decoders** (`CHD_DEC_HVD_2D`, `CHD_DEC_HVD_3D`) for NTSC, PAL, and
  PAL-M: vrunk11's holographic-variational Y/C separator, from the bundled
  `hvd-core` subproject. The 3D kind adds motion-compensated cross-field terms
  over a frame of look-behind and look-ahead. Tune with
  `CHD_OPT_HVD_CG_ITERATIONS` and `CHD_OPT_HVD_TEMPORAL_STRENGTH`; build with
  `-Dwith_hvd` and `-Dwith_hvd_openmp` (both auto) and query with
  `chd_has_feature("hvd")`.
- `chd_video_open_yc` accepts `.tbcy` / `.tbcc` plane names, resolving their
  shared sidecar beside the `.tbc` base name.
- Rust: `DecoderKind::Hvd2d` / `Hvd3d` and the `HVD_CG_ITERATIONS` /
  `HVD_TEMPORAL_STRENGTH` option names, plus a test that fails when a
  `CHD_DEC_*` or `CHD_OPT_*` has no counterpart in the crate.

### Fixed

- The readers cached every whole field they had read and never evicted any, so
  memory grew by a frame of source samples for every frame decoded — around a
  megabyte per frame per source, enough to exhaust a machine partway through a
  feature-length capture. The cache is now a bounded LRU, and reader memory
  follows the decoders' field window rather than the length of the capture.
- Y/C pairs with no ld-decode sidecar (every CVBS `.cvbsy`/`.cvbsc` pair, and
  `.tbc` pairs opened from an override alone) were summed into a composite and
  separated again, restoring crosstalk the capture never had. Both planes now
  decode on their own and merge, as sidecar-carrying `.tbc` pairs already did,
  so output changes for those sources. The 0.1.0 entry below claimed this of
  both pair kinds; it held only for `.tbc`.
- Rust: the `first_active_sample` and `last_active_sample` option names were
  missing from `chromadec::options`.

### Dependencies

- hvd-core, optional, gating the HVD decoders. Fetched by the bundled wrap
  from vrunk11's `hvd-cvbs-decoding` repository at v0.1.3 and linked
  statically. `-Dwith_hvd=disabled` leaves it out.
- OpenMP, optional, threading the hvd-core solver loops. `with_hvd_openmp`
  takes whatever the toolchain provides and falls back to serial loops
  without it.

## [0.1.0] - 2026-08-16

First release. libchromadec extracts the chroma decoding from `ld-decode`'s
`tools/` (`ld-chroma-decoder`, `ld-dropout-correct`, and the shared library) and
continues it as a standalone library. The classical decoders carry over intact
(mono, NTSC 1D/2D/3D comb, PALcolour, Transform PAL 2D/3D, dropout concealment,
`.tbc` ingest) and are not itemized below. What follows is what is new on top of
that heritage.

### Added

- **A library behind a stable C ABI**, rather than a command-line tool. Frames
  are decoded by index, in any order, with planes borrowed zero-copy from the
  frame. Qt is gone: the implementation is C++17 plus SQLite3, built with Meson,
  and installs a pkg-config file and a CMake package config. Meson consumers can
  skip installing it and build it inline as a subproject instead.
- **Rust bindings** under `rust/`: `chromadec-sys` (generated FFI) and
  `chromadec` (safe wrapper with owning handles, `Result` errors, and
  lifetime-tied plane views).
- **SECAM decoding** (`CHD_DEC_SECAM`), line-sequential FM chroma with
  identification from the porch or the bottles, FM click concealment, and 4:4:0
  output formats to carry its line-halved chroma without resampling.
- **Neural-network decoders.** nnTransform3D (per-tile 3D FFT plus a CNN mask;
  model and algorithm by **asdfqazsnbb**) and ldzeug2 `color_cnn` and `luma_sep`
  (by **jsaowji**). Models are loaded by the caller from a path or from memory;
  no weights ship with the library.
- **Two inference backends.** ONNX Runtime (CPU, CUDA, TensorRT, DirectML,
  MIGraphX, and the CoreML execution provider) and, on macOS, a native CoreML
  backend that reaches the GPU for models the ORT CoreML provider pushes back to
  the CPU. Either backend builds without the other.
- **CVBS file format support** (specification v1.5.0): `.cvbs` and `.cvbsy`/`.cvbsc`
  sources, the `.meta` metadata sidecar file, the field-raster and frame-native container
  layouts, and the sample encodings for PAL, NTSC, and PAL-M. Field phase and row
  alignment are measured from the signal, since the sidecar schema records
  neither.
- **Dual-file Y/C input** (`chd_video_open_yc`) for S-Video and colour-under
  captures, as a luma plus chroma `.tbc` pair or a CVBS `.cvbsy`/`.cvbsc` pair. Each
  plane is decoded on its own and the components merged, with no composite
  reconstruction.
- **Float output.** `yuv444ps`, `yuv440ps`, `grays`, and `rgbs` carry the
  canonical normalized BT.601 / H.273 signals that the integer formats quantize;
  `rgbs` and `rgb48` are matrixed straight from the component signals with no
  intermediate integer step. `CHD_OPT_OUTPUT_CLAMP` selects the legal-range box,
  if any, to hold the output to.
- **Dropout reporting without decoding.** `chd_decoder_get_dropout_spans` and
  `chd_decode_dropout_mask` hand back the detected dropouts, as spans or as a
  mask plane, with no chroma decode run at all.
- **Multi-source dropout correction inside the decode path**, drawing
  replacement data from extra captures of the same content. Extras are registered
  per source and aligned across captures by VBI frame number, falling back to
  positional alignment where there is no VBI.
- **Real per-worker parallelism.** Commit builds one decoder instance per worker;
  async decode races workers against a shared frame index, so unequal frame costs
  balance out, and a synchronous decode can run alongside it.

### Changed

Behaviour that differs from `ld-chroma-decoder`, for anyone porting an existing
integration:

- Active line ranges are **inclusive** at both ends, for frame and field lines
  alike. The defaults now yield a full-height picture: 486 lines for 525-line
  systems, 576 for 625-line.
- Output padding is off by default (`padding_multiple` = 1, was 8).
- Output is unclamped by default. The old BT.601 video-allowed clamp is now
  opt-in via `output_clamp = "legal_ycbcr_bt601"`.
- CVBS sample levels follow the CVBS file format specification v1.5.0 (PAL black
  at blanking, NTSC and PAL-M with the 7.5 IRE setup), which shifts decoded luma
  levels for CVBS sources. `.tbc` sources read their levels from their own
  sidecar and are unaffected.

### Dependencies

- SQLite3, required. Built from a bundled wrap when no system copy is found.
- ONNX Runtime, enabled by default, backing the neural decoders. Optional:
  disable with `-Dwith_onnxruntime=false`, or on macOS build the neural decoders
  against native CoreML alone.
- FFTW3, optional, gating Transform PAL and the nnTransform3D FFT front-end.