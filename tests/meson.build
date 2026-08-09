# Test suite. Grows as the library gains functionality.
#
# Unit tests are plain C++ programs that return 0 on pass; Catch2 is wired up
# later as the test surface grows. Integration tests generate fixtures via
# encode-orc (cloned and built in CI) and compare YUV planes to golden output
# from the legacy ld-chroma-decoder binary built at commit f39e59e18.
#
# Tests come in two flavours, one dependency object each:
#
# - chromadec_abi_dep links the shared library and sees only the public C ABI,
#   which is what a real consumer gets.
# - chromadec_internal_dep links the static archive, because the shared object's
#   export list exposes the chd_* ABI alone by design, and internal C++ symbols
#   (Comb, ComponentFrame, chd::decoders::chroma_filter, the metadata readers)
#   are not in it. It also carries CHD_WITH_NN, which has to match the library
#   build for any translation unit that sees an NN-gated internal header: the
#   gated members change object layout, so a mismatch is an ODR violation.
chromadec_abi_dep = declare_dependency(
  link_with: libchromadec,
  include_directories: [public_inc, version_inc])

chromadec_internal_dep = declare_dependency(
  link_with: libchromadec.get_static_lib(),
  include_directories: [public_inc, version_inc],
  dependencies: [sqlite3_dep, threads_dep, fftw3_dep],
  compile_args: has_nn ? ['-DCHD_WITH_NN=1'] : [])

# The public headers compiled as C, which no other translation unit in the
# build does: the library itself is C++ and Objective-C++. Links the shared
# object so the address-taking table also checks the export list.
abi_smoke = executable('abi_smoke',
  'unit/abi_smoke.c',
  dependencies: chromadec_abi_dep)
test('abi_smoke', abi_smoke)

# The same TU under C23, where CHD_ENUM expands to fixed underlying types in C
# as well. The std list falls back to c11 on compilers without C23 support, so
# this duplicates abi_smoke there instead of failing setup.
abi_smoke_c23 = executable('abi_smoke_c23',
  'unit/abi_smoke.c',
  override_options: {'c_std': 'c23,c11'},
  dependencies: chromadec_abi_dep)
test('abi_smoke_c23', abi_smoke_c23)

test_tbc_reader = executable('test_tbc_reader',
  'unit/test_tbc_reader.cpp',
  dependencies: [chromadec_abi_dep, sqlite3_dep])
test('tbc_reader', test_tbc_reader)

# JSON sidecar back-compat: open a synthetic `.tbc.json` pair through the
# C ABI and confirm chd_video_open_composite falls through from missing .tbc.db
# to the .tbc.json reader (legacy ld-decode format pre-2025-SQLite-migration).
test_tbc_json_reader = executable('test_tbc_json_reader',
  'unit/test_tbc_json_reader.cpp',
  dependencies: chromadec_abi_dep)
test('tbc_json_reader', test_tbc_json_reader)

# SECAM video-system plumbing: SECAM/ME-SECAM name parsing, system defaults,
# and the open-time colour-standard re-declaration (a vhs-decode ME-SECAM
# sidecar says PAL).
test_secam_metadata = executable('test_secam_metadata',
  'unit/test_secam_metadata.cpp',
  dependencies: chromadec_internal_dep)
test('secam_metadata', test_secam_metadata)

# 4:4:0 output paths and the line-sequential frame accessors: OutputWriter
# convert440/convertToFloat440 geometry + row provenance with synthetic
# component frames, the chd_frame plane-info/row-component/ident accessors,
# and the SECAM commit-time output-format gating.
test_secam_output_abi = executable('test_secam_output_abi',
  'unit/test_secam_output_abi.cpp',
  dependencies: chromadec_internal_dep)
test('secam_output_abi', test_secam_output_abi)

# SECAM decoder core against the synthetic Table 4 generator (both
# pre-corrections, ME-SECAM carrier offset, per-line AGC, porch carriers),
# plus an end-to-end Y/C ABI decode to a 4:4:0 float frame.
test_secam_decoder = executable('test_secam_decoder',
  'unit/test_secam_decoder.cpp',
  dependencies: chromadec_internal_dep)
# The SECAM decode is the heaviest test; under ASan/TSan/UBSan instrumentation
# it runs several times longer than the 30s meson default and was timing out.
test('secam_decoder', test_secam_decoder, timeout: 300)

# End-to-end pipeline sanity test.  Reaches into internal headers (the C ABI
# for decode isn't wired up yet).
test_encode_orc_color_bars = executable('test_encode_orc_color_bars',
  'unit/test_encode_orc_color_bars.cpp',
  dependencies: chromadec_internal_dep)
test('encode_orc_color_bars', test_encode_orc_color_bars)

# NTSC burst measurement and the phase-compensation corrections built on it
# (shared by the comb and ldzeug2 decoders). Pure signal maths — no NN.
test_ntsc_burst = executable('test_ntsc_burst',
  'unit/test_ntsc_burst.cpp',
  dependencies: chromadec_internal_dep)
test('ntsc_burst', test_ntsc_burst)

# NTSC-1953 SSB wideband-I reconstruction. Synthesizes burst-locked
# composite fields with single- and double-sideband chroma tones, decodes
# them through the comb decoder's ChromaFilterMode variants, and asserts
# amplitude recovery / axis separation.
test_ntsc_chroma_ssb = executable('test_ntsc_chroma_ssb',
  'unit/test_ntsc_chroma_ssb.cpp',
  dependencies: chromadec_internal_dep)
test('ntsc_chroma_ssb', test_ntsc_chroma_ssb)

# NTSC-1953 sideband-asymmetry calibration (BetaAccumulator, β fit, and
# the chd_chroma_sideband_calibrate argument validation). Synthesizes fields with
# a known vestige profile and checks the recovered β/coherence/
# classification.
test_ntsc1953_calibration = executable('test_ntsc1953_calibration',
  'unit/test_ntsc1953_calibration.cpp',
  dependencies: chromadec_internal_dep)
# Sits close to the 30s default under sanitizer instrumentation; give it the
# same headroom as the SECAM decode so instrumented runs don't flake out.
test('ntsc1953_calibration', test_ntsc1953_calibration, timeout: 300)

# format/ DATA-table sanity tests (preset lookup, sample conversion, signal-state
# mappings).
test_format_presets = executable('test_format_presets',
  'unit/test_format_presets.cpp',
  dependencies: chromadec_internal_dep)
test('format_presets', test_format_presets)

# Diagnostic sink: default-silent contract, threshold filtering, sink
# hand-off/uninstall, concurrent emit. Links the static archive so it can drive
# chd::log::info() etc. directly, the way the library's own call sites do.
test_log_abi = executable('test_log_abi',
  'unit/test_log_abi.cpp',
  dependencies: [chromadec_internal_dep, sqlite3_dep])
test('log_abi', test_log_abi)

# Failures reach the caller through the return path: distinct status codes and
# the detecting layer's own reason for chd_last_error(), plus the range checks
# that used to log and then read/write outside their buffers anyway.
test_error_propagation = executable('test_error_propagation',
  'unit/test_error_propagation.cpp',
  dependencies: [chromadec_internal_dep, sqlite3_dep])
test('error_propagation', test_error_propagation)

# Cross-system chroma-filter table (intent×system validity + standards cutoffs)
# and the equiband_vsb EQ synthesis (frequency-response checks).
test_chroma_filter = executable('test_chroma_filter',
  'unit/test_chroma_filter.cpp',
  dependencies: chromadec_internal_dep)
test('chroma_filter', test_chroma_filter)

# The colour-difference / broadcast-scaling precision options: the resolved
# matrix for each pairing, and RGB48 vs RGBS agreement on a frame with chroma.
test_color_conversion = executable('test_color_conversion',
  'unit/test_color_conversion.cpp',
  dependencies: chromadec_internal_dep)
test('color_conversion', test_color_conversion)

# CVBS reader end-to-end smoke tests. Uses the C ABI's
# chd_video_open_composite / chd_video_open_yc entry points + reaches into the internal
# CvbsCompositeSource / CvbsYcSource for sample-level assertions.
test_cvbs_reader = executable('test_cvbs_reader',
  'unit/test_cvbs_reader.cpp',
  dependencies: chromadec_internal_dep)
test('cvbs_reader', test_cvbs_reader)

# dropout corrector. Builds a synthetic NTSC field pair, injects
# dropouts, runs DropoutCorrector::correctFrame, and asserts stats + sample
# replacement.
test_dropout_corrector = executable('test_dropout_corrector',
  'unit/test_dropout_corrector.cpp',
  dependencies: chromadec_internal_dep)
test('dropout_corrector', test_dropout_corrector)

# Multi-source VBI frame alignment. Builds in-memory metadata for a primary
# capture plus a VBI-offset CAV capture and a VBI-less capture, and asserts
# sourceFrameForPrimaryFrame registers by disc frame number (positional
# fallback when no VBI codes are present).
test_multi_source_alignment = executable('test_multi_source_alignment',
  'unit/test_multi_source_alignment.cpp',
  dependencies: chromadec_internal_dep)
test('multi_source_alignment', test_multi_source_alignment)

# integration test driving real fixtures end-to-end through
# the public C ABI. Skips with PASS when CHD_ENCODE_ORC env var is unset
# (no encode-orc binary on this machine), so plain `meson test` stays
# green; CI / a local dev with encode-orc built will exercise the
# encode-orc → TBC → chd_decode_frame chain and assert real chroma
# content out of the decoder.
test_integration = executable('test_integration',
  'unit/test_integration.cpp',
  dependencies: chromadec_abi_dep)
test('integration', test_integration,
     timeout: 120)  # encode-orc takes a couple of seconds for one frame

# end-to-end C ABI decode loop (chd_decoder_create →
# chd_decoder_set_option_* → chd_decoder_commit → chd_decode_frame →
# chd_frame_get_plane). The synthetic-TBC fixture is built in-test (same shape
# as test_encode_orc_color_bars). Also covers chd_decode_frames_async +
# chd_cancel_t.
test_decode_frame_abi = executable('test_decode_frame_abi',
  'unit/test_decode_frame_abi.cpp',
  dependencies: [chromadec_abi_dep, sqlite3_dep, threads_dep])
test('decode_frame_abi', test_decode_frame_abi)

# Decode-free dropout detection C ABI: CHD_DEC_NONE geometry-only commit,
# chd_decoder_get_output_info, chd_decoder_get_dropout_spans (field-relative
# dropouts mapped into active-output coordinates), and chd_decode_dropout_mask
# (GRAY16 mask). Injects known dropouts into a synthetic NTSC sidecar and
# checks the mapped span coordinates, the painted mask, and that a NONE
# decoder refuses to decode.
test_dropout_spans_abi = executable('test_dropout_spans_abi',
  'unit/test_dropout_spans_abi.cpp',
  dependencies: [chromadec_abi_dep, sqlite3_dep, threads_dep])
test('dropout_spans_abi', test_dropout_spans_abi)

# Absolute path to the committed test fixtures (tiny_identity.onnx). Passed
# to the NN tests so their model locator can find the synthetic fixture
# regardless of the working directory the test runs from.
nn_fixture_arg = '-DCHD_TEST_FIXTURE_DIR="' + (meson.current_source_dir() / 'fixtures') + '"'

# Put the ONNX Runtime we built against next to the NN test binaries so it is
# the copy they load. Windows searches the directory of the running executable
# ahead of every PATH entry, and also ahead of System32 where Windows ML keeps
# its own older onnxruntime.dll. Leaving our copy only on PATH lets that
# System32 build load instead; being too old for the API version these headers
# ask for, it returns a null OrtApi that faults on first use. The executable's
# own directory is the one place searched before System32, so the DLLs have to
# live there. No-op off Windows, where the runtime is resolved by rpath.
#
# The execution providers travel with it. ORT reaches its own provider DLLs
# relative to the onnxruntime.dll that loaded, and DirectML.dll is delay-loaded
# by name, so a package built against a GPU-carrying ONNX Runtime would
# otherwise test a runtime whose providers are all missing and quietly report
# nothing but CPU. Whichever of these the linked package has get copied.
ort_test_runtime = []
if host_machine.system() == 'windows' and has_ort and ort_dll_dir != ''
  fs = import('fs')
  foreach dll : ['onnxruntime.dll',
                 'onnxruntime_providers_shared.dll',
                 'onnxruntime_providers_cuda.dll',
                 'onnxruntime_providers_tensorrt.dll',
                 'DirectML.dll']
    dll_src = ort_dll_dir / dll
    if fs.exists(dll_src)
      ort_test_runtime += fs.copyfile(dll_src, dll)
    endif
  endforeach
endif

# NN framework smoke tests (Ort::Env lifecycle, provider availability,
# chd_nn_model_load_from_file / _from_memory error paths, and a real-model
# happy path against the synthetic fixture). The body self-skips when the
# library was built with with_nn=false. ORT itself is not directly linked.
test_nn_framework = executable('test_nn_framework',
  'unit/test_nn_framework.cpp',
  cpp_args: [nn_fixture_arg],
  dependencies: chromadec_abi_dep)
test('nn_framework', test_nn_framework, depends: ort_test_runtime)

# nnTransform3D wiring smoke test. Constructs a session from the located model
# (the synthetic fixture by default, or real weights via $CHD_TEST_NN_MODEL) and
# binds it — both from a file path and from an in-memory buffer. Gated on the
# ORT backend specifically: these construct chd::nn::OrtSession / OrtEngine and
# link ort_dep, so they don't apply to a native-CoreML-only build.
if has_ort
  test_nntransform3d = executable('test_nntransform3d',
    'unit/test_nntransform3d.cpp',
    cpp_args: [nn_fixture_arg],
    dependencies: [chromadec_internal_dep, ort_dep])
  test('nntransform3d', test_nntransform3d, depends: ort_test_runtime)

  # ldzeug2 NN decoders. Same shape as the nnTransform3D test: binds sessions
  # from the located model (synthetic fixture by default, or real weights via
  # the CHD_TEST_LDZEUG_* env vars), from both a path and an in-memory buffer.
  test_ldzeug = executable('test_ldzeug',
    'unit/test_ldzeug.cpp',
    cpp_args: [nn_fixture_arg],
    dependencies: [chromadec_internal_dep, ort_dep])
  test('ldzeug', test_ldzeug, depends: ort_test_runtime)
endif
