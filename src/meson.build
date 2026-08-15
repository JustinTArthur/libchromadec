chromadec_sources = files(
  'abi/c_api_lifecycle.cpp',
  'abi/c_api_video.cpp',
  'abi/c_api_decoder.cpp',
  'abi/c_api_frame.cpp',
  'abi/c_api_nn.cpp',
  'abi/c_api_dropout.cpp',
  'abi/c_api_dropout_detect.cpp',
  'abi/c_api_pipeline.cpp',
  'abi/c_api_errors.cpp',
  'abi/c_api_log.cpp',
  'common/color_conversion.cpp',
  'common/error_state.cpp',
  'common/log.cpp',
  'format/sample_encoding.cpp',
  'format/signal_state.cpp',
  'format/video_standards.cpp',
  'metadata/core.cpp',
  'metadata/cvbs_metadata_sqlite.cpp',
  'metadata/dropout_extension_sqlite.cpp',
  'metadata/dropouts.cpp',
  'metadata/json_io.cpp',
  'metadata/ld_metadata_json.cpp',
  'metadata/ld_metadata_sqlite.cpp',
  'metadata/sqlite_query.cpp',
  'metadata/vbi_decoder.cpp',
  'reader/cvbs_composite_source.cpp',
  'reader/cvbs_yc_source.cpp',
  'reader/tbc_source.cpp',
  'decoders/registry.cpp',
  'decoders/ntsc_burst.cpp',
  'decoders/source_field.cpp',
  'decoders/mono/mono_decoder.cpp',
  'decoders/palcolour/palcolour.cpp',
  'decoders/palcolour/pal_decoder.cpp',
  'decoders/palcolour/transform_pal.cpp',
  'decoders/palcolour/transform_pal_2d.cpp',
  'decoders/palcolour/transform_pal_3d.cpp',
  'decoders/comb/comb.cpp',
  'decoders/comb/ntsc_decoder.cpp',
  'decoders/secam/secam_decoder.cpp',
  'dropout/dropout_corrector.cpp',
  'dropout/luma_filters.cpp',
  'dropout/multi_source_alignment.cpp',
  'pipeline/decoder_pool.cpp',
  'output/component_frame.cpp',
  'output/output_writer.cpp',
  'output/frame_canvas.cpp',
)

chromadec_deps = [threads_dep, sqlite3_dep, fftw3_dep]
chromadec_cpp_args = []
chromadec_cuda_args = []
# Backend-agnostic NN framework scaffolding: the decoders and the shared
# nnTransform3D FFT/window helpers. Built whenever any NN backend is present
# (has_nn = has_ort or has_coreml). ldzeug runs on either backend through the
# InferenceEngine interface; the nnTransform3D FFT/window helpers back both the
# ORT CPU path and the native CoreML host-FFT path.
if has_nn
  chromadec_sources += files(
    'decoders/nntransform3d/nntransform3d_fft_cpu.cpp',
    'decoders/nntransform3d/nntransform3d_window.cpp',
    'decoders/ldzeug/ldzeug_base.cpp',
    'decoders/ldzeug/ldzeug_color_cnn.cpp',
    'decoders/ldzeug/ldzeug_luma_sep.cpp',
  )
  chromadec_cpp_args += ['-DCHD_WITH_NN=1']
endif
# ONNX Runtime backend: the OrtEngine/OrtSession implementation and EP selection.
if has_ort
  chromadec_sources += files(
    'nn/ort_env.cpp',
    'nn/ort_session.cpp',
    'nn/ort_engine.cpp',
    'nn/provider_select.cpp',
  )
  chromadec_deps += [ort_dep]
  chromadec_cpp_args += ['-DCHD_WITH_ORT=1']
endif
if has_fftw
  chromadec_cpp_args += ['-DCHD_WITH_FFTW=1']
endif
if has_cuda
  # nnTransform3D's device-resident pipeline: cuFFT for the 3D FFTs +
  # ORT IoBinding so the CUDA / TensorRT EP can read inputs and write
  # outputs from device memory without round-tripping through the
  # host. .cu sources are compiled by nvcc (meson dispatches via the
  # `cuda` language declared in the top-level meson.build).
  chromadec_sources += files(
    'decoders/nntransform3d/nntransform3d_pipeline_cuda.cu',
  )
  chromadec_deps += [cuda_dep]
  chromadec_cpp_args += ['-DCHD_WITH_CUDA=1']
  # sm_60 (Pascal) is the floor for native FP64 atomicAdd, which the
  # OLA kernel needs. CUDA 13 dropped offline compilation for anything
  # below Turing, so 13.x toolchains build at sm_75 instead (still above
  # the atomicAdd floor); the resulting PTX JITs forward to newer GPUs
  # at runtime either way. A CUDA 13 build therefore does not run on
  # pre-Turing cards; ship a CUDA 12 build for those.
  if cuda_dep.version().version_compare('>=13.0')
    chromadec_cuda_args += ['-arch=sm_75']
  else
    chromadec_cuda_args += ['-arch=sm_60']
  endif
endif

if has_rocm
  # AMD ROCm counterpart of the CUDA path. Meson has no first-class HIP
  # language, so the pipeline source is compiled by hipcc via
  # custom_target and the resulting object is linked into the library.
  # ROCm install prefix + ORT include path are passed explicitly because
  # custom_target doesn't propagate include dirs from dependency objects.
  rocm_compile_args = [
    '-c', '-fPIC', '-std=c++17', '-O2',
    '-I' + meson.project_source_root() / 'include',
    '-I' + meson.project_source_root() / 'src',
    '-I' + meson.project_build_root() / 'include',
  ]
  # The include directories resolved from -Donnxruntime_root, rather than the
  # option joined with 'include': a root can also be a NuGet package, whose
  # headers live somewhere else entirely.
  foreach incdir : ort_incdirs
    rocm_compile_args += '-I' + incdir
  endforeach
  # hipcc writes the header dependencies with -MD -MF; without a depfile Meson
  # would treat the object as depending on the .cpp alone and editing any header
  # it includes would leave a stale object behind.
  hip_pipeline_o = custom_target(
    'nntransform3d_pipeline_hip',
    input: 'decoders/nntransform3d/nntransform3d_pipeline_hip.cpp',
    output: 'nntransform3d_pipeline_hip.o',
    depfile: 'nntransform3d_pipeline_hip.d',
    command: [hipcc] + rocm_compile_args
             + ['-MD', '-MF', '@DEPFILE@', '@INPUT@', '-o', '@OUTPUT@'],
  )
  chromadec_sources += hip_pipeline_o
  chromadec_deps += [hipfft_dep, hip_rt_dep]
  chromadec_cpp_args += ['-DCHD_WITH_ROCM=1']
endif

# Native CoreML path (macOS). The .mm sources are compiled by the `objcpp`
# language; they share the same defines + include dirs as the C++ sources, so
# objcpp_args mirrors cpp_args (plus ARC). The C++ standard comes from the
# project-wide cpp_std, which objcpp shares. Must come after the blocks above so
# chromadec_cpp_args is fully populated before we copy it.
chromadec_objcpp_args = []
if has_coreml
  chromadec_sources += files(
    'nn/coreml_engine.mm',
    'decoders/nntransform3d/nntransform3d_pipeline_coreml.mm',
  )
  chromadec_deps += [coreml_frameworks]
  chromadec_cpp_args += ['-DCHD_WITH_COREML=1']
  chromadec_objcpp_args = chromadec_cpp_args + ['-fobjc-arc']
endif

# Symbol export discipline:
# - On ELF (Linux), restrict exported symbols via a version script, whose node
#   name carries chromadec_abi (configured from meson/chromadec.map.in).
# - On macOS, restrict via -exported_symbols_list. Mach-O has no symbol
#   versioning, so the plain symbol list needs no configuring; the ABI version
#   reaches it through the install_name, which embeds the soversion.
# - On Windows, restrict via a .def file. Nothing is exported without one, since
#   the sources carry no __declspec(dllexport). Both filters above glob chd_*;
#   a .def has no wildcard, so its export set is generated from the public
#   headers at build time instead of being hand-maintained beside them.
#
# The filter file is named inside a raw link_args string, which Meson does not
# scan for filenames, so it is also passed as link_depends to make editing the
# exported-symbol set relink the library. (On Windows vs_module_defs already
# tracks it, but passing it twice is harmless and keeps the branches uniform.)
link_args = []
symbol_filter = []
# vs_module_defs rejects an empty value, so it is only passed at all on Windows,
# via kwargs, rather than defaulted to something inert on the other platforms.
def_kwarg = {}
host_os = host_machine.system()
if host_os == 'linux' or host_os.endswith('bsd')
  symbol_filter = configure_file(
    input:         chromadec_map_in,
    output:        'chromadec.map',
    configuration: {'CHROMADEC_ABI': chromadec_abi})
  link_args += ['-Wl,--version-script,' +
                meson.current_build_dir() / 'chromadec.map']
elif host_os == 'darwin'
  symbol_filter = chromadec_exports
  link_args += ['-Wl,-exported_symbols_list,' +
                meson.project_source_root() / 'meson/chromadec.exports']
elif host_os == 'windows'
  symbol_filter = custom_target('chromadec.def',
    input:   [public_headers, version_h],
    output:  'chromadec.def',
    command: [find_program('python3', 'python'), chromadec_def_gen,
              '@OUTPUT@', '@INPUT@'])
  def_kwarg = {'vs_module_defs': symbol_filter}
endif

libchromadec = library('chromadec',
  chromadec_sources,
  include_directories: [public_inc, version_inc],
  dependencies: chromadec_deps,
  link_whole: sqlite3_whole,
  cpp_args: chromadec_cpp_args,
  cuda_args: chromadec_cuda_args,
  objcpp_args: chromadec_objcpp_args,
  # NOTE: do NOT add -fvisibility=hidden. The version script (ELF) / exports
  # list (Mach-O) / .def (Windows) is the source of truth for what's exported.
  # -fvisibility=hidden would hide every internal symbol *and* the C ABI
  # before the linker filter even runs.
  link_args: link_args,
  link_depends: symbol_filter,
  kwargs: def_kwarg,
  install: true,
  # Not the major: pre-1.0 the ABI breaks at the minor, so the SONAME has to
  # move with it. Collapses to plain major-tracking from 1.0 on.
  soversion: chromadec_abi,
  version: meson.project_version())
