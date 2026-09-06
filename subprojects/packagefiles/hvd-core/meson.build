# Meson overlay for hvd-core (patch_directory in hvd-core.wrap). Upstream is
# the CMake-built hvd-cvbs-decoding monorepo; only its hvd-core/ engine subtree
# is built here, not the decode-orc plugin, Python package, or test suite.
#
# OpenMP is a feature here, where upstream's CMake forces -fopenmp. Every
# `#pragma omp` is _OPENMP-guarded, so without it the IRLS/CG solver loops
# compile serial and parallelism comes from libchromadec's frame-level pools
# alone; threaded, they cut single-frame latency for one-frame-at-a-time
# callers. The pragmas are all OpenMP 2.0 forms, so MSVC's /openmp takes them;
# Apple Clang bundles no runtime, so detection needs Homebrew libomp and
# Meson 1.5+.
project('hvd-core', 'cpp',
  version: '0.1.3',
  license: 'GPL-3.0-or-later',
  meson_version: '>=1.2.0',
  default_options: {
    'cpp_std': 'c++17',
    'warning_level': '0',
  })

fftw3f_dep  = dependency('fftw3f')
threads_dep = dependency('threads')
openmp_dep  = dependency('openmp', required: get_option('openmp'))
# Read by the parent for its feature summary.
hvd_openmp_enabled = openmp_dep.found()

# Parity with upstream's CMakeLists.txt: engine headers reach platform headers
# on Windows, so keep <windows.h> from defining min/max as macros.
hvd_core_args = []
if host_machine.system() == 'windows'
  hvd_core_args += ['-DNOMINMAX', '-DWIN32_LEAN_AND_MEAN']
endif

hvd_core_inc = include_directories('hvd-core/src')

hvd_core_lib = static_library('hvd_core',
  files(
    'hvd-core/src/engine/ntsc_geometry.cpp',
    'hvd-core/src/engine/lockin.cpp',
    'hvd-core/src/engine/fft2d.cpp',
    'hvd-core/src/engine/holographic_init.cpp',
    'hvd-core/src/engine/variational.cpp',
    'hvd-core/src/engine/colour.cpp',
    'hvd-core/src/engine/motion.cpp',
    'hvd-core/src/engine/temporal.cpp',
    'hvd-core/src/engine/sequence.cpp',
    'hvd-core/src/engine/engine.cpp',
    'hvd-core/src/frame_bridge.cpp',
  ),
  cpp_args: hvd_core_args,
  include_directories: hvd_core_inc,
  dependencies: [fftw3f_dep, threads_dep, openmp_dep],
  gnu_symbol_visibility: 'hidden',
  pic: true,
  install: false)

hvd_core_dep = declare_dependency(
  link_with: hvd_core_lib,
  include_directories: hvd_core_inc,
  dependencies: [fftw3f_dep, threads_dep, openmp_dep])
meson.override_dependency('hvd-core', hvd_core_dep)
