project('chromadec', 'cpp', 'c',
        # Single source of truth for the version. The Nix derivation reads the
        # same file, and scripts/check-version.sh gates the copies that cannot
        # read it (Cargo manifests, release tags).
        #
        # The .txt suffix is not decoration. The source root is on the include
        # path, so on a case-insensitive filesystem a file named plain VERSION
        # is what `#include <version>` resolves to, and every translation unit
        # that reaches the C++20 header through <string> or <filesystem> fails
        # to compile. Only macOS and Windows would break, which is to say the
        # two platforms we ship binaries for.
        version: files('VERSION.txt'),
        license: 'GPL-3.0-or-later',
        license_files: 'LICENSE',
        meson_version: '>=1.8.0',
        default_options: {
            # cpp_std also drives the Objective-C++ sources: since Meson 1.8 the
            # objcpp compiler shares the C++ standard option.
            'cpp_std': 'c++17',
            'c_std': 'c11',
            'cuda_std': 'c++17',
            'warning_level': '2',
            'default_library': 'both',
            'b_ndebug': 'if-release',
            # Compute the .pc file's `prefix` relative to where the file is
            # installed, so a relocated or vendored install tree resolves without
            # anyone rewriting the prefix by hand.
            'pkgconfig.relocatable': true,
            # A consumer static-linking libchromadec absorbs the bundled SQLite
            # (see the link_whole below), so install the licence texts plus a
            # manifest naming every subproject that ended up in the archive.
            # Packagers can redirect or empty this.
            'licensedir': 'share/licenses/chromadec',
        })

cpp = meson.get_compiler('cpp')
cc  = meson.get_compiler('c')

# The Microsoft CRT's <cmath> hides M_PI and the other POSIX math constants
# behind _USE_MATH_DEFINES; the decoders rely on M_PI throughout. Define it
# project-wide so every translation unit sees the constants, rather than guarding
# each file. The test is on argument syntax, not compiler id: clang-cl needs
# this too.
if cpp.get_argument_syntax() == 'msvc'
  add_project_arguments('-D_USE_MATH_DEFINES', language: ['c', 'cpp'])
endif

# ── ABI version ───────────────────────────────────────────────────────────────
# The version at which the ABI is allowed to break: the minor before 1.0, the
# major from 1.0 on (docs/abi-stability.md). Both the shared library's soversion
# and the ELF version-script node track it, so a breaking release lands under a
# SONAME that the previous release's consumers will not load. Derived rather than
# written down twice, because the two drifting apart is the failure this guards
# against.
version_components = meson.project_version().split('.')
if version_components[0] == '0'
  chromadec_abi = version_components[0] + '.' + version_components[1]
else
  chromadec_abi = version_components[0]
endif

# Consumed in src/meson.build. files() rejects '..', so the symbol-filter inputs
# are named here, where meson/ is a child of the source root.
chromadec_map_in  = files('meson/chromadec.map.in')
chromadec_exports = files('meson/chromadec.exports')
chromadec_def_gen = files('meson/gen-module-def.py')

# ── Dependencies ──────────────────────────────────────────────────────────────
threads_dep = dependency('threads')

# SQLite3 is required: TBC metadata sidecars can live in SQLite files
# (`*.tbc.db`) and chd_video_open_composite cannot work without it. A system libsqlite3
# is linked as-is; when none is found, the bundled overlay (subprojects/
# packagefiles/sqlite3) builds a trimmed, core-SQL-only static copy that gets
# folded into libchromadec via link_whole below rather than shipped separately.
sqlite3_dep = dependency('sqlite3', required: true)
fftw3_dep   = dependency('fftw3',   required: false)

# ONNX Runtime discovery. An explicitly configured -Donnxruntime_root is checked
# BEFORE pkg-config and CMake, because it is the caller naming the exact copy to
# link, usually a Microsoft release chosen for the execution providers it
# carries. Letting auto-discovery run first is not a harmless preference: the
# `onnxruntime` lookup falls through to CMake, so on a machine with a system copy
# installed the build would link that instead and ignore the root entirely,
# producing a library bound to a path the consumer does not have and to whatever
# version and provider set that copy happens to carry.
#
# Two on-disk layouts are accepted, because Microsoft publishes both. The
# release archives put the library in lib/ beside include/. The NuGet packages
# split them into runtimes/<rid>/native and build/native/include, and that is
# the only form the DirectML-carrying Windows package is published in, so a
# consumer who wants that execution provider has no archive to point at.
ort_root    = get_option('onnxruntime_root')
ort_dep     = dependency('', required: false)
ort_dll_dir = ''
ort_incdirs = []
if get_option('with_onnxruntime') and ort_root != ''
  fs = import('fs')
  nuget_rid = host_machine.cpu_family() == 'aarch64' ? 'win-arm64' : 'win-x64'
  ort_libdirs = []
  foreach candidate : [[ort_root / 'lib', ort_root / 'include'],
                       [ort_root / 'runtimes' / nuget_rid / 'native',
                        ort_root / 'build' / 'native' / 'include']]
    if fs.is_dir(candidate[0])
      ort_libdirs += candidate[0]
    endif
    if fs.is_dir(candidate[1])
      ort_incdirs += candidate[1]
    endif
  endforeach

  ort_lib = dependency('', required: false)
  if ort_libdirs.length() > 0
    ort_lib = cpp.find_library('onnxruntime', dirs: ort_libdirs, required: false)
  endif
  # A root the caller named but the build cannot use is fatal rather than a
  # warning: falling through to pkg-config and CMake discovery would quietly
  # link whatever other ONNX Runtime the machine happens to carry, which is the
  # copy the caller was explicitly steering away from, and with a different
  # execution-provider set than the one they chose the root for.
  if not ort_lib.found() or ort_incdirs.length() == 0
    error('onnxruntime_root=' + ort_root + ' is not a usable ONNX Runtime root. ' +
          'Expected the release-archive layout (lib/ + include/) or the NuGet ' +
          'package layout (runtimes/' + nuget_rid + '/native + ' +
          'build/native/include). Pass a different root, drop the option to ' +
          'discover ORT via pkg-config or CMake, or set -Dwith_onnxruntime=false.')
  endif
  ort_dep = declare_dependency(
    dependencies: ort_lib,
    include_directories: include_directories(ort_incdirs))
  # Where the runtime DLLs sit, for the Windows tests: the loader searches the
  # running executable's own directory ahead of System32, so they get copied
  # beside the test binaries (see tests/meson.build).
  ort_dll_dir = ort_libdirs[0]
endif
if not ort_dep.found()
  ort_dep = dependency('libonnxruntime', required: false)
endif
if not ort_dep.found()
  ort_dep = dependency('onnxruntime', required: false)
endif

# ONNX Runtime backend. The NN framework (interface, decoders, C ABI) is
# backend-agnostic; `has_nn` (the umbrella "any NN backend present") is computed
# after CoreML discovery below as `has_ort or has_coreml`.
has_ort    = get_option('with_onnxruntime') and ort_dep.found()
has_fftw   = fftw3_dep.found()
has_sqlite = sqlite3_dep.found()

# When SQLite comes from the bundled overlay (no system copy), fold its objects
# into libchromadec via link_whole so the installed static archive is self-
# contained for SQLite and no standalone libsqlite3 is published. A system SQLite
# stays external, resolved through Requires.private.
sqlite3_whole = []
if has_sqlite and sqlite3_dep.type_name() == 'internal'
  sqlite3_whole = [subproject('sqlite3').get_variable('libsqlite3')]
endif

# ── CUDA toolkit discovery ────────────────────────────────────────────────────
# When `with_cuda=auto` (the default), the build tries to find the CUDA
# toolkit + cuFFT and silently falls back to the CPU FFTW path if either
# isn't installed. `enabled` turns missing dependencies into a configure
# error; `disabled` forces the CPU path even where CUDA is available.
#
# The runtime EP-select chain in src/nn/provider_select.cpp is independent
# of this flag: a session can still attach the CUDA / TensorRT EP, the
# library just can't take the device-resident pipeline shortcut without
# this build option — every inference round-trips through host memory.
cuda_opt = get_option('with_cuda').disable_if(
  not has_ort,
  error_message: 'the CUDA pipeline feeds ONNX Runtime IoBinding and needs the ORT backend')
has_cuda = false
if not cuda_opt.disabled()
  cuda_dep = dependency('cuda',
                        version: '>=11.0',
                        modules: ['cudart', 'cufft'],
                        required: cuda_opt)
  if cuda_dep.found()
    cuda_lang_ok = add_languages('cuda', required: cuda_opt, native: false)
    if cuda_lang_ok
      has_cuda = true
    endif
  endif
endif

# ── ROCm / HIP toolkit discovery ──────────────────────────────────────────────
# Mirror of the CUDA discovery block, but for AMD's HIP runtime + hipFFT.
# Meson has no first-class HIP language, so the HIP pipeline source is
# compiled by hipcc via a custom_target in src/meson.build rather than
# being listed alongside the C++ sources. ROCm is Linux-only.
rocm_opt = get_option('with_rocm').disable_if(
  not has_ort,
  error_message: 'the HIP pipeline feeds ONNX Runtime IoBinding and needs the ORT backend'
).disable_if(
  host_machine.system() != 'linux',
  error_message: 'ROCm is Linux-only')
has_rocm = false
if not rocm_opt.disabled()
  hipcc = find_program('hipcc', required: rocm_opt)
  # Always include /opt/rocm (AMD's apt/dnf default on Ubuntu/RHEL/SLES);
  # CMake silently ignores it on distros installing under /usr. User-set
  # rocm_root takes precedence for non-standard installs.
  hip_cmake_prefixes = ['/opt/rocm']
  rocm_root = get_option('rocm_root')
  if rocm_root != ''
    hip_cmake_prefixes = [rocm_root] + hip_cmake_prefixes
  endif
  hip_cmake_args = ['-DCMAKE_PREFIX_PATH=' + ';'.join(hip_cmake_prefixes)]
  hipfft_dep = dependency('hipfft',
                          cmake_args: hip_cmake_args,
                          required: rocm_opt)
  # `hip::host` is the host-only imported target — propagates
  # libamdhip64.so and `-D__HIP_PLATFORM_AMD__=1`. Bypasses
  # `hip::device` (CMake's default) which pulls in GPU bitcode meant
  # for hipcc, not the host C++ linker.
  hip_rt_dep = dependency('hip',
                          modules: ['hip::host'],
                          cmake_args: hip_cmake_args,
                          required: rocm_opt)
  if hipcc.found() and hipfft_dep.found() and hip_rt_dep.found()
    has_rocm = true
  endif
endif

# ── CoreML / Metal toolkit discovery (macOS only) ─────────────────────────────
# Native CoreML inference for the models the ORT CoreML execution provider
# can't accelerate (nnTransform3D's 3D conv) plus an MPSGraph device FFT. The
# Objective-C++ sources are compiled by the `objcpp` language (added here,
# native: false so it targets the host like the rest of the library) and link
# against the CoreML / Metal system frameworks.
#
# This is a fully independent inference backend — it does NOT require ONNX
# Runtime. `with_onnxruntime=false -Dwith_coreml=enabled` yields a CoreML-only
# library on macOS. (It also sits alongside the ORT CoreML EP when ORT is
# present, adding the native .mlpackage path the EP can't accelerate.)
coreml_opt = get_option('with_coreml').disable_if(
  host_machine.system() != 'darwin',
  error_message: 'the native CoreML backend is macOS-only')
has_coreml = false
if not coreml_opt.disabled()
  objcpp_ok = add_languages('objcpp', required: coreml_opt, native: false)
  # MetalPerformanceShadersGraph carries the GPU FFT used by nnTransform3D
  # Layer 2; CoreML + Foundation + Metal cover model load + buffer interop.
  coreml_frameworks = dependency('appleframeworks',
                                 modules: ['CoreML', 'Foundation', 'Metal',
                                           'MetalPerformanceShadersGraph'],
                                 required: coreml_opt)
  if objcpp_ok and coreml_frameworks.found()
    has_coreml = true
  endif
endif

# The NN framework (the chd::nn::InferenceEngine interface, the ldzeug and
# nnTransform3D decoders, and the chd_nn_* C ABI) is backend-agnostic and is
# built when EITHER inference backend is present.
has_nn = has_ort or has_coreml

# If the user explicitly requested the ONNX Runtime backend but ORT is missing,
# fail loudly rather than silently dropping it. The default for with_onnxruntime
# is `true`, so silent absence would mean every Linux/macOS dev build without ORT
# installed would quietly omit it — and that bites later when an integration test
# loads a model expecting ORT to be present. (On macOS you can instead build the
# native CoreML backend alone with -Dwith_onnxruntime=false -Dwith_coreml=enabled.)
if get_option('with_onnxruntime') and not ort_dep.found()
  error('with_onnxruntime=true but ONNX Runtime was not found via pkg-config or ' +
        '-Donnxruntime_root. Install ORT (Homebrew formula `onnxruntime`, ' +
        'apt `libonnxruntime-dev`, or download Microsoft GitHub release) or ' +
        'pass -Dwith_onnxruntime=false to disable the ORT backend.')
endif

# ── Public headers + library (include/ defines version_h via configure_file,
#    placed at build/include/chromadec/version.h so <chromadec/version.h> works).
subdir('include')
subdir('src')

# ── Subproject dependency ─────────────────────────────────────────────────────
# Let a parent Meson project consume this tree as a subproject without an install
# step. override_dependency makes the parent's `dependency('chromadec')` resolve
# to this object when no system copy is found, so vendored and installed
# consumption use the identical line.
chromadec_dep = declare_dependency(
  link_with: libchromadec,
  include_directories: [public_inc, version_inc])
meson.override_dependency('chromadec', chromadec_dep)

if get_option('with_tests')
  subdir('tests')
endif

if get_option('with_examples')
  subdir('examples')
endif

# ── pkg-config + CMake config installation ────────────────────────────────────
# libchromadec is implemented in C++ behind a C ABI and ships a static archive
# (default_library=both). A shared object records its language runtime as a
# NEEDED entry, but a static archive does not, so a C consumer static-linking via
# `pkg-config --static` must put the C++ standard library on its own link line.
# pkg-config has no portable way to query a library's link language, so emit the
# stdlib into Libs.private ourselves. Detect the one actually in use by probing
# its version macro (honours a forced -stdlib=): _LIBCPP_VERSION means libc++,
# otherwise assume libstdc++. A compiler driving the MSVC runtime (cl.exe or
# clang-cl) links it implicitly through #pragma comment(lib), so emit nothing.
cxx_runtime_libs = []
if cpp.get_argument_syntax() != 'msvc'
  if cpp.get_define('_LIBCPP_VERSION', prefix: '#include <utility>') != ''
    cxx_runtime_libs = ['-lc++']
  else
    cxx_runtime_libs = ['-lstdc++']
  endif
endif

pkg = import('pkgconfig')
pkg.generate(libchromadec,
             name: 'chromadec',
             description: 'chroma-decode: composite video chroma decoding library',
             version: meson.project_version(),
             url: 'https://github.com/JustinTArthur/chroma-decode',
             libraries_private: cxx_runtime_libs,
             # Advertise the link language for tooling that learns to consume it
             # (Meson custom-variable / pkgconf Link.ABI / CPS link_languages).
             variables: ['link_language=c++'])

cmake_mod = import('cmake')
cmake_dir = get_option('libdir') / 'cmake' / 'chromadec'

# Hand-rolled rather than cmake_mod.write_basic_package_version_file(), which
# would make CMake a build-time dependency.
cmake_ver_cfg = configuration_data()
cmake_ver_cfg.set('CVF_VERSION', meson.project_version())
configure_file(
  input:         'meson/chromadec-configversion.cmake.in',
  output:        'chromadecConfigVersion.cmake',
  configuration: cmake_ver_cfg,
  install:       true,
  install_dir:   cmake_dir)

cmake_pkg_cfg = configuration_data()
cmake_pkg_cfg.set('VERSION', meson.project_version())
cmake_mod.configure_package_config_file(
  name: 'chromadec',
  input: 'meson/chromadec-config.cmake.in',
  configuration: cmake_pkg_cfg)

# The imported-target definition chromadecConfig.cmake includes. Written by
# hand and configured here because the cmake module has no install(EXPORT)
# equivalent; see the header comment in the template, and mesonbuild/meson#10504.
#
# CMake reads backslashes in a quoted string as escapes, so the Windows paths
# meson hands back have to be normalised to forward slashes, which CMake accepts
# on every platform.
cmake_tgt_cfg = configuration_data()
cmake_tgt_cfg.set('CHROMADEC_LIBDIR',
                  (get_option('prefix') / get_option('libdir')).replace('\\', '/'))
cmake_tgt_cfg.set('CHROMADEC_INCLUDEDIR',
                  (get_option('prefix') / get_option('includedir')).replace('\\', '/'))
cmake_tgt_cfg.set('CHROMADEC_BINDIR',
                  (get_option('prefix') / get_option('bindir')).replace('\\', '/'))
configure_file(
  input:         'meson/chromadec-targets.cmake.in',
  output:        'chromadec-targets.cmake',
  configuration: cmake_tgt_cfg,
  install:       true,
  install_dir:   cmake_dir)

# ── Summary ───────────────────────────────────────────────────────────────────
summary({
  'ONNX Runtime backend':          has_ort,
  'CUDA cuFFT pipeline (NV/TRT)':  has_cuda,
  'HIP hipFFT pipeline (ROCm/MX)': has_rocm,
  'CoreML pipeline (macOS)':       has_coreml,
  'FFTW3 (Transform-PAL)':         has_fftw,
  'SQLite3 (metadata)':            has_sqlite,
  'Tests':                         get_option('with_tests'),
  'Examples':                      get_option('with_examples'),
  'Debug overlay':                 get_option('with_debug_overlay'),
}, section: 'Optional features')
