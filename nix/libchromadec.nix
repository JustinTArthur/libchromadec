{ lib
, stdenv
, meson
, ninja
, cmake
, pkg-config
, python3
, sqlite
, fftw

# ── ONNX Runtime seam ────────────────────────────────────────────────────────
# `onnxruntime` is left as a plain argument rather than being pinned inside the
# derivation, so a consumer that already carries its own ORT (a Microsoft
# prebuilt with the CUDA or CoreML execution providers, say) can pass that copy
# in and keep a single ORT in the closure:
#
#   libchromadec.override { onnxruntime = myPrebuiltOrt; }
#
# Two install layouts have to work here, and they need different plumbing:
#
#   * Multi-output, nixpkgs style (headers in `$dev`, libraries in `$out`).
#     There is no single prefix containing both, so discovery goes through
#     pkg-config, which the dev output's `libonnxruntime.pc` drives.
#   * Single-output, Microsoft-prebuilt style (`include/` and `lib/` under one
#     root, no .pc file at all). pkg-config finds nothing, so the prefix is
#     handed to meson's `onnxruntime_root` fallback instead.
#
# The default below picks whichever applies; override `onnxruntimeRoot`
# directly for a layout that fits neither.
, onnxruntime ? null
, onnxruntimeRoot ? null
, withOnnxruntime ? onnxruntime != null

# CoreML is an independent inference backend, not an ORT execution provider, so
# it can be built with or without ORT present.
, withCoreml ? stdenv.hostPlatform.isDarwin
# macOS SDK providing CoreML / Metal / MetalPerformanceShadersGraph. nixpkgs
# releases before the apple-sdk rework have no such attribute; the flake passes
# null there and the frameworks come from the stdenv.
, apple-sdk ? null

, cudaSupport ? false
, cudaPackages ? null

, withTests ? false
}:

let
  # An ORT whose only output is `out` keeps include/ and lib/ under one prefix,
  # which is exactly what -Donnxruntime_root expects. Anything split across
  # outputs has to be found by pkg-config instead.
  ortIsSinglePrefix =
    onnxruntime != null && (onnxruntime.outputs or [ "out" ]) == [ "out" ];

  ortRoot =
    if onnxruntimeRoot != null then onnxruntimeRoot
    else if ortIsSinglePrefix then "${onnxruntime}"
    else null;

  mesonFeature = enabled: if enabled then "enabled" else "disabled";
in

assert cudaSupport -> cudaPackages != null;
assert cudaSupport -> withOnnxruntime;

stdenv.mkDerivation (finalAttrs: {
  pname = "libchromadec";
  # Same file meson's project() reads, so the store path name cannot drift from
  # the version baked into chromadec.pc and the soversion. readFile keeps this
  # a pure evaluation, no import-from-derivation, so the flake still evaluates
  # without building anything.
  version = lib.fileContents ../VERSION.txt;

  # cleanSource strips VCS and editor droppings but keeps local build trees. A
  # stale build/ reaching the sandbox makes `meson setup build` take its
  # reconfigure path against a configuration produced by a completely different
  # toolchain, so the build directories are excluded by name as well. (Flake
  # consumption filters through git and never sees them; a plain callPackage on
  # a working tree does.)
  src =
    let
      excludedRoots = [
        "build" "buildDir" "result" "site" "dist-local"
        ".venv" ".venv-coreml" ".direnv" "__pycache__"
      ];
    in
    lib.cleanSourceWith {
      name = "libchromadec-source";
      src = lib.cleanSource ../.;
      filter = path: type:
        !(type == "directory" && builtins.elem (baseNameOf path) excludedRoots);
    };

  outputs = [ "out" "dev" ];

  strictDeps = true;

  nativeBuildInputs = [
    meson
    ninja
    # Only ever run as a tool: meson's cmake module shells out to it to write
    # chromadec-config.cmake and its version file. The build itself is meson,
    # so cmake's setup hook must not claim the configure phase.
    cmake
    pkg-config
    # Generates the Windows .def from the public headers. Unused on Unix, but
    # meson's find_program runs during configure regardless of host.
    python3
  ] ++ lib.optionals cudaSupport [
    cudaPackages.cuda_nvcc
  ];

  buildInputs = [
    sqlite
    fftw
  ] ++ lib.optionals withOnnxruntime [
    onnxruntime
  ] ++ lib.optionals (withCoreml && apple-sdk != null) [
    apple-sdk
  ] ++ lib.optionals cudaSupport [
    cudaPackages.cuda_cudart
    cudaPackages.libcufft
  ];

  dontUseCmakeConfigure = true;

  # nixpkgs' meson hook defaults to --buildtype=plain, which leaves b_ndebug at
  # its non-release value and keeps every assert live in a consumer's hot loop.
  mesonBuildType = "release";

  mesonFlags = [
    (lib.mesonBool "with_onnxruntime" withOnnxruntime)
    (lib.mesonOption "with_coreml" (mesonFeature withCoreml))
    (lib.mesonOption "with_cuda" (mesonFeature cudaSupport))
    # HIP needs hipcc driving a separate compile step, which does not survive
    # the meson/nix cross-compilation split cleanly. Build it outside Nix.
    (lib.mesonOption "with_rocm" "disabled")
    (lib.mesonBool "with_tests" withTests)
    (lib.mesonBool "with_examples" false)
    (lib.mesonBool "with_debug_overlay" false)
    # The project defaults this on so an extracted release archive resolves
    # wherever it lands, but it derives `prefix` from the .pc file's own
    # directory. Splitting headers and pkg-config into $dev then makes
    # `libdir=${prefix}/lib` point at $dev/lib, which holds no library, and a
    # consumer links with a bare -lchromadec and no -L that finds it. Nix store
    # paths are absolute and never relocated, so pin them absolute here.
    (lib.mesonBool "pkgconfig.relocatable" false)
    # Keep the bundled-SQLite licence manifest out of the fixed-output paths
    # packagers care about; sqlite comes from nixpkgs here anyway.
    (lib.mesonOption "licensedir" "share/licenses/libchromadec-${finalAttrs.version}")
  ] ++ lib.optionals (ortRoot != null) [
    (lib.mesonOption "onnxruntime_root" ortRoot)
  ];

  # The suite drives real TBC fixtures through encode-orc, which is not
  # packaged here, so checks stay off unless a caller opts in.
  doCheck = withTests;

  meta = {
    description = "Composite video chroma decoding library";
    homepage = "https://github.com/JustinTArthur/libchromadec";
    license = lib.licenses.gpl3Plus;
    pkgConfigModules = [ "chromadec" ];
    platforms = lib.platforms.unix ++ lib.platforms.windows;
  };
})
