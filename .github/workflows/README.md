# CI workflows

| Workflow | Purpose                                                                                                                                             | Status |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------------|---|
| `build.yml` → `fixtures` | Builds encode-orc once and synthesizes the NTSC + PAL colour-bars TBCs via `scripts/make-fixtures.sh`, publishing them as the `tbc-fixtures` artifact every test job below consumes | Active |
| `build.yml` → `linux` | Build + `meson test` (with encode-orc-driven integration) + symbol-surface check, matrix on x86_64 / arm64                                          | Active |
| `build.yml` → `linux-asan-ubsan` | ASan + UBSan build + `meson test` (with encode-orc-driven integration), x86_64 only                                                                 | Active |
| `build.yml` → `linux-tsan` | ThreadSanitizer build + `meson test` (with encode-orc-driven integration), x86_64 only                                                              | Active |
| `build.yml` → `macos` | Build + `meson test` (with encode-orc-driven integration) + symbol-surface check, matrix on arm64 / x86_64                                          | Active |
| `build.yml` → `windows` | MSVC build + `meson test` (with encode-orc-driven integration) + symbol-surface check via `dumpbin`, matrix on x64 / arm64; vcpkg for sqlite3/fftw3, Microsoft ORT zip. Both legs run natively, so each tests what it builds; the arm64 image ships no vcpkg, so the job bootstraps one | Active |
| `build.yml` → `version` | Gates the version copies that cannot read `VERSION.txt` themselves (Cargo manifests, and the release tag), via `scripts/check-version.sh` | Active |
| `release.yml` | On `v*` tags: packages relocatable binary archives with `chromadecConfig.cmake` + `chromadec.pc`, verifies layout and exports, smoke-tests a CMake consumer against each, then uploads to the release. Windows x64/arm64 (MSVC, `.zip`) and macOS arm64 (`.tar.xz`). `workflow_dispatch` runs everything but the upload | Active |
| `rust.yml` | Builds libchromadec, then gates the first-party Rust bindings: `cargo fmt --check`, clippy with warnings as errors, and `cargo test` including the encode-orc-driven stream integration. Linux x86_64 only — the crate is platform-portable and the C library is matrix-tested in `build.yml` | Active |
| `gpu-cuda.yml` | Nvidia GPU validation on AWS spot, manual only; runs the suite pinned to the CUDA EP, then the NN suites again pinned to TensorRT | Active (`workflow_dispatch`) |
| `gpu-amd.yml` | AMD GPU validation on AWS spot market, running inside `rocm/dev-ubuntu-24.04` Docker, manual only; HIP/hipFFT pipeline + MIGraphX EP                        | Active (`workflow_dispatch`) |
| ABI check | `abi-compliance-checker` against previous release `.so`                                                                                             | Not yet — added at first tagged release |

## Reusable actions

- `.github/actions/setup-encode-orc/` — composite action that clones,
  builds (with cache), and exports `CHD_ENCODE_ORC`. Pinned to a specific
  encode-orc commit via the `commit` input. encode-orc has no
  GitHub Releases — only ephemeral workflow artifacts — so source-build
  with commit-keyed cache is the durable pattern.

  Linux only, and deliberately: `build.yml` runs it in one `fixtures` job
  and hands the resulting TBCs to the other platforms as an artifact, which
  the integration test picks up through `CHD_ENCODE_ORC_FIXTURE_DIR`. A TBC
  is raw samples plus a SQLite sidecar and says nothing about the machine
  that wrote it, so building the encoder five times bought nothing but five
  chances to fail on a platform encode-orc does not support. It also means
  every platform decodes byte-identical input. `rust.yml` and the two GPU
  workflows still call the action directly; all three are Linux.

## AWS GPU runners

`gpu-cuda.yml` and `gpu-amd.yml` have no GitHub-hosted equivalent. Each launches
a single-use EC2 spot instance, registers it as an ephemeral self-hosted runner
under a per-run label, runs the suite on it, and terminates it. Both are manual
only; nothing launches on a schedule.

Required repository configuration:

| Name | Kind | Purpose |
|---|---|---|
| `AWS_GPU_RUNNER_ROLE_ARN` | variable | Role assumed via OIDC. Both workflows skip entirely while unset, so an unconfigured fork stays quiet instead of failing. |
| `GH_RUNNER_MGMT_TOKEN` | secret | Mints runner registration tokens and lists runners. Needs Administration read and write, which `GITHUB_TOKEN` cannot be granted. |

Environments:

| Environment | Job | Protection |
|---|---|---|
| `gpu-ci` | `launch-spot` | Branch policy and any approval requirement go here. |
| `gpu-ci-cleanup` | `cleanup` | Deliberately none. |

Naming an environment replaces the branch ref in the OIDC subject with
`environment:<name>`, so the role's trust policy lists the two environment
subjects and no branch patterns at all. Any branch may run these workflows,
while only jobs naming an environment can assume the role, and branch limits
stay in GitHub rather than in IAM. `cleanup` terminates the instance and must
never wait on an approval, hence its own unprotected environment instead of
sharing `gpu-ci`.

Instances self-terminate four independent ways: the runner is `--ephemeral` and
user-data ends in `shutdown -h now`; instances launch with
`instance-initiated-shutdown-behavior terminate`; user-data starts a wall-clock
watchdog as its first action, before any step that could fail; and `cleanup`
terminates by instance id regardless of how the run ended.

Regions are pinned per GPU vendor by spot price, and each region needs G-family
spot vCPU quota granted before a launch will succeed.

## ONNX Runtime artifacts

Downloaded from Microsoft's official GitHub releases on every platform.

| Job | Artifact | Pin |
|---|---|---|
| `linux` (x86_64) | `onnxruntime-linux-x64-${VER}.tgz` | 1.27.1 |
| `linux` (arm64) | `onnxruntime-linux-aarch64-${VER}.tgz` | 1.27.1 |
| `linux-asan-ubsan` / `linux-tsan` | `onnxruntime-linux-x64-${VER}.tgz` | 1.27.1 |
| `macos` (arm64) | `onnxruntime-osx-arm64-${VER}.tgz` | 1.27.1 |
| `macos` (x86_64) | none — ORT backend off (`no_ort`) | n/a |
| `windows` (x64) | `onnxruntime-win-x64-${VER}.zip` | 1.27.1 |
| `windows` (arm64) | `onnxruntime-win-arm64-${VER}.zip` | 1.27.1 |
| `release` (windows) | same per-arch zips as `windows` | 1.27.1 |
| `release` (macos arm64) | `onnxruntime-osx-arm64-${VER}.tgz` | 1.27.1 |
| `gpu-cuda` | `onnxruntime-linux-x64-gpu_cuda12-${VER}.tgz` | 1.27.1 |
| `gpu-amd` | `onnxruntime-migraphx` wheel from PyPI, headers from `onnxruntime-linux-x64` | 1.27.1 |

The CPU artifact is what the hosted jobs need — GH-hosted runners have no GPU
and `with_cuda=auto` falls back silently. CUDA requires the GPU artifact and a
self-hosted GPU runner. The GPU artifact is published per CUDA major
(`gpu_cuda12` / `gpu_cuda13`); we take `gpu_cuda12`, matching a CUDA 12 toolkit
on the deep-learning base AMI.

That archive also carries `libonnxruntime_providers_tensorrt.so`, which needs a
`libnvinfer` the AMI does not have and the archive does not bundle, so the job
installs TensorRT itself from NVIDIA's package index. Two traps there: ONNX
Runtime links the TensorRT **10** ABI while a plain `tensorrt-cu12` now resolves
to 11.x, and the pypi.org package is sdist-only, so the wheels have to come from
`pypi.nvidia.com`.

Because the Linux auto chain leads with TensorRT, asserting the provider that
AUTO chose cannot distinguish the two EPs. The tests take
`CHD_TEST_NN_BACKEND` to *request* one and `CHD_TEST_EXPECT_NN_BACKEND` to
assert what attached, and the job runs the suite pinned to CUDA and then the NN
suites pinned to TensorRT.

AMD publishes no C/C++ ONNX Runtime build for MIGraphX in any form: the wheel
carries only `capi/` runtime objects, with no headers and no linker symlinks,
and neither the ROCm apt repo nor the `rocm/onnxruntime` images fill the gap.
`gpu-amd.yml` therefore composes an install root at job time, taking headers
from Microsoft's release of the same version, libraries from the wheel, and
adding the `.so`/`.so.1` symlinks the wheel omits.

The wheel comes straight from PyPI at the same version as every other platform.
Reaching it is why the AMD job runs on the `24.04` ROCm image: the wheel requires
Python 3.11+, and the `22.04` image ships 3.10, which caps pip at ORT 1.24.2.
The MIGraphX EP's only ROCm coupling is `libmigraphx_c.so.3` and
`libamdhip64.so.7`, both satisfied by ROCm 7.2.3.

ROCm 10 drops that built-in EP for a plugin EP wheel,
`onnxruntime-ep-migraphx`, whose `libmigraphx-ep.so` lives outside any composed
root. `test_nn_framework` takes `CHD_TEST_NN_PLUGIN_LIBRARY` to name that file
and passes it to `chd_nn_backend_set_library_path` before anything loads.

macOS releases include the CoreML execution provider. CI takes Microsoft's
builds rather than a system package so every platform pins the same version
from the same publisher, not because a system copy is unusable.

**macOS x86_64 builds with no ORT at all.** Microsoft dropped the
`onnxruntime-osx-x86_64` asset after 1.23.2 (1.24.0 onward ships macOS arm64
only), and that frozen build is not reliably discoverable on the retiring
`macos-15-intel` runner, so the job sets `no_ort` and validates the C core plus
the native CoreML backend instead. Note what this does and does not cost: the
native CoreML backend is independent of ORT and works on Intel Macs, so NN
decoding still runs there, but only from `.mlpackage`/`.mlmodelc` artifacts.
`.onnx` models need ONNX Runtime and are unavailable in an ORT-less build.

Windows arm64 has no `-gpu-` variant published at all, so that leg takes the
plain archive. The release archives do not bundle ONNX Runtime on any platform:
libchromadec links it, and the consumer supplies a copy meeting the version
floor recorded in the package's `BUILD-INFO.txt`. That keeps them free to pick
their own execution providers without a second ONNX Runtime competing on the
library search path.

## Release archives

`release.yml` ships Windows x64/arm64 and macOS arm64. No Linux archive: a
prebuilt `.so` carries a glibc and libstdc++ floor with no equivalent of the
vcpkg triplet or the macOS deployment target to name it by, and Linux consumers
can build from source or take the Nix flake. No macOS x86_64 either, because
Microsoft published no `onnxruntime-osx-x86_64` asset after 1.23.2, so an
x86_64 package at our pinned ORT would ship with the NN decoders missing.

Everything the macOS package needs beyond system libraries is absorbed rather
than referenced: SQLite comes from the bundled amalgamation (`link_whole`) and
FFTW is built from source as a static library, so `otool -L` shows only
`/usr/lib`, system frameworks and `@rpath`. A CI step enforces that, since a
stray `/opt/homebrew` reference would be unresolvable on a consumer's machine.
Meson records the absolute install path as the dylib's `install_name`, which is
useless in a relocatable archive, so the job rewrites it to `@rpath` and
re-signs ad-hoc; arm64 macOS refuses to load a dylib whose signature
`install_name_tool` has invalidated.

Archives are `.tar.xz` on macOS (`xz -9 -e -T1`, single-threaded so the
published checksum is reproducible) and `.zip` on Windows, where it stays the
native convention. tar also preserves the `libchromadec.dylib` symlink that zip
would flatten into a second copy.

Nothing in the workflow makes a release visible on its own. A tag push builds
and uploads the archives as workflow artifacts, then the `publish` job waits on
the `release` environment; only after approval does it create the GitHub release
object (as a **draft**, with generated notes) and attach the archives. So a tag
pushed at a bad commit can be declined at the gate, leaving nothing to clean up
before moving the tag and re-running. Re-runs reuse the tag's existing release
rather than creating a second one.

| Environment | Job | Protection |
|---|---|---|
| `release` | `publish` | Required reviewers go here. Naming the environment in the workflow does not gate anything by itself. |

All Python tooling in CI runs inside a fresh per-job virtual environment.
