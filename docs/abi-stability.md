# ABI stability

libchromadec follows [Semantic Versioning](https://semver.org).

## Pre-1.0 (0.x.y)

- ABI may break across **minor** versions: `0.1.x → 0.2.0` is an ABI break.
- ABI is stable within a minor version: `0.1.0 → 0.1.x` is binary-compatible.
- Each release builds with `SameMinorVersion` CMake compatibility, so
  `find_package(chromadec 0.1 CONFIG)` accepts `0.1.x` but not `0.2.0`.

## After 1.0

- ABI stable within a **major** version: `1.x.y → 1.(x+1).0` is
  binary-compatible.
- ABI may break only at major-version boundaries: `1.x.y → 2.0.0` is an ABI
  break.

## SONAME

The shared library's soversion tracks the version at which the ABI is allowed to
break, which is the **minor** before 1.0 and the **major** from 1.0 on. It is
derived from the project version rather than written down separately, so the two
cannot drift:

|Project version|Library soversion|
|---|---|
|`0.1.x`|`0.1`|
|`0.2.x`|`0.2`|
|`1.x.y`|`1`|
|`2.x.y`|`2`|

So an ABI-breaking release always lands under a SONAME the previous release's
binaries will not load, instead of silently satisfying them. On macOS the same
value reaches the install name (`libchromadec.0.1.dylib`) and the Mach-O
compatibility version, which Meson derives from the soversion.

## Symbol export discipline

Only symbols matching `chd_*` are exported.

- ELF (Linux, *BSD): enforced by a version script generated from
  `meson/chromadec.map.in`. Its node (`CHROMADEC_0.1`) carries the same ABI
  version as the soversion.
- macOS: enforced by `meson/chromadec.exports`. Mach-O has no symbol
  versioning, so this is a plain symbol list.
- Windows: a `.def` file is added in a follow-up.

Any new public function must:
1. Have a name matching the `chd_*` pattern.
2. Be declared in a header under `include/chromadec/`.
3. Have its rationale and lifetime rules documented in the header.
4. Pass the CI ABI-checker against the previous release tag.

## Enum representation

Every public enum is declared through `CHD_ENUM` (see the
[API reference](api-reference.md#enums)): a fixed `int32_t` underlying type in
C++ and C23, a plain enum with the identical 4-byte representation in older C.
Size, alignment and calling convention are the same in every language mode, so
adding enumerators to an existing enum is ABI-safe within a minor version.

## Extending option structs

Caller-populated option structs (`chd_nn_session_opts_t`, `chd_dropout_opts_t`)
are passed **by pointer** and carry no `size`/version field, so the library
cannot tell how large the caller's struct is. That constrains how they may grow
without breaking binary compatibility:

- **ABI-safe (allowed within a major version):** add a field by consuming a
  slot from the trailing `reserved[]` array, leaving `sizeof` and every existing
  field offset unchanged. Two rules make this sound:
  1. The new field's **zero value must mean "previous behavior"** — an old
     binary zero-fills `reserved`, so the library reads the new field as `0`.
  2. Callers **must** zero-initialise, via the struct's `*_default()` initializer
     (e.g. `chd_nn_session_opts_default`) or `= {0}`. This is already required.
- **ABI break (major version only):** growing the struct past `reserved`,
  reordering fields, or changing a field's type/size. An old caller binary then
  allocates a smaller struct than a newer library reads through the pointer,
  running past the caller's allocation.

Once a struct's `reserved[]` is exhausted, the next field needs a versioned
struct (`chd_..._opts_v2_t`) or a retrofitted leading `size_t struct_size` the
library validates — both larger changes best timed to a major bump.

New caller-populated option structs should ship a `reserved[]` tail from the
start, as `chd_nn_session_opts_t` and `chd_dropout_opts_t` both do (`reserved[4]`).

## Version probe

```c
#include <chromadec/version.h>

int major, minor, patch;
chd_version(&major, &minor, &patch);
// or:
const char *v = chd_version_string();   // e.g. "0.1.0"
```
