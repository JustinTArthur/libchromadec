# Rust bindings

libchromadec ships first-party Rust bindings as a two-crate workspace under
`rust/` in the source tree:

- **`chromadec-sys`** — raw, `unsafe` FFI declarations generated from the C
  headers with [bindgen](https://github.com/rust-lang/rust-bindgen). One-to-one
  with the C API, no policy of its own. Its build script locates the native
  library and emits the link directives.
- **`chromadec`** — the safe, idiomatic wrapper. Owning handle types with
  `Drop` impls, `Result`-based error handling, slices instead of
  pointer-plus-length pairs, and Rust enums mirroring the C enums.

Rust has no stable ABI, so these are consumed as **source crates** layered on
the same C ABI everything else links against — the native library artifact is
unchanged. Almost all consumers want the safe `chromadec` crate; reach for
`chromadec-sys` directly only when you need a C entry point the wrapper does not
expose yet.

## Adding the dependency

Depend on the crates by path or git.

```toml
[dependencies]
chromadec = { git = "https://github.com/JustinTArthur/libchromadec", branch = "main" }
```

`chromadec-sys` is pulled in transitively; you do not name it unless you call
raw FFI.

## Finding the native library

`chromadec-sys`'s build script locates libchromadec in this order:

1. **`CHROMADEC_LIB_DIR` + `CHROMADEC_INCLUDE_DIR`** — explicit override. Set
   both. `CHROMADEC_STATIC=1` links the static archive instead of the shared
   library. Being an explicit override it is trusted as-is, so the version
   check below does not apply: you are responsible for pointing it at a
   compatible library.
2. **pkg-config** — probes for `chromadec`. Point `PKG_CONFIG_PATH` at an
   installed prefix's `lib/pkgconfig`, or at a Meson build tree's
   `meson-uninstalled` directory to build against an in-tree library without
   installing.

The pkg-config probe is constrained to the libchromadec versions the bindings
are ABI-compatible with, and fails rather than binding to a library outside it.
The range follows the C library's ABI boundary (see
[ABI stability](abi-stability.md)): the minor before 1.0, the major from 1.0 on.
So `chromadec-sys` 0.1.x accepts libchromadec `>= 0.1.0, < 0.2.0`, and a 1.x
crate would accept `>= 1.0.0, < 2.0.0`. The bindings are generated from the
headers of whatever the probe finds, so a library outside the range could
otherwise be bound with mismatched struct layouts or enum numbering.

For a shared library that is not on the system's default search path at
runtime, set the loader path (`LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH`
on macOS, `PATH` on Windows) or link statically.

```sh
# Against an installed prefix
cargo build

# Against an in-tree Meson build, no install
PKG_CONFIG_PATH=/path/to/libchromadec/build/meson-uninstalled \
  DYLD_LIBRARY_PATH=/path/to/libchromadec/build/src \
  cargo run --example video_info -- capture.tbc
```

## A minimal decode

```rust
use chromadec::{Decoder, DecoderKind, Plane, Video};

fn main() -> chromadec::Result<()> {
    let mut video = Video::open_composite("capture.tbc", None, None)?;
    let mut decoder = Decoder::new(&mut video, DecoderKind::Ntsc3d)?;
    decoder.set_option_f64(chromadec::options::CHROMA_GAIN, 1.0)?;
    decoder.commit()?;

    let frame = decoder.decode_frame(0)?;
    let y = frame.plane_u16(Plane::Y)?;
    for row in y.rows() {
        // row: &[u16], `y.width()` samples wide
        let _ = row;
    }
    Ok(())
}
```

`chd_init` is called automatically the first time you open a source or load a
model, so there is no explicit init step.

## How the C API maps to Rust

| C API                  | Rust                                           |
|------------------------|------------------------------------------------|
| `chd_video_t *`        | `Video` (frees on drop)                        |
| `chd_decoder_t *`      | `Decoder<'v>` (borrows the `Video`)            |
| `chd_frame_t *`        | `Frame` (frees on drop)                        |
| `chd_nn_model_t *`     | `NnModel` (frees on drop)                      |
| `chd_cancel_t *`       | `Cancel` (frees on drop)                       |
| `chd_status_t` return  | `Result<T, Error>`                             |
| `chd_last_error()`     | captured into `Error::message`                 |
| `chd_set_log_callback` | `chromadec::log::set_callback`                 |
| `CHD_OPT_*` names      | `chromadec::options::*`                        |
| `chd_frame_get_plane*` | `Frame::plane_u16` / `plane_f32` → `PlaneView` |

A `Decoder` mutably borrows its `Video` for its whole lifetime, so the
borrow checker enforces the C contract that the video outlives the decoder.
`PlaneView` borrows its `Frame`, so plane data cannot outlive the frame that
owns it — the zero-copy borrow is sound without a copy.

### Errors

Every fallible call returns `chromadec::Result<T>`. An `Error` carries the
[`Status`] code and the thread-local detail string the library recorded
(`chd_last_error`), captured on the thread the failing call ran on. `Error`
implements `std::error::Error`, so it composes with `?` and error libraries.

The C caveat about reading the detail string only after a non-`CHD_OK` status
does not reach Rust: an `Error` is only constructed on the failure path, so a
detail left behind by a failure the library handled itself cannot end up in an
`Ok`.

### Decoding paths

| You want                                         | Use                                                            |
|--------------------------------------------------|----------------------------------------------------------------|
| Random access, or simple sequential decode       | `decode_frame(i)` — synchronous, no worker pool                |
| Max throughput over a batch, blocking the caller | `decode_frames` — parallel, completion-order callback          |
| Async / Tokio integration                        | `decode_frames_stream` — `FrameOrder::Completion` or `Indexed` |

### Parallel decode

`Decoder::decode_frames` wraps `chd_decode_frames_async`. It takes a closure
invoked from worker threads as each frame completes (in completion order, not
index order), and blocks until the batch finishes. The closure must be `Sync`;
a panic inside it is caught, the batch is unwound, and the panic is re-raised
on the calling thread once the C call returns — it never unwinds across the FFI
boundary. Pass a [`Cancel`] to make still-queued frames report
`Status::Cancelled`.

Need the results in index order? `decode_frames` only returns once the whole
batch is done, so no special call is required — index a `Vec<Option<Frame>>`
by position from the callback, or collect the items and sort by index. For
ordered *streaming* without holding the whole batch in memory, use the async
stream's `FrameOrder::Indexed` instead.

### Async stream

With the `tokio` feature, `chromadec::decode_frames_stream` runs that same
batch decode on a `spawn_blocking` worker and yields each frame through a
bounded channel, returning a `FrameStream` that implements
`futures_core::Stream` (and has an inherent `recv().await`). It is a runtime
consumer, not a provider: call it from within your own Tokio runtime (either
flavor).

Ownership of the `Video` moves into the worker, which builds, configures
(via the closure), commits, and runs the decoder there — so the non-`'static`
decoder borrow never has to cross threads. Behaviour is set by a
`StreamOptions`:

- `channel_depth` bounds how many decoded frames may sit buffered before the
  worker blocks, applying backpressure to a slow consumer; `None` uses
  `DEFAULT_CHANNEL_DEPTH` (8, which caps channel-held memory in the tens of MB
  for SD frames), `Some(0)` is rejected.
- `order` selects delivery: `FrameOrder::Completion` (default) yields each
  frame as soon as it finishes — maximum throughput, any order, the same
  contract as `decode_frames`. `FrameOrder::Indexed` yields them in
  requested-index order instead: a worker that finishes ahead of its turn
  parks until that frame is emitted, so at most the decoder's worker count of
  frames are held back — the worker count bounds the reorder window, no extra
  knob. The trade-off is head-of-line blocking: a slow early frame stalls
  delivery of later frames already decoded.

Cancel externally with `FrameStream::cancel` / `cancel_handle`; dropping the
stream requests cancel so an in-flight decode stops promptly. Call
`finish().await` afterwards for the overall decode status, including a setup
error that produced an empty stream.

## Diagnostics

The C library is silent until a sink is installed (see the [C API
reference](api-reference.md#diagnostics)); `chromadec::log` is the safe front
end for that. `set_callback` takes any `Fn(Level, &str) + Send + Sync +
'static` closure, boxes it, and keeps it alive for exactly as long as the
library can call it:

```rust
use chromadec::log::{self, Level, LevelFilter};

log::set_filter(LevelFilter::Debug);
log::set_callback(|d| eprintln!("[chromadec/{}] {}", d.level, d.message));
```

The closure receives a `&Diagnostic<'_>` with `level`, `message`, and
`returned`. The last is the one to know about: a failure coming back as an
`Error` is announced on the sink too, carrying the same text, so a program that
reports both says it twice. Skip the marked ones if you already surface the
`Err`:

```rust
# use chromadec::log;
log::set_callback(|d| {
    if d.returned {
        return; // the `Err` carries this
    }
    eprintln!("[chromadec/{}] {}", d.level, d.message);
});
```

Never filter on `Level::Error` alone instead: an unmarked error is one with no
return path, and dropping those loses it entirely. `Diagnostic` is
`#[non_exhaustive]`, so read the fields you want rather than destructuring it
whole.

`Level` and `LevelFilter` mirror the `log` crate's own `Level` /
`LevelFilter` split, so the threshold can express "off" without polluting the
severity a message arrives with. A panic inside your closure is caught rather
than unwound into C, which would be undefined. `log::clear_callback` detaches
on the C side first and drops the closure only once the library guarantees it is
no longer running, so a sink capturing owned state is sound.

The rest of the module: `log::to_stderr()` installs the library's built-in
stderr sink, replacing any closure; `log::filter()` reads the threshold back;
and `log::is_enabled(level)` reports whether a message at that level would
reach a sink, for guarding diagnostic work on your own side. `Level` has no
"off" variant, so the C wart of asking whether an off-level message would be
delivered cannot be expressed here.

The sink is process-wide rather than per-`Video`, so in a plugin loaded
alongside other libchromadec consumers the last `set_callback` wins.

With the `log` feature, `log::forward_to_log_crate()` wires the sink into the
[`log`](https://docs.rs/log) facade under the `"chromadec"` target and sets the
library threshold from `log::max_level()`. Call it after your logger is
installed:

```toml
[dependencies]
chromadec = { version = "0.2", features = ["log"] }
```

## Neural decoders

`NnModel::load_from_file` / `load_from_memory` mirror the C loaders;
`SessionOpts` mirrors `chd_nn_session_opts_t` with `Default` matching
`chd_nn_session_opts_default()`. Bind a model with `Decoder::set_nn_model`
before `commit`. The decoder borrows the model and does not take ownership, so
keep the `NnModel` alive for the bound decoder's lifetime. Backend availability
in the current build/host is queryable with `chromadec::backend_is_available`.

[`Status`]: https://docs.rs/chromadec/latest/chromadec/enum.Status.html
[`Cancel`]: https://docs.rs/chromadec/latest/chromadec/struct.Cancel.html
