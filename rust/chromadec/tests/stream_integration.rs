// SPDX-License-Identifier: GPL-3.0-or-later
//
// End-to-end test for the async `FrameStream` adapter, driven by a real
// encode-orc colour-bars fixture. Mirrors the C `test_integration` fixture
// flow: when `CHD_ENCODE_ORC` points at a built encode-orc binary, synthesise
// a 3-frame NTSC fixture and decode it through the stream; otherwise self-skip
// with PASS so a plain `cargo test` stays green without the generator.
//
// The whole file is gated on the `tokio` feature, so it compiles to nothing
// when the feature (and the stream API) is absent.
#![cfg(feature = "tokio")]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;

use chromadec::{
    DecoderKind, Frame, FrameOrder, Plane, StreamOptions, Video, decode_frames_stream, options,
};
use futures_util::StreamExt;

const FRAMES: i64 = 3;

// encode-orc project for a 3-frame NTSC 75% colour-bars composite TBC. The
// `${...}` references are expanded by encode-orc from the environment we set
// on the child process, matching the C integration test.
const NTSC_PROJECT_YAML: &str = r#"name: "chd-rust-it-ntsc-bars"
output:
  filename: "${ENCODE_ORC_OUTPUT_ROOT}/fixture"
  format: "ntsc-composite"
  writer: "tbc"
  metadata_decoder: "encode-orc"
laserdisc:
  mode: "cav"
pipeline:
  preprocessing:
    filters:
      chroma:
        enabled: true
      luma:
        enabled: false
sections:
  - name: "Bars"
    duration: 3
    source:
      type: "yuv422-image"
      file: "${ENCODE_ORC_ASSETS}/720x480/stills/raw/75_BARS.raw"
"#;

/// A generated fixture; deletes its temp dir on drop.
struct Fixture {
    dir: PathBuf,
    tbc: PathBuf,
    sidecar: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Synthesises the NTSC fixture via encode-orc, or returns `None` to skip when
/// `CHD_ENCODE_ORC` is unset. Panics if it is set but unusable, so a broken CI
/// wiring fails loudly rather than silently skipping.
fn generate_ntsc_fixture() -> Option<Fixture> {
    let encode_orc = PathBuf::from(std::env::var_os("CHD_ENCODE_ORC")?);
    assert!(
        encode_orc.exists(),
        "CHD_ENCODE_ORC set but missing: {}",
        encode_orc.display()
    );
    // Assets live in the sibling `assets/` of the binary's grandparent
    // ($repo/build/encode-orc -> $repo/assets), overridable for odd layouts.
    // The subpath in the YAML follows encode-orc's resolution-first asset
    // layout; a checkout predating that reorganisation needs the override
    // pointed at a directory laid out to match.
    let assets = match std::env::var_os("CHD_ENCODE_ORC_ASSETS") {
        Some(a) => PathBuf::from(a),
        None => encode_orc
            .parent()
            .and_then(|p| p.parent())
            .expect("CHD_ENCODE_ORC has no grandparent dir")
            .join("assets"),
    };

    let dir = std::env::temp_dir().join(format!("chromadec_rust_it_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp fixture dir");
    let yaml = dir.join("project.yaml");
    std::fs::write(&yaml, NTSC_PROJECT_YAML).expect("write project.yaml");

    let status = Command::new(&encode_orc)
        .arg(&yaml)
        .args(["--log-level", "warn"])
        .env("ENCODE_ORC_OUTPUT_ROOT", &dir)
        .env("ENCODE_ORC_ASSETS", &assets)
        .status()
        .expect("spawn encode-orc");
    assert!(status.success(), "encode-orc failed: {status}");

    let fixture = Fixture {
        tbc: dir.join("fixture.tbc"),
        sidecar: dir.join("fixture.tbc.db"),
        dir,
    };
    assert!(
        fixture.tbc.exists() && fixture.sidecar.exists(),
        "encode-orc produced no fixture.tbc/.db"
    );
    Some(fixture)
}

/// A content fingerprint over a frame's geometry and all three planes, so the
/// stream-decoded frames can be checked bit-for-bit against the sync path.
fn fingerprint(frame: &Frame) -> u64 {
    let mut h = DefaultHasher::new();
    let info = frame.info().expect("frame info");
    info.width.hash(&mut h);
    info.height.hash(&mut h);
    for plane in [Plane::Y, Plane::Cb, Plane::Cr] {
        let view = frame.plane_u16(plane).expect("plane view");
        for row in view.rows() {
            row.hash(&mut h);
        }
    }
    h.finish()
}

/// Sync reference: `decode_frame` for 0..FRAMES, fingerprint each.
fn sync_fingerprints(fixture: &Fixture) -> Vec<u64> {
    let mut video = Video::open_composite(&fixture.tbc, Some(fixture.sidecar.as_path()), None)
        .expect("open tbc (sync ref)");
    assert!(
        video.info().expect("video info").num_frames >= FRAMES,
        "fixture has fewer than {FRAMES} frames"
    );
    let mut decoder = chromadec::Decoder::new(&mut video, DecoderKind::Ntsc2d).expect("decoder");
    decoder
        .set_option_i32(options::PADDING_MULTIPLE, 1)
        .expect("padding option");
    decoder.commit().expect("commit");
    (0..FRAMES)
        .map(|i| fingerprint(&decoder.decode_frame(i).expect("sync decode_frame")))
        .collect()
}

#[tokio::test]
async fn stream_decodes_match_sync_and_respect_order() {
    let Some(fixture) = generate_ntsc_fixture() else {
        eprintln!("CHD_ENCODE_ORC unset — skipping stream integration test");
        return;
    };

    let expected = sync_fingerprints(&fixture);

    // Indexed: must arrive strictly in requested-index order, bit-identical to
    // the sync reference.
    let video = Video::open_composite(&fixture.tbc, Some(fixture.sidecar.as_path()), None)
        .expect("open tbc (indexed)");
    let mut stream = decode_frames_stream(
        video,
        DecoderKind::Ntsc2d,
        0..FRAMES,
        StreamOptions {
            channel_depth: None,
            order: FrameOrder::Indexed,
        },
        |d| d.set_option_i32(options::PADDING_MULTIPLE, 1),
    )
    .expect("start indexed stream");

    let mut indexed = Vec::new();
    while let Some(decoded) = stream.next().await {
        let frame = decoded.result.expect("indexed frame decoded");
        indexed.push((decoded.index, fingerprint(&frame)));
    }
    stream.finish().await.expect("indexed stream finished");

    let order: Vec<i64> = indexed.iter().map(|(i, _)| *i).collect();
    assert_eq!(
        order,
        (0..FRAMES).collect::<Vec<_>>(),
        "indexed out of order"
    );
    for (index, fp) in &indexed {
        assert_eq!(
            *fp, expected[*index as usize],
            "indexed frame {index} drifted"
        );
    }

    // Completion: same set of frames, same content, any order.
    let video = Video::open_composite(&fixture.tbc, Some(fixture.sidecar.as_path()), None)
        .expect("open tbc (completion)");
    let mut stream = decode_frames_stream(
        video,
        DecoderKind::Ntsc2d,
        0..FRAMES,
        StreamOptions {
            channel_depth: None,
            order: FrameOrder::Completion,
        },
        |d| d.set_option_i32(options::PADDING_MULTIPLE, 1),
    )
    .expect("start completion stream");

    let mut completion = Vec::new();
    while let Some(decoded) = stream.next().await {
        let frame = decoded.result.expect("completion frame decoded");
        completion.push((decoded.index, fingerprint(&frame)));
    }
    stream.finish().await.expect("completion stream finished");

    for (index, fp) in &completion {
        assert_eq!(
            *fp, expected[*index as usize],
            "completion frame {index} drifted"
        );
    }
    let mut seen: Vec<i64> = completion.iter().map(|(i, _)| *i).collect();
    seen.sort_unstable();
    assert_eq!(
        seen,
        (0..FRAMES).collect::<Vec<_>>(),
        "completion missed frames"
    );
}
