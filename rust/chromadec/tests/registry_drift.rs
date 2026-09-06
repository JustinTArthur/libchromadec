// SPDX-License-Identifier: GPL-3.0-or-later
//
// Guards the hand-maintained mirrors of the C registries against drift: every
// CHD_OPT_* in <chromadec/decoder.h> needs a value in `chromadec::options`,
// and every CHD_DEC_* a `DecoderKind` arm. The check is textual so it needs no
// API surface of its own, and self-skips when the headers are not alongside
// (the crate built from a published tarball rather than the repo).

const OPTIONS_SRC: &str = include_str!("../src/lib.rs");
const KINDS_SRC: &str = include_str!("../src/types.rs");

fn decoder_header() -> Option<String> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../include/chromadec/decoder.h");
    std::fs::read_to_string(path).ok()
}

#[test]
fn every_c_option_is_mirrored() {
    let Some(header) = decoder_header() else {
        eprintln!("PASS: decoder.h not alongside the crate, skipping");
        return;
    };
    let missing: Vec<&str> = header
        .lines()
        .filter(|line| line.starts_with("#define CHD_OPT_"))
        .filter_map(|line| line.split('"').nth(1))
        .filter(|value| !OPTIONS_SRC.contains(&format!("= \"{value}\";")))
        .collect();
    assert!(
        missing.is_empty(),
        "chromadec::options is missing {missing:?}"
    );
}

#[test]
fn every_c_decoder_kind_is_mirrored() {
    let Some(header) = decoder_header() else {
        eprintln!("PASS: decoder.h not alongside the crate, skipping");
        return;
    };
    let missing: Vec<&str> = header
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("CHD_DEC_"))
        .filter_map(|line| line.split([' ', '=']).next())
        .filter(|name| !KINDS_SRC.contains(&format!("{name},")))
        .collect();
    assert!(missing.is_empty(), "DecoderKind is missing {missing:?}");
}
