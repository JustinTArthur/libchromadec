// SPDX-License-Identifier: GPL-3.0-or-later
//
// Locates libchromadec and generates the FFI bindings.
//
// Discovery order:
//   1. CHROMADEC_LIB_DIR + CHROMADEC_INCLUDE_DIR environment variables
//      (CHROMADEC_STATIC=1 selects the static archive). An explicit override,
//      so it is trusted as-is and the ABI range below is not enforced.
//   2. pkg-config probe for "chromadec" (also honours a Meson devenv /
//      meson-uninstalled PKG_CONFIG_PATH), constrained to the ABI-compatible
//      version range.

use std::env;
use std::path::PathBuf;

/// The range of libchromadec versions these bindings work with, as
/// `[lower, upper)`.
///
/// libchromadec breaks its ABI at the minor before 1.0 and at the major from
/// 1.0 on, which sets the upper bound. Releases inside that range may add
/// functions, so the lower bound is this crate's own version: an older library
/// lacks symbols the safe crate calls. Both bounds derive from this crate's
/// version, which tracks the C library's, so they cannot be left behind.
fn abi_range() -> (String, String) {
    let major: u64 = env::var("CARGO_PKG_VERSION_MAJOR")
        .expect("CARGO_PKG_VERSION_MAJOR is set by cargo")
        .parse()
        .expect("CARGO_PKG_VERSION_MAJOR is numeric");
    let minor: u64 = env::var("CARGO_PKG_VERSION_MINOR")
        .expect("CARGO_PKG_VERSION_MINOR is set by cargo")
        .parse()
        .expect("CARGO_PKG_VERSION_MINOR is numeric");

    let lower = env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION is set by cargo");
    // Drop any pre-release suffix; pkg-config compares the numeric parts.
    let lower = lower.split(['-', '+']).next().unwrap_or(&lower).to_string();

    if major == 0 {
        (lower, format!("0.{}.0", minor + 1))
    } else {
        (lower, format!("{}.0.0", major + 1))
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=CHROMADEC_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CHROMADEC_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CHROMADEC_STATIC");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    let statik = env::var_os("CHROMADEC_STATIC").is_some_and(|v| v != "0" && !v.is_empty());

    let mut include_paths: Vec<PathBuf> = Vec::new();
    if let Ok(lib_dir) = env::var("CHROMADEC_LIB_DIR") {
        let include_dir = env::var("CHROMADEC_INCLUDE_DIR").expect(
            "CHROMADEC_INCLUDE_DIR must be set when CHROMADEC_LIB_DIR overrides pkg-config",
        );
        println!("cargo:rustc-link-search=native={lib_dir}");
        let kind = if statik { "static" } else { "dylib" };
        println!("cargo:rustc-link-lib={kind}=chromadec");
        include_paths.push(PathBuf::from(include_dir));
    } else {
        let (abi_low, abi_high) = abi_range();
        let lib = pkg_config::Config::new()
            .range_version(abi_low.as_str()..abi_high.as_str())
            .statik(statik)
            .probe("chromadec")
            .unwrap_or_else(|e| {
                panic!(
                    "no libchromadec in [{abi_low}, {abi_high}) found via pkg-config: {e}\n\
                     These bindings work only with that range. Install a matching \
                     libchromadec, point PKG_CONFIG_PATH at a build tree's meson-uninstalled \
                     directory, or set CHROMADEC_LIB_DIR and CHROMADEC_INCLUDE_DIR (which skip \
                     the version check)."
                )
            });
        include_paths.extend(lib.include_paths);
    }

    let mut builder = bindgen::Builder::default()
        .header_contents("wrapper.h", "#include <chromadec/chromadec.h>\n")
        .allowlist_item("(chd|CHD|CHROMADEC)_.*")
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .derive_default(true);
    for path in &include_paths {
        builder = builder.clang_arg(format!("-I{}", path.display()));
    }

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    builder
        .generate()
        .expect("bindgen failed to generate bindings for <chromadec/chromadec.h>")
        .write_to_file(&out_path)
        .expect("failed to write bindings.rs");
}
