# libchromadec documentation

A shared library for decoding 4×fsc-sampled composite analog video (CVBS / TBC)
into component Y′ C′b C′r, E′Y E′Cb E′Cr, R′G′B′, or grayscale output.

## Contents

- [Architecture](architecture.md): current design and module layout
- [Integration guide](integration-guide.md): open a source, decode, handle
  output; includes migrating from ld-chroma-decoder
- [C API reference](api-reference.md): every public `chd_*` function, type,
  and option, with ownership and threading rules
- [Rust bindings](rust-bindings.md): the first-party `chromadec` and
  `chromadec-sys` crates
- [ABI stability](abi-stability.md): versioning and compatibility rules
- [NN model conventions](nn-models.md): model paths, magnitude scales,
  execution provider notes
- [File formats](file-formats.md): TBC vs CVBS reader matrix
- [Changelog](changelog.md): what each release added
- [Attribution](attribution.md): original contributors whose work is
  preserved here
