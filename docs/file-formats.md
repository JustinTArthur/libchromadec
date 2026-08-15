# File formats

libchromadec reads two families of composite-video capture: the ld-decode
**`.tbc`** format, and the **CVBS file format**. This page describes both and
maps each to the right opening function. The opening functions are named by
signal layout (composite vs Y/C), not by which family the file came from: an
ld-decode `.tbc` and a CVBS `.cvbs` both open with `chd_video_open_composite`.

## Reader matrix

| On disk | Open with | Sidecar | Notes |
|---|---|---|---|
| `capture.tbc` | [`chd_video_open_composite`](api-reference.md#chd_video_open_composite) | `capture.tbc.db` (SQLite) or `capture.tbc.json` | ld-decode output. Composite, time-base-corrected. |
| `luma.tbc` + `chroma.tbc` | [`chd_video_open_yc`](api-reference.md#chd_video_open_yc) | `luma.tbc.db` + `chroma.tbc.db` | vhs-decode Y/C-separated pair (e.g. S-Video). Decoded per plane, then merged. |
| `capture.cvbs` | [`chd_video_open_composite`](api-reference.md#chd_video_open_composite) | `capture.meta` (SQLite) | CVBS single-file composite. |
| `capture.cvbsy` + `capture.cvbsc` | [`chd_video_open_yc`](api-reference.md#chd_video_open_yc) | `capture.meta` (SQLite) | CVBS dual-file luma/chroma pair. |

In every case the **sample data and the metadata live in separate files**: the
data file is a bare stream of samples with no header, and all the information
needed to interpret it (geometry, levels, standard, field order) comes from a
**metadata sidecar file** next to the data. A `.tbc` without its sidecar cannot
be decoded; for CVBS files a
[`chd_video_params_t`](api-reference.md#chd_video_params_t) override can stand
in for a missing `.meta`.

## The TBC format

A `.tbc` ("time-base-corrected") file is the output of ld-decode's RF
demodulation and time-base correction stage. It has historically been
under-documented, so this section is deliberately thorough.

### On-disk layout

A `.tbc` file is a **raw, headerless stream of unsigned 16-bit little-endian
samples**. There is no magic number, no version field, and no structure marker
anywhere in the file: it is purely sample values, back to back.

The samples are **field-sequential**. The file is a concatenation of fields,
each field being exactly `fieldWidth × fieldHeight` samples, stored row-major
(line by line). Because every field is the same fixed size, the field count is
simply:

```
numberOfFields = fileSizeInBytes / (fieldWidth * fieldHeight * 2)
```

and any field or line can be located by arithmetic alone (the format is fully
random-access). The `× 2` is because each sample is two bytes.

Two consecutive fields make an interlaced frame; which field of the pair comes
first is recorded in the metadata (`isFirstFieldFirst`), not implied by file
order. (The `reverse_field_order` decoder option flips this; see the
[option registry](api-reference.md#option-registry).)

Because the raster is fixed while the analogue field lengths are not whole
numbers of lines, each frame pair carries padding. The fields hold temporally
consecutive lines, split at a whole-line boundary: a 525-line frame is stored
as two 263-line fields, the first carrying frame lines 1-263 (including the
boundary line that contains the half-line) and the second carrying lines
264-525 plus one padding line at its end. A 625-line PAL frame is stored as
two 313-line fields split 313/312, again with the padding at the end of the
second field. (This is the layout ld-chroma-encoder writes; libchromadec's
frame-native CVBS conform reproduces it bit-exactly.)

### Line-locked and subcarrier-locked PAL layouts

For NTSC and PAL-M the story ends there: the line-locked sample grid is also
exactly four samples per subcarrier cycle (910 = 4 × 227.5 cycles/line;
909 = 4 × 227.25), so the lattice is orthogonal and both fields of a pair are
sample-aligned.

PAL is the nuanced one. At true 4×f<sub>SC</sub> a 64 µs PAL line is
**1135 + 4/625 sample periods**, so the lattice repeats per frame, not per
line, and two `.tbc` layouts exist for the same signal:

- **Line-locked** (ld-decode, vhs-decode, encode-orc): every line is produced
  at exactly 1135 samples and the 4-samples-per-frame residue is rounded away.
  The grid is orthogonal, both fields are aligned, and `isSubcarrierLocked` is
  false. The sidecar still stamps `sampleRate = 4 × fSC` (17,734,475 Hz)
  although the physical line-locked rate is 1135 × 15,625 = 17,734,375 Hz,
  about 5.6 ppm lower; the decoder keys its four-samples-per-cycle maths off
  the stamped ratio and absorbs the residual drift with per-line burst
  detection.
- **Subcarrier-locked** (ld-chroma-encoder's scLocked output): the true
  lattice is kept. 0H drifts +4/625 of a sample per line within the uniform
  1135-sample rows, the four leftover samples per frame are parked at the
  start of a dummy 626th line (the padding line above), and the two fields of
  a pair end up two samples apart horizontally. `isSubcarrierLocked` is true,
  and the decoder removes the inter-field offset with a whole-sample shift of
  the second field before chroma separation.

`isSubcarrierLocked` is therefore a **sampling-lattice property**, distinct
from whether the capture's burst phase is stable (the CVBS signal state's
burst lock). A capture can be burst-phase-stable and still line-locked.

### Sample encoding

Each 16-bit word holds a **10-bit sample value left-shifted by 6 bits** (the low
6 bits are zero). In the terms of the CVBS specification below, this is exactly
the `U16_4FSC` sample encoding. Two consequences worth knowing:

- The usable range is `0..1023` scaled into `0..65472` in steps of 64.
- Sampling is **4×f<sub>SC</sub>** (four samples per colour-subcarrier cycle),
  and the signal is **composite**: luma and chroma are interleaved in the same
  samples, exactly as the decoder expects to separate them.

The black, white, and blanking reference levels are *not* fixed by the format;
they are recorded per-capture in the sidecar as `black16bIre`, `white16bIre`,
and `blanking16bIre` (16-bit IRE reference points). Older sidecars predate
`blanking16bIre`; when it is absent the library falls back to `black16bIre`.

### The metadata sidecar

Everything the raw samples omit lives in a sidecar file next to the `.tbc`:

- **`capture.tbc.json`** is the original JSON sidecar.
- **`capture.tbc.db`** is the newer SQLite sidecar (ld-decode migrated to
  SQLite for large captures).

[`chd_video_open_composite`](api-reference.md#chd_video_open_composite) auto-locates either
(`.tbc.db` preferred, then `.tbc.json`) when you pass `NULL` for the sidecar
path, or you can pass an explicit path to either form. The library reads both;
the file extension (case-insensitive) selects the parser.

The sidecar carries two kinds of information:

**Capture-wide video parameters**, including:

| Field | Meaning |
|---|---|
| `fieldWidth`, `fieldHeight` | Field geometry, in samples and lines. |
| `sampleRate`, `fSC` | Sample rate and colour-subcarrier frequency (Hz). |
| `isSubcarrierLocked` | Whether sampling is locked to the subcarrier. |
| `isWidescreen` | 16:9 flag. |
| `colourBurstStart` / `End` | Burst window, in samples. |
| `activeVideoStart` / `End` | Active-line sample range. |
| `firstActiveFrameLine` / `lastActiveFrameLine` | Active frame-line range (the active field-line range is derived from this). |
| `white16bIre`, `black16bIre`, `blanking16bIre` | 16-bit IRE reference levels. |
| `numberOfSequentialFields` | Field count recorded by the capturer. |

**Per-field records** (one per field, keyed by a sequential number), including
the field's `isFirstField` flag, sync confidence, median burst IRE, subcarrier
phase ID, dropout ranges (start/end sample per line), and optional VBI / VITC /
closed-caption / NTSC-specific / PCM-audio sub-records. Dropout ranges from
these records drive [dropout correction](api-reference.md#dropout-correction).

### What the format does not contain

There is no audio in the `.tbc` itself (PCM-audio parameters in the sidecar
describe a separate stream), no colour-decoded output, and no per-frame index
beyond the implicit field arithmetic. The decoder synthesises component
Y'CbCr output from these composite samples at decode time; the `.tbc` is the
raw composite signal, nothing more.

## The CVBS file format

The CVBS file format captures composite video more flexibly than `.tbc`: it
adds dual-file Y/C, several sample encodings, and a structured `.meta` sidecar.
libchromadec reads it through
[`chd_video_open_composite`](api-reference.md#chd_video_open_composite)
and [`chd_video_open_yc`](api-reference.md#chd_video_open_yc).

Rather than restate the specification, this page summarises the shape and defers
to the authoritative source:

!!! info "Authoritative specification"
    The CVBS File Format Specification is published at
    **[decode-orc.github.io/cvbs-file-format-specification](https://decode-orc.github.io/cvbs-file-format-specification)**.
    Treat it as the source of truth; the summary here is orientation only.

### Shape of the format

The format separates three **independent preset axes**, so a capture is
described by one choice on each axis rather than a single monolithic mode:

1. **Video Standard** (e.g. the line standard and colour system).
2. **Sample Encoding** (how samples are quantised and packed).
3. **Signal State** (TBC-locked, TBC-unlocked, raw, and standard vs
   non-standard variants).

It defines **two file layouts**:

- **Composite** (`.cvbs`): a single file carrying the full CVBS signal.
- **Dual-file Y/C** (`.cvbsy` + `.cvbsc`): luma and chroma in separate files, for
  S-Video-style sources where the two are already separated.

Metadata lives in a **`.meta` SQLite sidecar** (the specification pins the
schema version). As with `.tbc`, the data files hold samples and the `.meta`
file holds everything needed to interpret them; pass `NULL` to the open
function to auto-locate `capture.meta`, or give an explicit path.

**SECAM captures.** The specification's preset list has no SECAM entry yet,
so a SECAM capture is stored under the byte-compatible 625/50 `PAL` preset
at a `NONSTANDARD_*` signal state (the PAL-specific frame-sample-count and
burst-lock semantics do not apply there) and declared SECAM at open time
with `chd_video_params_t.standard = CHD_STD_SECAM`. That declaration selects
the PAL lattice for geometry and re-declares the colour standard over it;
with a `.meta` sidecar present, the sidecar's preset must be `PAL`. In a
dual-file Y/C pair the `.cvbsc` file holds the SECAM FM block oscillating about
the centred-chroma zero, the same convention QAM chroma uses.

### Container layouts

The specification addresses CVBS data **by frame**, with exact per-frame
sample totals that are normative whenever TBC has been applied at the standard
4×f<sub>SC</sub> rate. There is no per-field padding in that layout; a frame is
one contiguous run of samples. The library also reads CVBS data blocked the
way `.tbc` files are: fields padded to the fixed raster. The
[`chd_frame_layout_t`](api-reference.md#enumerations) axis names the two:

| Standard | `FRAME_NATIVE` samples/frame | `FIELD_RASTER` samples/frame pair |
|---|---|---|
| PAL | 709,379 | 710,510 (2 × 1135 × 313) |
| NTSC | 477,750 | 478,660 (2 × 910 × 263) |
| PAL-M | 477,225 | 478,134 (2 × 909 × 263) |

Today's producers (ld-decode, vhs-decode, encode-orc) all emit field rasters;
frame-native files are what spec-conformant producers write, since the frame
integrity rules make the exact totals mandatory for the `STANDARD_TBC_*`
signal states.

The layout is resolved at open time, in this order:

1. A `layout` value in the `chd_video_params_t` override (merged even when a
   `.meta` sidecar is present; the schema has no layout column).
2. Signal states without TBC at the standard rate are always `FIELD_RASTER`;
   the frame totals are not normative for them.
3. The `.meta` `number_of_sequential_frames` count, when present: the file
   size divided by it matches exactly one of the two totals.
4. A file-size modulo test against both totals. For NTSC and PAL-M the totals
   are not coprime: 526 frame-native frames and 525 field-raster frames are
   the same byte count, so files at exact multiples of that length need the
   frame count or the override. Ties and truncated files fall back to
   `FIELD_RASTER` with a warning.

`chd_video_get_info` reports the resolved layout, and `samples_per_frame`
gives the standard's native frame total whichever layout the file uses.

Frame-native files are conformed onto the decoder's field raster on the fly,
with no resampling: the two field buffers are consecutive windows of the
continuous native frame (a flat cut), and the part of the second buffer past
the native total is filled with blanking. For NTSC/PAL-M that split is 263
whole lines then 262 plus a padding line. For PAL the cut into uniform
1135-sample rows leaves 0H drifting +4/625 of a sample per line, the four
leftover samples per frame land at the start of the second field's final row,
and the two fields come out two samples apart; that offset is exactly what
the decoder's subcarrier-locked shift removes. The PAL conform is validated
bit-exactly against ld-chroma-encoder's subcarrier-locked TBC output over the
full native extent of every frame.

For a field-raster CVBS file at the standard rate the `.meta` cannot say
whether the sampling was line-locked or subcarrier-locked; the two are
byte-identical on disk. The library defaults to line-locked (matching every
current field-raster producer) and reports `is_subcarrier_locked = 0` even
for burst-locked signal states; set `is_subcarrier_locked` in the override to
mark a subcarrier-locked raster. Frame-native PAL implies a
subcarrier-locked lattice.

The burst-gate and active-video windows follow the rows' horizontal
alignment (where 0H sits within a stored row; see
[Sample numbering](api-reference.md#sample-numbering) for the sample-0
origin). Field rasters are declared: plain ones use the sync-start (0H)
cut, and the `is_subcarrier_locked` override selects the blanking-start
cut, whose burst gate matches what ld-chroma-encoder writes
into its own scLocked sidecars (PAL burst 109-149). The default active-video
crop is the interface standard's digital active line (SMPTE ST 244 / EBU
Tech 3280-E), positioned for the row's alignment. Frame-native files are
measured: the reader locates 0H from the sync edges of fifty early lines
and selects the matching cut. Real hardware captures (Snell and Wilcox
TPG21 references) measure as sync-start, their rows beginning at 0H of
line 1; data flattened from encoder buffers measures as blanking-start;
an unmeasurable signal falls back to sync-start. The active-start cut
that a literal reading of the spec's line numbering would produce has not
been seen in the wild: it is warned about and served with sync-start
windows, since serving it correctly needs a row re-cut (pending upstream
clarification of the stored frame's origin). The selected alignment is
visible through the active window `chd_video_get_info` reports.

NTSC needs one more per-field fact than the sidecar can express: the
RS-170 four-field sequence position (an ld-decode `.tbc` records it as
each field's `fieldPhaseID`; the comb decoder derives every line's chroma
sign from it). CVBS opens measure it from the signal: each field's colour
burst is demodulated over a dozen post-VBI lines with the decoder's own
quadrature convention, and the resulting polarity pair positions the
frame's two fields in the four-field sequence. A field without a
measurable burst (chroma-free content, raw encodings, a capture whose
burst is not locked to the sample lattice) keeps an unknown phase, with a
warning, and NTSC frames built from it may decode with inverted chroma.
PAL and PAL-M decoders detect burst phase per line and need no field-level
measurement.

### Sample encodings libchromadec accepts

The library recognises the CVBS sample encodings below. An ld-decode `.tbc`
reads as `U16_4FSC` (its on-disk layout). The exact bit layouts are specified
in the document above; in brief:

| Encoding | Summary |
|---|---|
| `U16_4FSC` | Unsigned 16-bit, 4×f<sub>SC</sub>. Same packing the `.tbc` format uses. |
| `U10_4FSC` | Unsigned 10-bit, 4×f<sub>SC</sub>. |
| `TPG21_4FSC` | Test-pattern-generator encoding at 4×f<sub>SC</sub> (fixed device offset 508, ×64). |
| `S16_4FSC` | Signed 16-bit, blanking-centred at ×32 scale; the offset follows the standard's blanking level (256 PAL, 240 NTSC/PAL-M). |
| `S16_28M` / `S16_40M` | Signed 16-bit raw composite at 28.6 MHz / 40 MHz sample rates. |

`S16_4FSC` was spelled `S16_FSC` before specification v1.4.0. Both names read,
and both give you `CHD_ENC_CVBS_S16_4FSC`: the rename carried no change to the
encoding, and it did not bump the schema version, so a sidecar cannot say which
spelling it will use.

The `.meta` reader accepts schema `user_version` 7 through 10. Those revisions
differ only in how they carry audio metadata, which the reader does not read;
the `cvbs_file` columns it does read are identical across all four.

When metadata is absent or you need to force parameters, both CVBS open
functions accept a
[`chd_video_params_t`](api-reference.md#chd_video_params_t) override. With a
`.meta` present the override still contributes its `layout`,
`is_subcarrier_locked`, and `is_second_field_first` fields, the three
properties the sidecar schema cannot express.