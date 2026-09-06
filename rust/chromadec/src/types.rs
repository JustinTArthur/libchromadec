// SPDX-License-Identifier: GPL-3.0-or-later

use chromadec_sys as sys;

use crate::error::{Error, Result};

/// Decoder kind (`chd_decoder_kind_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecoderKind {
    Auto,
    Mono,
    Ntsc1d,
    Ntsc2d,
    Ntsc3d,
    Ntsc3dNoAdapt,
    Pal2d,
    Transform2d,
    Transform3d,
    NnTransform3d,
    LdzeugColorCnn,
    LdzeugLumaSep,
    LdzeugLumaSepFrame,
    /// Geometry/metadata only: no chroma decode engines are built and frame
    /// decoding is rejected, but output-info and dropout span/mask queries
    /// work.
    None,
    Secam,
    Hvd2d,
    Hvd3d,
}

impl DecoderKind {
    pub(crate) fn raw(self) -> sys::chd_decoder_kind {
        match self {
            DecoderKind::Auto => sys::chd_decoder_kind::CHD_DEC_AUTO,
            DecoderKind::Mono => sys::chd_decoder_kind::CHD_DEC_MONO,
            DecoderKind::Ntsc1d => sys::chd_decoder_kind::CHD_DEC_NTSC_1D,
            DecoderKind::Ntsc2d => sys::chd_decoder_kind::CHD_DEC_NTSC_2D,
            DecoderKind::Ntsc3d => sys::chd_decoder_kind::CHD_DEC_NTSC_3D,
            DecoderKind::Ntsc3dNoAdapt => sys::chd_decoder_kind::CHD_DEC_NTSC_3D_NO_ADAPT,
            DecoderKind::Pal2d => sys::chd_decoder_kind::CHD_DEC_PAL_2D,
            DecoderKind::Transform2d => sys::chd_decoder_kind::CHD_DEC_TRANSFORM_2D,
            DecoderKind::Transform3d => sys::chd_decoder_kind::CHD_DEC_TRANSFORM_3D,
            DecoderKind::NnTransform3d => sys::chd_decoder_kind::CHD_DEC_NN_TRANSFORM3D,
            DecoderKind::LdzeugColorCnn => sys::chd_decoder_kind::CHD_DEC_LDZEUG_COLOR_CNN,
            DecoderKind::LdzeugLumaSep => sys::chd_decoder_kind::CHD_DEC_LDZEUG_LUMA_SEP,
            DecoderKind::LdzeugLumaSepFrame => sys::chd_decoder_kind::CHD_DEC_LDZEUG_LUMA_SEP_FRAME,
            DecoderKind::None => sys::chd_decoder_kind::CHD_DEC_NONE,
            DecoderKind::Secam => sys::chd_decoder_kind::CHD_DEC_SECAM,
            DecoderKind::Hvd2d => sys::chd_decoder_kind::CHD_DEC_HVD_2D,
            DecoderKind::Hvd3d => sys::chd_decoder_kind::CHD_DEC_HVD_3D,
        }
    }
}

/// Video standard (`chd_video_standard_t`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VideoStandard {
    #[default]
    Unknown,
    Ntsc,
    Pal,
    PalM,
    Secam,
}

impl VideoStandard {
    pub(crate) fn raw(self) -> sys::chd_video_standard {
        match self {
            VideoStandard::Unknown => sys::chd_video_standard::CHD_STD_UNKNOWN,
            VideoStandard::Ntsc => sys::chd_video_standard::CHD_STD_NTSC,
            VideoStandard::Pal => sys::chd_video_standard::CHD_STD_PAL,
            VideoStandard::PalM => sys::chd_video_standard::CHD_STD_PAL_M,
            VideoStandard::Secam => sys::chd_video_standard::CHD_STD_SECAM,
        }
    }

    pub(crate) fn from_raw(raw: sys::chd_video_standard) -> VideoStandard {
        match raw {
            sys::chd_video_standard::CHD_STD_NTSC => VideoStandard::Ntsc,
            sys::chd_video_standard::CHD_STD_PAL => VideoStandard::Pal,
            sys::chd_video_standard::CHD_STD_PAL_M => VideoStandard::PalM,
            sys::chd_video_standard::CHD_STD_SECAM => VideoStandard::Secam,
            _ => VideoStandard::Unknown,
        }
    }
}

/// Sample encoding (`chd_sample_encoding_t`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleEncoding {
    #[default]
    Unknown,
    CvbsU10_4fsc,
    CvbsU16_4fsc,
    CvbsTpg21_4fsc,
    CvbsS16_4fsc,
    RawS16_28m,
    RawS16_40m,
}

impl SampleEncoding {
    pub(crate) fn raw(self) -> sys::chd_sample_encoding {
        match self {
            SampleEncoding::Unknown => sys::chd_sample_encoding::CHD_ENC_UNKNOWN,
            SampleEncoding::CvbsU10_4fsc => sys::chd_sample_encoding::CHD_ENC_CVBS_U10_4FSC,
            SampleEncoding::CvbsU16_4fsc => sys::chd_sample_encoding::CHD_ENC_CVBS_U16_4FSC,
            SampleEncoding::CvbsTpg21_4fsc => sys::chd_sample_encoding::CHD_ENC_CVBS_TPG21_4FSC,
            SampleEncoding::CvbsS16_4fsc => sys::chd_sample_encoding::CHD_ENC_CVBS_S16_4FSC,
            SampleEncoding::RawS16_28m => sys::chd_sample_encoding::CHD_ENC_RAW_S16_28M,
            SampleEncoding::RawS16_40m => sys::chd_sample_encoding::CHD_ENC_RAW_S16_40M,
        }
    }

    pub(crate) fn from_raw(raw: sys::chd_sample_encoding) -> SampleEncoding {
        match raw {
            sys::chd_sample_encoding::CHD_ENC_CVBS_U10_4FSC => SampleEncoding::CvbsU10_4fsc,
            sys::chd_sample_encoding::CHD_ENC_CVBS_U16_4FSC => SampleEncoding::CvbsU16_4fsc,
            sys::chd_sample_encoding::CHD_ENC_CVBS_TPG21_4FSC => SampleEncoding::CvbsTpg21_4fsc,
            sys::chd_sample_encoding::CHD_ENC_CVBS_S16_4FSC => SampleEncoding::CvbsS16_4fsc,
            sys::chd_sample_encoding::CHD_ENC_RAW_S16_28M => SampleEncoding::RawS16_28m,
            sys::chd_sample_encoding::CHD_ENC_RAW_S16_40M => SampleEncoding::RawS16_40m,
            _ => SampleEncoding::Unknown,
        }
    }
}

/// Signal state (`chd_signal_state_t`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SignalState {
    #[default]
    Unknown,
    StandardTbcLocked,
    StandardTbcUnlocked,
    StandardRaw,
    NonstandardTbcLocked,
    NonstandardTbcUnlocked,
    NonstandardRaw,
}

impl SignalState {
    pub(crate) fn raw(self) -> sys::chd_signal_state {
        match self {
            SignalState::Unknown => sys::chd_signal_state::CHD_SIG_UNKNOWN,
            SignalState::StandardTbcLocked => sys::chd_signal_state::CHD_SIG_STANDARD_TBC_LOCKED,
            SignalState::StandardTbcUnlocked => {
                sys::chd_signal_state::CHD_SIG_STANDARD_TBC_UNLOCKED
            }
            SignalState::StandardRaw => sys::chd_signal_state::CHD_SIG_STANDARD_RAW,
            SignalState::NonstandardTbcLocked => {
                sys::chd_signal_state::CHD_SIG_NONSTANDARD_TBC_LOCKED
            }
            SignalState::NonstandardTbcUnlocked => {
                sys::chd_signal_state::CHD_SIG_NONSTANDARD_TBC_UNLOCKED
            }
            SignalState::NonstandardRaw => sys::chd_signal_state::CHD_SIG_NONSTANDARD_RAW,
        }
    }

    pub(crate) fn from_raw(raw: sys::chd_signal_state) -> SignalState {
        match raw {
            sys::chd_signal_state::CHD_SIG_STANDARD_TBC_LOCKED => SignalState::StandardTbcLocked,
            sys::chd_signal_state::CHD_SIG_STANDARD_TBC_UNLOCKED => {
                SignalState::StandardTbcUnlocked
            }
            sys::chd_signal_state::CHD_SIG_STANDARD_RAW => SignalState::StandardRaw,
            sys::chd_signal_state::CHD_SIG_NONSTANDARD_TBC_LOCKED => {
                SignalState::NonstandardTbcLocked
            }
            sys::chd_signal_state::CHD_SIG_NONSTANDARD_TBC_UNLOCKED => {
                SignalState::NonstandardTbcUnlocked
            }
            sys::chd_signal_state::CHD_SIG_NONSTANDARD_RAW => SignalState::NonstandardRaw,
            _ => SignalState::Unknown,
        }
    }
}

/// Container addressing (`chd_frame_layout_t`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FrameLayout {
    #[default]
    Unknown,
    FieldRaster,
    FrameNative,
}

impl FrameLayout {
    pub(crate) fn raw(self) -> sys::chd_frame_layout {
        match self {
            FrameLayout::Unknown => sys::chd_frame_layout::CHD_FRAME_LAYOUT_UNKNOWN,
            FrameLayout::FieldRaster => sys::chd_frame_layout::CHD_FRAME_LAYOUT_FIELD_RASTER,
            FrameLayout::FrameNative => sys::chd_frame_layout::CHD_FRAME_LAYOUT_FRAME_NATIVE,
        }
    }

    pub(crate) fn from_raw(raw: sys::chd_frame_layout) -> FrameLayout {
        match raw {
            sys::chd_frame_layout::CHD_FRAME_LAYOUT_FIELD_RASTER => FrameLayout::FieldRaster,
            sys::chd_frame_layout::CHD_FRAME_LAYOUT_FRAME_NATIVE => FrameLayout::FrameNative,
            _ => FrameLayout::Unknown,
        }
    }
}

/// Plane selector (`chd_plane_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Plane {
    Y,
    Cb,
    Cr,
    R,
    G,
    B,
}

impl Plane {
    pub(crate) fn raw(self) -> sys::chd_plane {
        match self {
            Plane::Y => sys::chd_plane::CHD_PLANE_Y,
            Plane::Cb => sys::chd_plane::CHD_PLANE_CB,
            Plane::Cr => sys::chd_plane::CHD_PLANE_CR,
            Plane::R => sys::chd_plane::CHD_PLANE_R,
            Plane::G => sys::chd_plane::CHD_PLANE_G,
            Plane::B => sys::chd_plane::CHD_PLANE_B,
        }
    }
}

/// Output pixel format (`chd_pixel_format_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PixelFormat {
    Yuv444p16,
    Yuv444ps,
    Rgb48,
    Rgbs,
    Gray16,
    Grays,
    Yuv440p16,
    Yuv440ps,
}

impl PixelFormat {
    pub(crate) fn from_raw(raw: sys::chd_pixel_format) -> Result<PixelFormat> {
        Ok(match raw {
            sys::chd_pixel_format::CHD_PIXEL_YUV444P16 => PixelFormat::Yuv444p16,
            sys::chd_pixel_format::CHD_PIXEL_YUV444PS => PixelFormat::Yuv444ps,
            sys::chd_pixel_format::CHD_PIXEL_RGB48 => PixelFormat::Rgb48,
            sys::chd_pixel_format::CHD_PIXEL_RGBS => PixelFormat::Rgbs,
            sys::chd_pixel_format::CHD_PIXEL_GRAY16 => PixelFormat::Gray16,
            sys::chd_pixel_format::CHD_PIXEL_GRAYS => PixelFormat::Grays,
            sys::chd_pixel_format::CHD_PIXEL_YUV440P16 => PixelFormat::Yuv440p16,
            sys::chd_pixel_format::CHD_PIXEL_YUV440PS => PixelFormat::Yuv440ps,
            other => {
                return Err(Error::internal(&format!(
                    "unrecognized chd_pixel_format value {}",
                    other.0
                )));
            }
        })
    }

    /// True for the float formats (`f32` planes); false for the 16-bit
    /// integer formats.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            PixelFormat::Yuv444ps | PixelFormat::Rgbs | PixelFormat::Grays | PixelFormat::Yuv440ps
        )
    }
}

/// Per-plane geometry (`chd_plane_info_t`). Full-height planes report the
/// frame dimensions and `first_frame_row` 0; 4:4:0 chroma planes report
/// their subsampled height and the output frame row of their first row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlaneInfo {
    pub width: i32,
    pub height: i32,
    pub first_frame_row: i32,
}

/// Colour-difference component of one frame row of a line-sequential
/// (SECAM) decode (`chd_chroma_row_component_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChromaRowComponent {
    Db,
    Dr,
}

impl ChromaRowComponent {
    pub(crate) fn from_raw(raw: sys::chd_chroma_row_component) -> Result<ChromaRowComponent> {
        Ok(match raw {
            sys::chd_chroma_row_component::CHD_CHROMA_ROW_DB => ChromaRowComponent::Db,
            sys::chd_chroma_row_component::CHD_CHROMA_ROW_DR => ChromaRowComponent::Dr,
            other => {
                return Err(Error::internal(&format!(
                    "unrecognized chd_chroma_row_component value {}",
                    other.0
                )));
            }
        })
    }
}

/// How per-line Db/Dr identity was decided
/// (`chd_chroma_ident_mechanism_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChromaIdentMechanism {
    Porch,
    Bottles,
    Content,
    Manual,
}

/// Per-frame Db/Dr ident summary (`chd_chroma_ident_report_t`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChromaIdentReport {
    pub mechanism: ChromaIdentMechanism,
    pub confidence: f64,
    pub field_confidence: [f64; 2],
    pub first_row_component: ChromaRowComponent,
}

/// Source parameter overrides for the CVBS open functions
/// (`chd_video_params_t`). `standard`, `encoding`, and `signal_state` are
/// required when no sidecar is found; `layout`, `is_subcarrier_locked`, and
/// `is_second_field_first` merge field-wise over sidecar metadata.
#[derive(Clone, Copy, Debug, Default)]
pub struct VideoParams {
    pub standard: VideoStandard,
    pub encoding: SampleEncoding,
    pub signal_state: SignalState,
    pub layout: FrameLayout,
    pub is_subcarrier_locked: bool,
    pub is_second_field_first: bool,
}

impl VideoParams {
    pub(crate) fn raw(&self) -> sys::chd_video_params {
        sys::chd_video_params {
            standard: self.standard.raw(),
            encoding: self.encoding.raw(),
            signal_state: self.signal_state.raw(),
            layout: self.layout.raw(),
            is_subcarrier_locked: self.is_subcarrier_locked as _,
            is_second_field_first: self.is_second_field_first as _,
        }
    }
}

/// Source description (`chd_video_info_t`).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct VideoInfo {
    pub standard: VideoStandard,
    pub encoding: SampleEncoding,
    pub signal_state: SignalState,
    pub layout: FrameLayout,
    pub field_width: i32,
    pub field_height: i32,
    pub samples_per_frame: i32,
    pub sample_rate_hz: f64,
    pub fsc_hz: f64,
    /// Inclusive, 0-indexed active-sample bounds within a stored row.
    pub first_active_sample: i32,
    pub last_active_sample: i32,
    /// Inclusive, 0-indexed active frame-line bounds: lines of the woven
    /// interlaced frame (field 1 on the even lines, field 2 on the odd). Use
    /// `Video::frame_line_to_signal_line` / `Video::signal_line_to_frame_line`
    /// to convert to and from the analogue standards' field-sequential signal
    /// line numbers.
    pub first_active_frame_line: i32,
    pub last_active_frame_line: i32,
    pub black_16b_ire: i32,
    pub white_16b_ire: i32,
    pub blanking_16b_ire: i32,
    pub num_frames: i64,
    pub is_widescreen: bool,
    pub is_subcarrier_locked: bool,
    pub is_first_field_first: bool,
}

impl VideoInfo {
    pub(crate) fn from_raw(raw: &sys::chd_video_info) -> VideoInfo {
        VideoInfo {
            standard: VideoStandard::from_raw(raw.standard),
            encoding: SampleEncoding::from_raw(raw.encoding),
            signal_state: SignalState::from_raw(raw.signal_state),
            layout: FrameLayout::from_raw(raw.layout),
            field_width: raw.field_width,
            field_height: raw.field_height,
            samples_per_frame: raw.samples_per_frame,
            sample_rate_hz: raw.sample_rate_hz,
            fsc_hz: raw.fsc_hz,
            first_active_sample: raw.first_active_sample,
            last_active_sample: raw.last_active_sample,
            first_active_frame_line: raw.first_active_frame_line,
            last_active_frame_line: raw.last_active_frame_line,
            black_16b_ire: raw.black_16b_ire,
            white_16b_ire: raw.white_16b_ire,
            blanking_16b_ire: raw.blanking_16b_ire,
            num_frames: raw.num_frames,
            is_widescreen: raw.is_widescreen != 0,
            is_subcarrier_locked: raw.is_subcarrier_locked != 0,
            is_first_field_first: raw.is_first_field_first != 0,
        }
    }
}

/// Committed output framing (`chd_output_info_t`).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct OutputInfo {
    pub format: PixelFormat,
    pub width: i32,
    pub height: i32,
    pub num_planes: i32,
    pub num_frames: i64,
}

/// Decoded frame description (`chd_frame_info_t`).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct FrameInfo {
    pub format: PixelFormat,
    pub width: i32,
    pub height: i32,
    pub num_planes: i32,
    pub frame_index: i64,
}
