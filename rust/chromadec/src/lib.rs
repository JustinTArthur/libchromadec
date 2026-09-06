// SPDX-License-Identifier: GPL-3.0-or-later

#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(rustdoc::broken_intra_doc_links)]

pub use chromadec_sys as sys;

mod cancel;
mod decoder;
mod error;
mod frame;
pub mod log;
mod nn;
#[cfg(feature = "tokio")]
mod stream;
mod types;
mod util;
mod video;

pub use cancel::Cancel;
pub use decoder::{
    Decoder, DropoutDetectMode, DropoutOpts, DropoutOrigin, DropoutSpan, DropoutStats,
};
pub use error::{Error, Result, Status};
pub use frame::{Frame, PlaneView};
pub use nn::{
    CoremlCompute, EngineCacheDir, NnBackend, NnModel, SessionOpts, backend_is_available,
};
#[cfg(feature = "tokio")]
pub use stream::{
    DEFAULT_CHANNEL_DEPTH, DecodedFrame, FrameOrder, FrameStream, StreamOptions,
    decode_frames_stream,
};
pub use types::{
    ChromaIdentMechanism, ChromaIdentReport, ChromaRowComponent, DecoderKind, FrameInfo,
    FrameLayout, OutputInfo, PixelFormat, Plane, PlaneInfo, SampleEncoding, SignalState, VideoInfo,
    VideoParams, VideoStandard,
};
pub use video::Video;

use std::ffi::{CStr, CString};
use std::sync::OnceLock;

/// Stable decoder option names for the typed setters on [`Decoder`],
/// mirroring the `CHD_OPT_*` registry. The comment gives the value type.
pub mod options {
    pub const CHROMA_GAIN: &str = "chroma_gain"; // f64
    pub const CHROMA_PHASE_DEG: &str = "chroma_phase_deg"; // f64
    pub const CHROMA_NR_LEVEL: &str = "chroma_nr_level"; // f64
    pub const LUMA_NR_LEVEL: &str = "luma_nr_level"; // f64
    pub const PADDING_MULTIPLE: &str = "padding_multiple"; // i32
    pub const REVERSE_FIELD_ORDER: &str = "reverse_field_order"; // bool
    pub const PHASE_COMPENSATION: &str = "phase_compensation"; // bool
    pub const COMB_ADAPT_THRESHOLD: &str = "comb_adapt_threshold"; // f64, NTSC 3D
    pub const COMB_CHROMA_WEIGHT: &str = "comb_chroma_weight"; // f64, NTSC 3D
    pub const COMB_SHOW_MAP: &str = "comb_show_map"; // bool, NTSC 3D
    pub const CHROMA_IDENT_MODE: &str = "chroma_ident_mode"; // str, SECAM
    pub const CHROMA_IDENT_MANUAL: &str = "chroma_ident_manual"; // str, SECAM manual mode
    pub const CHROMA_CLICK_NR_LEVEL: &str = "chroma_click_nr_level"; // f64, SECAM
    pub const CHROMA_CLICK_ENV_DIP_DB: &str = "chroma_click_env_dip_db"; // f64, SECAM expert
    pub const CHROMA_CLICK_FREQ_OVERSHOOT: &str = "chroma_click_freq_overshoot"; // f64, SECAM expert
    pub const TRANSFORM_THRESHOLD: &str = "transform_threshold"; // f64
    pub const TRANSFORM_THRESHOLDS_FILE: &str = "transform_thresholds_file"; // str
    pub const FIRST_ACTIVE_SAMPLE: &str = "first_active_sample"; // i32
    pub const LAST_ACTIVE_SAMPLE: &str = "last_active_sample"; // i32
    pub const FIRST_ACTIVE_FRAME_LINE: &str = "first_active_frame_line"; // i32
    pub const LAST_ACTIVE_FRAME_LINE: &str = "last_active_frame_line"; // i32
    pub const NN_INPUT_MAGNITUDE_SCALE: &str = "nn_input_magnitude_scale"; // f64
    pub const NN_CHROMA_BANDPASS: &str = "nn_chroma_bandpass"; // bool
    pub const HVD_CG_ITERATIONS: &str = "hvd_cg_iterations"; // i32, HVD
    pub const HVD_TEMPORAL_STRENGTH: &str = "hvd_temporal_strength"; // f64, HVD 3D
    pub const OUTPUT_FORMAT: &str = "output_format"; // str
    pub const OUTPUT_CLAMP: &str = "output_clamp"; // str
    pub const COLOR_DIFFERENCE_PRECISION: &str = "color_difference_precision"; // str
    pub const BROADCAST_SCALING_PRECISION: &str = "broadcast_scaling_precision"; // str
    pub const OUTPUT_Y4M_HEADERS: &str = "output_y4m_headers"; // bool
    pub const THREAD_COUNT: &str = "thread_count"; // i32
}

/// Library version as (major, minor, patch).
pub fn version() -> (i32, i32, i32) {
    let (mut major, mut minor, mut patch) = (0, 0, 0);
    unsafe { sys::chd_version(&mut major, &mut minor, &mut patch) };
    (major, minor, patch)
}

/// Library version string.
pub fn version_string() -> &'static str {
    unsafe { CStr::from_ptr(sys::chd_version_string()) }
        .to_str()
        .unwrap_or("")
}

/// Whether a feature (e.g. `"nn"`, `"onnxruntime"`, `"coreml"`, `"cuda"`,
/// `"rocm"`, `"fftw"`, `"hvd"`, `"sqlite"`) was compiled into the library.
pub fn has_feature(feature: &str) -> bool {
    let Ok(feature) = CString::new(feature) else {
        return false;
    };
    unsafe { sys::chd_has_feature(feature.as_ptr()) != 0 }
}

/// Calls `chd_init` exactly once, caching the outcome.
pub(crate) fn ensure_init() -> Result<()> {
    static INIT: OnceLock<Result<()>> = OnceLock::new();
    INIT.get_or_init(|| error::check(unsafe { sys::chd_init() }))
        .clone()
}
