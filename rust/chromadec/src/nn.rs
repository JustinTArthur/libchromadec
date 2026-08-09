// SPDX-License-Identifier: GPL-3.0-or-later

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};

use chromadec_sys as sys;

use crate::ensure_init;
use crate::error::{Result, check};
use crate::util::{non_null, path_to_cstring};

/// NN inference backend (`chd_nn_backend_t`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NnBackend {
    /// Best across all backends; inferred from the model artifact.
    #[default]
    Auto,
    OrtAuto,
    OrtCpu,
    OrtCuda,
    OrtTensorrt,
    OrtCoreml,
    OrtDirectml,
    OrtMigraphx,
    /// Native CoreML `.mlpackage` via MLModel.
    Coreml,
    /// A backend value this crate doesn't know about (newer library).
    Other(u32),
}

impl NnBackend {
    pub(crate) fn raw(self) -> sys::chd_nn_backend {
        match self {
            NnBackend::Auto => sys::chd_nn_backend::CHD_NN_BACKEND_AUTO,
            NnBackend::OrtAuto => sys::chd_nn_backend::CHD_NN_ORT_AUTO,
            NnBackend::OrtCpu => sys::chd_nn_backend::CHD_NN_ORT_CPU,
            NnBackend::OrtCuda => sys::chd_nn_backend::CHD_NN_ORT_CUDA,
            NnBackend::OrtTensorrt => sys::chd_nn_backend::CHD_NN_ORT_TENSORRT,
            NnBackend::OrtCoreml => sys::chd_nn_backend::CHD_NN_ORT_COREML,
            NnBackend::OrtDirectml => sys::chd_nn_backend::CHD_NN_ORT_DIRECTML,
            NnBackend::OrtMigraphx => sys::chd_nn_backend::CHD_NN_ORT_MIGRAPHX,
            NnBackend::Coreml => sys::chd_nn_backend::CHD_NN_COREML,
            NnBackend::Other(v) => sys::chd_nn_backend(v),
        }
    }

    pub(crate) fn from_raw(raw: sys::chd_nn_backend) -> NnBackend {
        match raw {
            sys::chd_nn_backend::CHD_NN_BACKEND_AUTO => NnBackend::Auto,
            sys::chd_nn_backend::CHD_NN_ORT_AUTO => NnBackend::OrtAuto,
            sys::chd_nn_backend::CHD_NN_ORT_CPU => NnBackend::OrtCpu,
            sys::chd_nn_backend::CHD_NN_ORT_CUDA => NnBackend::OrtCuda,
            sys::chd_nn_backend::CHD_NN_ORT_TENSORRT => NnBackend::OrtTensorrt,
            sys::chd_nn_backend::CHD_NN_ORT_COREML => NnBackend::OrtCoreml,
            sys::chd_nn_backend::CHD_NN_ORT_DIRECTML => NnBackend::OrtDirectml,
            sys::chd_nn_backend::CHD_NN_ORT_MIGRAPHX => NnBackend::OrtMigraphx,
            sys::chd_nn_backend::CHD_NN_COREML => NnBackend::Coreml,
            other => NnBackend::Other(other.0),
        }
    }
}

/// Whether a backend is usable in this build/host (`chd_nn_backend_is_available`).
pub fn backend_is_available(backend: NnBackend) -> bool {
    unsafe { sys::chd_nn_backend_is_available(backend.raw()) != 0 }
}

/// Compute units for the native CoreML backend (`chd_nn_coreml_compute_t`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CoremlCompute {
    /// CPU + GPU, no ANE.
    #[default]
    CpuAndGpu,
    /// CPU + GPU + Apple Neural Engine.
    All,
    CpuOnly,
}

impl CoremlCompute {
    fn raw(self) -> sys::chd_nn_coreml_compute {
        match self {
            CoremlCompute::CpuAndGpu => sys::chd_nn_coreml_compute::CHD_NN_COREML_CPU_AND_GPU,
            CoremlCompute::All => sys::chd_nn_coreml_compute::CHD_NN_COREML_ALL,
            CoremlCompute::CpuOnly => sys::chd_nn_coreml_compute::CHD_NN_COREML_CPU_ONLY,
        }
    }
}

/// Inference compute precision (`chd_nn_compute_precision_t`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ComputePrecision {
    #[default]
    Fp32,
    /// Permit a backend with a reduced-precision engine mode (TensorRT) to
    /// build a mixed fp16/fp32 engine. Backends without one ignore it. Only
    /// enable for weights whose input contract keeps fp16 in range.
    Fp16Allowed,
}

impl ComputePrecision {
    fn raw(self) -> sys::chd_nn_compute_precision {
        match self {
            ComputePrecision::Fp32 => sys::chd_nn_compute_precision::CHD_NN_PRECISION_FP32,
            ComputePrecision::Fp16Allowed => {
                sys::chd_nn_compute_precision::CHD_NN_PRECISION_FP16_ALLOWED
            }
        }
    }
}

/// Compiled-engine cache directory policy (`engine_cache_dir`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum EngineCacheDir {
    /// Library picks a per-user cache directory.
    #[default]
    Auto,
    /// No caching; recompile on every load.
    Disabled,
    Path(PathBuf),
}

/// NN session options (`chd_nn_session_opts_t`). `Default` mirrors
/// `chd_nn_session_opts_default()`.
#[derive(Clone, Debug)]
pub struct SessionOpts {
    pub backend: NnBackend,
    pub device_id: i32,
    pub enable_graph_optim: bool,
    pub enable_mem_pattern: bool,
    pub inter_op_threads: i32,
    pub intra_op_threads: i32,
    pub coreml_compute: CoremlCompute,
    pub engine_cache_dir: EngineCacheDir,
    pub precision: ComputePrecision,
}

impl Default for SessionOpts {
    fn default() -> SessionOpts {
        let mut raw = sys::chd_nn_session_opts::default();
        unsafe { sys::chd_nn_session_opts_default(&mut raw) };
        SessionOpts {
            backend: NnBackend::from_raw(raw.backend),
            device_id: raw.device_id,
            enable_graph_optim: raw.enable_graph_optim != 0,
            enable_mem_pattern: raw.enable_mem_pattern != 0,
            inter_op_threads: raw.inter_op_threads,
            intra_op_threads: raw.intra_op_threads,
            coreml_compute: CoremlCompute::default(),
            engine_cache_dir: EngineCacheDir::Auto,
            precision: ComputePrecision::default(),
        }
    }
}

impl SessionOpts {
    /// Returns the raw struct plus the CString backing `engine_cache_dir`,
    /// which must stay alive for the duration of the load call.
    fn raw(&self) -> Result<(sys::chd_nn_session_opts, Option<CString>)> {
        let mut raw = sys::chd_nn_session_opts::default();
        unsafe { sys::chd_nn_session_opts_default(&mut raw) };
        raw.backend = self.backend.raw();
        raw.device_id = self.device_id;
        raw.enable_graph_optim = self.enable_graph_optim as _;
        raw.enable_mem_pattern = self.enable_mem_pattern as _;
        raw.inter_op_threads = self.inter_op_threads;
        raw.intra_op_threads = self.intra_op_threads;
        raw.coreml_compute = self.coreml_compute.raw();
        raw.precision = self.precision.raw();
        // An owned CString backs an explicit cache path and must outlive the
        // returned struct; Auto (null) and Disabled (empty literal) own nothing.
        let cache = match &self.engine_cache_dir {
            EngineCacheDir::Auto => {
                raw.engine_cache_dir = ptr::null();
                None
            }
            EngineCacheDir::Disabled => {
                raw.engine_cache_dir = c"".as_ptr();
                None
            }
            EngineCacheDir::Path(p) => {
                let path = path_to_cstring(p)?;
                raw.engine_cache_dir = path.as_ptr();
                Some(path)
            }
        };
        Ok((raw, cache))
    }
}

/// A loaded NN model (`chd_nn_model_t`). A decoder that binds it with
/// [`crate::Decoder::set_nn_model`] borrows it without taking ownership, so the
/// handle must outlive every decoder it is bound to.
#[derive(Debug)]
pub struct NnModel {
    raw: NonNull<sys::chd_nn_model>,
}

unsafe impl Send for NnModel {}

impl NnModel {
    /// Loads a model artifact from disk (`chd_nn_model_load_from_file`):
    /// `.onnx` via ONNX Runtime, `.mlpackage`/`.mlmodelc` via native CoreML
    /// when the backend is `Auto`.
    pub fn load_from_file(
        model_path: impl AsRef<Path>,
        opts: Option<&SessionOpts>,
    ) -> Result<NnModel> {
        ensure_init()?;
        let path = path_to_cstring(model_path.as_ref())?;
        let raw_opts = opts.map(SessionOpts::raw).transpose()?;
        let mut raw = ptr::null_mut();
        unsafe {
            check(sys::chd_nn_model_load_from_file(
                path.as_ptr(),
                raw_opts.as_ref().map_or(ptr::null(), |(o, _)| o),
                &mut raw,
            ))?;
        }
        Ok(NnModel {
            raw: non_null(raw)?,
        })
    }

    /// Loads a serialized ONNX model from memory
    /// (`chd_nn_model_load_from_memory`). The bytes are consumed during the
    /// call. Backends that can't ingest a buffer (native CoreML) reject this.
    pub fn load_from_memory(model_data: &[u8], opts: Option<&SessionOpts>) -> Result<NnModel> {
        ensure_init()?;
        let raw_opts = opts.map(SessionOpts::raw).transpose()?;
        let mut raw = ptr::null_mut();
        unsafe {
            check(sys::chd_nn_model_load_from_memory(
                model_data.as_ptr().cast(),
                model_data.len(),
                raw_opts.as_ref().map_or(ptr::null(), |(o, _)| o),
                &mut raw,
            ))?;
        }
        Ok(NnModel {
            raw: non_null(raw)?,
        })
    }

    /// The backend actually in use after load
    /// (`chd_nn_model_get_active_backend`).
    pub fn active_backend(&self) -> Result<NnBackend> {
        let mut raw = sys::chd_nn_backend::CHD_NN_BACKEND_AUTO;
        unsafe {
            check(sys::chd_nn_model_get_active_backend(
                self.raw.as_ptr(),
                &mut raw,
            ))?
        };
        Ok(NnBackend::from_raw(raw))
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::chd_nn_model {
        self.raw.as_ptr()
    }
}

impl Drop for NnModel {
    fn drop(&mut self) {
        unsafe { sys::chd_nn_model_free(self.raw.as_ptr()) };
    }
}
