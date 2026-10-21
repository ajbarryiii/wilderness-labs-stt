//! MB1 / M1 encoder dimensions (Parakeet-TDT 0.6B v2/v3 FastConformer; LocalFlow lf-asr/src/config.rs).

pub const LAYERS: usize = 24;
pub const D_MODEL: usize = 1024;
pub const HEADS: usize = 8;
pub const D_FF: usize = 4 * D_MODEL;
