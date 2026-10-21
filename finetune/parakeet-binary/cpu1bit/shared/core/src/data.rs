//! Captured inputs and model weights for the replay (plans/cpu-benchmark-mac.md "Inputs", M5).
//!
//! - A clip directory (shared/capture.py) holds `captures.f32` (the 192 quantization inputs in
//!   schedule capture order, each `[T][K]` f32 LE) and `index.json` (T, per-capture offsets, SHA-256).
//!   One clip is resident at a time (M5: at most 2 GB of captures per process; the largest is 0.49 GB).
//! - Weights come from an export directory (`export.safetensors` + `manifest.json`): parakeet-ternary-v1
//!   (M1, or MB1's derived 2-bit export) or parakeet-binary-v1 (MB1). The safetensors buffer is
//!   dropped once the 240 replayed matrices are built (the storage audit counts what remains).

use std::path::{Path, PathBuf};

use sha2::Digest;

use crate::schedule::Schedule;
use crate::weights::{BinaryMatrix, SafeTensors, TernaryMatrix, WeightError, WeightFormat};

pub fn sha256_hex(bytes: &[u8]) -> String {
    sha2::Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// One clip's captured projection inputs.
pub struct Clip {
    pub dir: PathBuf,
    pub index: serde_json::Value,
    pub t: usize,
    data: Vec<f32>,
    /// capture name -> (offset in floats, k)
    offsets: std::collections::HashMap<String, (usize, usize)>,
    pub sha256: String,
}

impl Clip {
    /// Loads a clip; with `verify`, recomputes the file's SHA-256 against index.json.
    pub fn load(dir: &Path, verify: bool) -> Clip {
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap_or_else(|e| panic!("{}: {e}", dir.display())))
                .expect("index.json");
        let bytes = std::fs::read(dir.join("captures.f32")).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        let want = index["sha256"].as_str().expect("sha256").to_string();
        if verify {
            let got = sha256_hex(&bytes);
            assert_eq!(got, want, "{}: captures.f32 does not match index.json", dir.display());
        }
        assert_eq!(bytes.len() as u64, index["bytes"].as_u64().unwrap(), "{}: size", dir.display());
        let t = index["T"].as_u64().unwrap() as usize;
        let mut data = vec![0f32; bytes.len() / 4];
        for (d, c) in data.iter_mut().zip(bytes.chunks_exact(4)) {
            *d = f32::from_le_bytes(c.try_into().unwrap());
        }
        drop(bytes);
        let mut offsets = std::collections::HashMap::new();
        for c in index["captures"].as_array().unwrap() {
            assert_eq!(c["rows"].as_u64().unwrap() as usize, t);
            offsets.insert(
                c["name"].as_str().unwrap().to_string(),
                (c["offset"].as_u64().unwrap() as usize, c["k"].as_u64().unwrap() as usize),
            );
        }
        Clip { dir: dir.to_path_buf(), index, t, data, offsets, sha256: want }
    }

    /// The `[T][K]` input of capture `name`.
    pub fn input(&self, name: &str) -> &[f32] {
        let &(off, k) = self.offsets.get(name).unwrap_or_else(|| panic!("capture {name} missing"));
        &self.data[off..off + self.t * k]
    }

    pub fn names(&self) -> Vec<String> {
        self.index["captures"].as_array().unwrap().iter().map(|c| c["name"].as_str().unwrap().to_string()).collect()
    }

    pub fn bytes(&self) -> usize {
        self.data.len() * 4
    }

    pub fn id(&self) -> String {
        self.index["clip"]["id"].as_str().unwrap_or("?").to_string()
    }
}

/// Clip directories of a capture group (`<group>/index.json` lists `dir`s), in index order.
pub fn group_clips(group_dir: &Path) -> Vec<PathBuf> {
    let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(group_dir.join("index.json")).expect("group index.json"))
        .expect("group index");
    idx["utterances"].as_array().unwrap().iter().map(|u| group_dir.join(u["dir"].as_str().unwrap())).collect()
}

/// One replayed call's weights in the layout a kernel consumes.
pub enum Matrix {
    Binary(BinaryMatrix),
    Ternary(TernaryMatrix),
}

impl Matrix {
    pub fn as_ref(&self) -> crate::weights::WeightsRef<'_> {
        match self {
            Matrix::Binary(m) => crate::weights::WeightsRef::Binary(m),
            Matrix::Ternary(m) => crate::weights::WeightsRef::Ternary(m),
        }
    }
}

/// The export's identity, recorded with every block.
#[derive(Clone, Debug)]
pub struct ExportInfo {
    pub dir: PathBuf,
    pub format: String,
    pub sha256: String,
    pub manifest_sha256: String,
}

impl ExportInfo {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"dir": self.dir, "format": self.format, "export_safetensors_sha256": self.sha256,
                           "manifest_sha256": self.manifest_sha256})
    }
}

/// Loads the 240 replayed matrices (schedule order) of an export in `format`, verifying the
/// safetensors file against its manifest. The file buffer is freed before returning.
pub fn load_weights(dir: &Path, format: WeightFormat, sched: &Schedule) -> Result<(Vec<Matrix>, ExportInfo), WeightError> {
    let manifest_bytes = std::fs::read(dir.join("manifest.json")).map_err(|e| WeightError(format!("{}: {e}", dir.display())))?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).map_err(|e| WeightError(e.to_string()))?;
    let file = dir.join(manifest["file"].as_str().unwrap_or("export.safetensors"));
    let bytes = std::fs::read(&file).map_err(|e| WeightError(format!("{}: {e}", file.display())))?;
    let digest = sha256_hex(&bytes);
    if manifest["sha256"].as_str() != Some(digest.as_str()) {
        return Err(WeightError(format!("{}: SHA-256 {digest} != manifest", file.display())));
    }
    let fmt = manifest["format"].as_str().unwrap_or("").to_string();
    let st = SafeTensors::parse(bytes)?;
    let mut out = Vec::new();
    for c in sched.replayed() {
        let shape = &manifest["quantized_layers"][&c.module]["shape"];
        let (n, k) = (shape[0].as_u64().unwrap() as usize, shape[1].as_u64().unwrap() as usize);
        if (n, k) != (c.n, c.k) {
            return Err(WeightError(format!("{}: manifest shape {n}x{k} != schedule {}x{}", c.module, c.n, c.k)));
        }
        let m = match (fmt.as_str(), format) {
            ("parakeet-ternary-v1", WeightFormat::Ternary) => Matrix::Ternary(st.ternary(&c.module)?),
            ("parakeet-binary-v1", WeightFormat::Binary) => Matrix::Binary(st.binary(&c.module, k)?),
            (f, want) => return Err(WeightError(format!("export format {f} cannot feed a {want:?} kernel"))),
        };
        let (mn, mk) = match &m {
            Matrix::Binary(b) => (b.n, b.k),
            Matrix::Ternary(t) => (t.n, t.k),
        };
        if (mn, mk) != (c.n, c.k) {
            return Err(WeightError(format!("{}: tensor {mn}x{mk} != schedule {}x{}", c.module, c.n, c.k)));
        }
        out.push(m);
    }
    drop(st);
    let info = ExportInfo { dir: dir.to_path_buf(), format: fmt, sha256: digest, manifest_sha256: sha256_hex(&manifest_bytes) };
    Ok((out, info))
}
