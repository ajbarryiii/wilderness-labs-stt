//! Weight matrices in the two export layouts, and a minimal safetensors reader.
//!
//! - `parakeet-binary-v1` (finetune/parakeet-binary/binary_export.py `pack_codes`): uint8
//!   `[N][ceil(K/8)]`, code `j` of a row is bit `j % 8` of byte `j / 8` (LSB first), bit 1 = +1,
//!   bit 0 = -1, unused tail bits 0 (rejected otherwise).
//! - `parakeet-ternary-v1` (finetune/parakeet-ternary/export.py; LocalFlow lf-model/src/ternary.rs):
//!   uint8 `[N][ceil(K/4)]`, code `j` at bits `2 * (j % 4)` of byte `j / 4`; `00` = 0, `01` = +1,
//!   `10` = -1; `11` and nonzero tail fields rejected.
//!
//! `W[o][i] = scale[o] * code(o, i)` in both. MB1's derived 2-bit export
//! (gpu1bit/shared/derive.py) is `parakeet-ternary-v1` with no `00` fields.

use std::collections::BTreeMap;
use std::path::Path;

use crate::rng::SplitMix64;

#[derive(Debug)]
pub struct WeightError(pub String);

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for WeightError {}

macro_rules! bail {
    ($($t:tt)*) => { return Err(WeightError(format!($($t)*))) };
}

/// Which export layout a kernel consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    /// parakeet-binary-v1, 1 bit per weight.
    Binary,
    /// parakeet-ternary-v1, 2 bits per weight.
    Ternary,
}

#[derive(Clone)]
pub struct BinaryMatrix {
    /// Outputs.
    pub n: usize,
    /// Inputs.
    pub k: usize,
    /// `[n][row_bytes()]`.
    pub codes: Vec<u8>,
    pub scale: Vec<f32>,
}

impl BinaryMatrix {
    pub fn row_bytes(&self) -> usize {
        self.k.div_ceil(8)
    }

    pub fn validate(&self) -> Result<(), WeightError> {
        if self.n == 0 || self.k == 0 || self.codes.len() != self.n * self.row_bytes() || self.scale.len() != self.n {
            bail!("binary matrix: inconsistent sizes");
        }
        if self.k % 8 != 0 {
            let mask = !((1u8 << (self.k % 8)) - 1);
            for o in 0..self.n {
                if self.codes[o * self.row_bytes() + self.row_bytes() - 1] & mask != 0 {
                    bail!("binary matrix: nonzero tail bits in row {o}");
                }
            }
        }
        Ok(())
    }

    /// The weight bit (true = +1).
    #[inline]
    pub fn bit(&self, o: usize, i: usize) -> bool {
        (self.codes[o * self.row_bytes() + i / 8] >> (i % 8)) & 1 == 1
    }

    #[inline]
    pub fn sign(&self, o: usize, i: usize) -> i8 {
        if self.bit(o, i) { 1 } else { -1 }
    }

    /// P = number of +1 weights in row `o`.
    pub fn popcount(&self, o: usize) -> u32 {
        self.codes[o * self.row_bytes()..(o + 1) * self.row_bytes()].iter().map(|b| b.count_ones()).sum()
    }

    /// `[n][k]` in {-1, +1}.
    pub fn signs(&self) -> Vec<i8> {
        let mut out = Vec::with_capacity(self.n * self.k);
        for o in 0..self.n {
            for i in 0..self.k {
                out.push(self.sign(o, i));
            }
        }
        out
    }

    /// The derived 2-bit layout (gpu1bit/shared/derive.py): +1 -> 01, -1 -> 10, no 00 fields.
    pub fn to_ternary(&self) -> TernaryMatrix {
        TernaryMatrix::from_codes(self.n, self.k, &self.signs(), self.scale.clone())
    }

    pub fn from_signs(n: usize, k: usize, signs: &[i8], scale: Vec<f32>) -> Self {
        assert_eq!(signs.len(), n * k);
        let rb = k.div_ceil(8);
        let mut codes = vec![0u8; n * rb];
        for o in 0..n {
            for i in 0..k {
                match signs[o * k + i] {
                    1 => codes[o * rb + i / 8] |= 1 << (i % 8),
                    -1 => {}
                    v => panic!("binary sign {v}"),
                }
            }
        }
        BinaryMatrix { n, k, codes, scale }
    }

    /// Uniform random signs, scales in [0.01, 0.05).
    pub fn random(n: usize, k: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let signs: Vec<i8> = (0..n * k).map(|_| if rng.next_u64() & 1 == 1 { 1 } else { -1 }).collect();
        let scale = (0..n).map(|_| 0.01 + 0.04 * rng.next_f32()).collect();
        Self::from_signs(n, k, &signs, scale)
    }
}

#[derive(Clone)]
pub struct TernaryMatrix {
    pub n: usize,
    pub k: usize,
    /// `[n][row_bytes()]`.
    pub codes: Vec<u8>,
    pub scale: Vec<f32>,
}

#[inline]
pub fn ternary_field_value(field: u8) -> i8 {
    match field & 3 {
        1 => 1,
        2 => -1,
        _ => 0,
    }
}

impl TernaryMatrix {
    pub fn row_bytes(&self) -> usize {
        self.k.div_ceil(4)
    }

    pub fn validate(&self) -> Result<(), WeightError> {
        if self.n == 0 || self.k == 0 || self.codes.len() != self.n * self.row_bytes() || self.scale.len() != self.n {
            bail!("ternary matrix: inconsistent sizes");
        }
        for o in 0..self.n {
            for (b, &byte) in self.codes[o * self.row_bytes()..(o + 1) * self.row_bytes()].iter().enumerate() {
                for j in 0..4 {
                    let field = (byte >> (2 * j)) & 3;
                    if field == 3 || (field != 0 && b * 4 + j >= self.k) {
                        bail!("ternary matrix: invalid code at row {o}, column {}", b * 4 + j);
                    }
                }
            }
        }
        Ok(())
    }

    #[inline]
    pub fn code(&self, o: usize, i: usize) -> i8 {
        ternary_field_value(self.codes[o * self.row_bytes() + i / 4] >> (2 * (i % 4)))
    }

    /// `[n][k]` in {-1, 0, +1}.
    pub fn values(&self) -> Vec<i8> {
        let mut out = Vec::with_capacity(self.n * self.k);
        for o in 0..self.n {
            for i in 0..self.k {
                out.push(self.code(o, i));
            }
        }
        out
    }

    pub fn row_sum(&self, o: usize) -> i32 {
        (0..self.k).map(|i| self.code(o, i) as i32).sum()
    }

    pub fn from_codes(n: usize, k: usize, values: &[i8], scale: Vec<f32>) -> Self {
        assert_eq!(values.len(), n * k);
        let rb = k.div_ceil(4);
        let mut codes = vec![0u8; n * rb];
        for o in 0..n {
            for i in 0..k {
                let field: u8 = match values[o * k + i] {
                    0 => 0,
                    1 => 1,
                    -1 => 2,
                    v => panic!("ternary value {v}"),
                };
                codes[o * rb + i / 4] |= field << (2 * (i % 4));
            }
        }
        TernaryMatrix { n, k, codes, scale }
    }

    /// Exactly representable as binary (no zero codes).
    pub fn to_binary(&self) -> Option<BinaryMatrix> {
        let v = self.values();
        if v.contains(&0) {
            return None;
        }
        Some(BinaryMatrix::from_signs(self.n, self.k, &v, self.scale.clone()))
    }

    /// Uniform over {-1, 0, +1}, scales in [0.01, 0.05).
    pub fn random(n: usize, k: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let v: Vec<i8> = (0..n * k).map(|_| rng.below(3) as i8 - 1).collect();
        let scale = (0..n).map(|_| 0.01 + 0.04 * rng.next_f32()).collect();
        Self::from_codes(n, k, &v, scale)
    }
}

/// A borrowed weight matrix in the layout a kernel declared.
#[derive(Clone, Copy)]
pub enum WeightsRef<'a> {
    Binary(&'a BinaryMatrix),
    Ternary(&'a TernaryMatrix),
}

impl WeightsRef<'_> {
    pub fn n(&self) -> usize {
        match self {
            WeightsRef::Binary(m) => m.n,
            WeightsRef::Ternary(m) => m.n,
        }
    }
    pub fn k(&self) -> usize {
        match self {
            WeightsRef::Binary(m) => m.k,
            WeightsRef::Ternary(m) => m.k,
        }
    }
    pub fn scale(&self) -> &[f32] {
        match self {
            WeightsRef::Binary(m) => &m.scale,
            WeightsRef::Ternary(m) => &m.scale,
        }
    }
    pub fn format(&self) -> WeightFormat {
        match self {
            WeightsRef::Binary(_) => WeightFormat::Binary,
            WeightsRef::Ternary(_) => WeightFormat::Ternary,
        }
    }
    /// `[n][k]` in {-1, 0, +1}: the oracle's view of the weights.
    pub fn values(&self) -> Vec<i8> {
        match self {
            WeightsRef::Binary(m) => m.signs(),
            WeightsRef::Ternary(m) => m.values(),
        }
    }
}

/// One tensor of a safetensors file.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

/// A whole safetensors file in memory (header: u64 LE length, JSON, then the data block).
pub struct SafeTensors {
    data: Vec<u8>,
    base: usize,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl SafeTensors {
    pub fn read(path: &Path) -> Result<Self, WeightError> {
        let data = std::fs::read(path).map_err(|e| WeightError(format!("{}: {e}", path.display())))?;
        Self::parse(data)
    }

    pub fn parse(data: Vec<u8>) -> Result<Self, WeightError> {
        if data.len() < 8 {
            bail!("safetensors: truncated");
        }
        let hlen = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        if 8 + hlen > data.len() {
            bail!("safetensors: header length {hlen} past end");
        }
        let header: serde_json::Value =
            serde_json::from_slice(&data[8..8 + hlen]).map_err(|e| WeightError(format!("safetensors header: {e}")))?;
        let obj = header.as_object().ok_or_else(|| WeightError("safetensors: header not an object".into()))?;
        let mut tensors = BTreeMap::new();
        let mut metadata = serde_json::Map::new();
        let base = 8 + hlen;
        for (name, v) in obj {
            if name == "__metadata__" {
                if let Some(m) = v.as_object() {
                    metadata = m.clone();
                }
                continue;
            }
            let dtype = v["dtype"].as_str().ok_or_else(|| WeightError(format!("{name}: dtype")))?.to_string();
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .ok_or_else(|| WeightError(format!("{name}: shape")))?
                .iter()
                .map(|d| d.as_u64().map(|x| x as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| WeightError(format!("{name}: shape")))?;
            let off = v["data_offsets"].as_array().ok_or_else(|| WeightError(format!("{name}: offsets")))?;
            let (start, end) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
            let elem = match dtype.as_str() {
                "U8" | "I8" | "BOOL" => 1,
                "F16" | "BF16" | "I16" => 2,
                "F32" | "I32" => 4,
                "F64" | "I64" => 8,
                other => bail!("{name}: unsupported dtype {other}"),
            };
            if end < start || base + end > data.len() || end - start != shape.iter().product::<usize>() * elem {
                bail!("{name}: inconsistent data offsets");
            }
            tensors.insert(name.clone(), TensorInfo { dtype, shape, start, end });
        }
        Ok(SafeTensors { data, base, tensors, metadata })
    }

    pub fn get(&self, name: &str) -> Result<(&TensorInfo, &[u8]), WeightError> {
        let info = self.tensors.get(name).ok_or_else(|| WeightError(format!("missing tensor {name}")))?;
        Ok((info, &self.data[self.base + info.start..self.base + info.end]))
    }

    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>, WeightError> {
        let (info, bytes) = self.get(name)?;
        if info.dtype != "F32" {
            bail!("{name}: expected F32, got {}", info.dtype);
        }
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// `<module>.codes` / `<module>.scale` as a binary matrix with `k` inputs (`k` from the manifest).
    pub fn binary(&self, module: &str, k: usize) -> Result<BinaryMatrix, WeightError> {
        if self.tensors.contains_key(&format!("{module}.bias")) {
            bail!("{module}: biases are not supported");
        }
        let (info, codes) = self.get(&format!("{module}.codes"))?;
        if info.dtype != "U8" || info.shape.len() != 2 || info.shape[1] != k.div_ceil(8) {
            bail!("{module}.codes: expected U8 [N, ceil({k}/8)], got {} {:?}", info.dtype, info.shape);
        }
        let m = BinaryMatrix { n: info.shape[0], k, codes: codes.to_vec(), scale: self.f32_vec(&format!("{module}.scale"))? };
        m.validate().map_err(|e| WeightError(format!("{module}: {e}")))?;
        Ok(m)
    }

    /// `<module>.codes` / `<module>.scale` as a ternary matrix (K = 4 * code bytes, as LocalFlow).
    pub fn ternary(&self, module: &str) -> Result<TernaryMatrix, WeightError> {
        if self.tensors.contains_key(&format!("{module}.bias")) {
            bail!("{module}: biases are not supported");
        }
        let (info, codes) = self.get(&format!("{module}.codes"))?;
        if info.dtype != "U8" || info.shape.len() != 2 {
            bail!("{module}.codes: expected 2-D U8");
        }
        let m = TernaryMatrix {
            n: info.shape[0],
            k: info.shape[1] * 4,
            codes: codes.to_vec(),
            scale: self.f32_vec(&format!("{module}.scale"))?,
        };
        m.validate().map_err(|e| WeightError(format!("{module}: {e}")))?;
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_layout_lsb_first_and_tail_rejection() {
        // Row: [+1, -1, -1, +1, +1, -1, -1, -1, +1] -> byte0 = 0b0001_1001, byte1 = 0b1.
        let m = BinaryMatrix { n: 1, k: 9, codes: vec![0b0001_1001, 0b1], scale: vec![1.0] };
        m.validate().unwrap();
        assert_eq!(m.signs(), vec![1, -1, -1, 1, 1, -1, -1, -1, 1]);
        assert_eq!(m.popcount(0), 4);
        let bad = BinaryMatrix { n: 1, k: 9, codes: vec![0, 0b10], scale: vec![1.0] };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn ternary_layout_and_derivation() {
        // Row codes: [+1, -1, 0, +1, -1] -> byte0 = 01 | 10<<2 | 00<<4 | 01<<6, byte1 = 10.
        let m = TernaryMatrix { n: 1, k: 5, codes: vec![0b01_00_10_01, 0b10], scale: vec![1.0] };
        m.validate().unwrap();
        assert_eq!(m.values(), vec![1, -1, 0, 1, -1]);
        assert!(TernaryMatrix { n: 1, k: 4, codes: vec![0b11], scale: vec![1.0] }.validate().is_err());
        assert!(TernaryMatrix { n: 1, k: 3, codes: vec![0b01 << 6], scale: vec![1.0] }.validate().is_err());
        let b = BinaryMatrix::random(7, 37, 3);
        let t = b.to_ternary();
        t.validate().unwrap();
        assert_eq!(t.values(), b.signs());
        assert_eq!(t.to_binary().unwrap().codes, b.codes);
    }
}
