//! The frozen projection schedule (shared/schedule.json, written by shared/make_schedule.py from the
//! x86 snapshot's lf-asr/src/encoder.rs; plans/cpu-benchmark-mac.md M1), embedded at compile time.

use crate::kernel::Slot;

/// The exact bytes of shared/schedule.json this build embeds.
pub const SCHEDULE_JSON: &str = include_str!("../../schedule.json");

/// `content_sha256` field of the frozen file (make_schedule.py canonical JSON).
pub const SCHEDULE_CONTENT_SHA256: &str = "2404cc9e1968bcc0f9758b6ea80c0c9c14c1aeeb6a38d6d1a369d28bedf8cedd";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// Encoder frames T.
    T,
    /// 2T - 1 relative positions (linear_pos only).
    PosRows,
}

impl Rows {
    pub fn count(self, t: usize) -> usize {
        match self {
            Rows::T => t,
            Rows::PosRows => 2 * t - 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledCall {
    pub index: usize,
    pub layer: usize,
    /// Full module name, e.g. "encoder.layers.3.self_attn.linear_k".
    pub module: String,
    /// Module name within the layer, e.g. "self_attn.linear_k".
    pub short: String,
    pub n: usize,
    pub k: usize,
    pub rows: Rows,
    pub slot: Slot,
    /// Reuse the slot's current quantization (K and V after Q). `false` for cached calls.
    pub reuse: bool,
    /// Capture / quantization id "<layer>.<quant>" (None for linear_pos).
    pub quant: Option<String>,
    /// Part of the replay (and of the timed scope). False only for the cached linear_pos.
    pub replayed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureSpec {
    /// "<layer>.<quant>", in capture (= quantization) order.
    pub name: String,
    pub layer: usize,
    pub k: usize,
}

pub struct Schedule {
    pub calls: Vec<ScheduledCall>,
    pub captures: Vec<CaptureSpec>,
    pub doc: serde_json::Value,
}

impl Schedule {
    pub fn load() -> Schedule {
        let doc: serde_json::Value = serde_json::from_str(SCHEDULE_JSON).expect("schedule.json parses");
        assert_eq!(doc["content_sha256"].as_str(), Some(SCHEDULE_CONTENT_SHA256), "schedule.json changed");
        let calls = doc["calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| ScheduledCall {
                index: c["index"].as_u64().unwrap() as usize,
                layer: c["layer"].as_u64().unwrap() as usize,
                module: c["module"].as_str().unwrap().to_string(),
                short: c["short"].as_str().unwrap().to_string(),
                n: c["n"].as_u64().unwrap() as usize,
                k: c["k"].as_u64().unwrap() as usize,
                rows: match c["rows"].as_str().unwrap() {
                    "T" => Rows::T,
                    "2T-1" => Rows::PosRows,
                    r => panic!("rows {r}"),
                },
                slot: Slot::parse(c["slot"].as_str().unwrap()).unwrap(),
                reuse: c["reuse"].as_bool().unwrap_or(false),
                quant: c["quant"].as_str().map(str::to_string),
                replayed: c["replayed"].as_bool().unwrap(),
            })
            .collect();
        let captures = doc["captures"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| CaptureSpec {
                name: c["name"].as_str().unwrap().to_string(),
                layer: c["layer"].as_u64().unwrap() as usize,
                k: c["k"].as_u64().unwrap() as usize,
            })
            .collect();
        Schedule { calls, captures, doc }
    }

    /// The replayed (timed) calls in order.
    pub fn replayed(&self) -> impl Iterator<Item = &ScheduledCall> {
        self.calls.iter().filter(|c| c.replayed)
    }

    /// Unique (n, k) shapes of the replayed calls.
    pub fn shapes(&self) -> Vec<(usize, usize)> {
        let mut s: Vec<_> = self.replayed().map(|c| (c.n, c.k)).collect();
        s.sort();
        s.dedup();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trace contract on the frozen file itself (M1): counts, shapes, reuse and capture order.
    #[test]
    fn schedule_matches_localflow_encoder() {
        let s = Schedule::load();
        assert_eq!(s.calls.len(), 264);
        let replayed: Vec<_> = s.replayed().collect();
        assert_eq!(replayed.len(), 240);
        let per_layer: Vec<(&str, usize, usize, Slot, bool)> = replayed[..10]
            .iter()
            .map(|c| (c.short.as_str(), c.n, c.k, c.slot, c.reuse))
            .collect();
        assert_eq!(
            per_layer,
            vec![
                ("feed_forward1.linear1", 4096, 1024, Slot::Model, false),
                ("feed_forward1.linear2", 1024, 4096, Slot::Ff, false),
                ("self_attn.linear_q", 1024, 1024, Slot::Model, false),
                ("self_attn.linear_k", 1024, 1024, Slot::Model, true),
                ("self_attn.linear_v", 1024, 1024, Slot::Model, true),
                ("self_attn.linear_out", 1024, 1024, Slot::Model, false),
                ("conv.pointwise_conv1", 2048, 1024, Slot::Model, false),
                ("conv.pointwise_conv2", 1024, 1024, Slot::Model, false),
                ("feed_forward2.linear1", 4096, 1024, Slot::Model, false),
                ("feed_forward2.linear2", 1024, 4096, Slot::Ff, false),
            ]
        );
        for (l, chunk) in replayed.chunks(10).enumerate() {
            assert!(chunk.iter().all(|c| c.layer == l && c.rows == Rows::T));
        }
        let quantized: Vec<_> = replayed.iter().filter(|c| !c.reuse).map(|c| c.quant.clone().unwrap()).collect();
        let caps: Vec<_> = s.captures.iter().map(|c| c.name.clone()).collect();
        assert_eq!(quantized, caps);
        let pos: Vec<_> = s.calls.iter().filter(|c| !c.replayed).collect();
        assert_eq!(pos.len(), 24);
        assert!(pos.iter().all(|c| c.short == "self_attn.linear_pos" && c.rows == Rows::PosRows && c.slot == Slot::Pos));
        assert_eq!(s.shapes(), vec![(1024, 1024), (1024, 4096), (2048, 1024), (4096, 1024)]);
    }
}
