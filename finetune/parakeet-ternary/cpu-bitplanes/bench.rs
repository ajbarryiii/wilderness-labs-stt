//! Matched full-transcription benchmark on public audio, with no transcript logs.
//! Each window consists of complete corpus cycles; order rotates between windows.
//! Frontend, encoder, decoder, and text generation are timed; I/O, loading,
//! weight packing, validation, first-use caches, and output hashing are excluded.

use lf_asr::encoder::{self, EncoderBuffers, PositionCache, Projector};
use lf_asr::frontend::{self, FrontendBuffers};
use lf_asr::{Model, Precision, Timings};
use lf_cpu::decoder::Storage;
use lf_cpu::gemm_bitplanes::Engine;
use lf_cpu::{QuantizedActs, Scratch, ThreadPool, gemm_i8};
use lf_model::Export;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn float_hash(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

struct Item {
    id: String,
    source: String,
    samples: Vec<f32>,
}
struct State {
    name: String,
    projector: Projector,
    front: FrontendBuffers,
    enc: EncoderBuffers,
    positions: PositionCache,
}
struct Run {
    timings: Timings,
    wall_ms: f64,
    text: String,
    tokens: Vec<(usize, usize)>,
    frames: usize,
}

impl State {
    fn new(name: &str, tile: usize, components: usize, threads: usize) -> Self {
        // Single-threaded configuration phase, before inference starts.
        unsafe {
            std::env::set_var("LF_BITPLANE_BACKEND", name);
            std::env::set_var("LF_BITPLANE_FRAMES", tile.to_string());
        }
        Self {
            name: name.into(),
            projector: Projector::new(Precision::Int8(components), threads),
            front: Default::default(),
            enc: Default::default(),
            positions: PositionCache::new(encoder::frames_for_samples(60 * 16000)),
        }
    }

    fn run(&mut self, model: &Model, pool: &ThreadPool, samples: &[f32]) -> Run {
        let total = Instant::now();
        let mut timings = Timings::default();
        let start = Instant::now();
        let mel = frontend::features(&model.frontend, pool, samples, &mut self.front).unwrap();
        timings.frontend = start.elapsed();
        let frames = encoder::encode(
            model,
            pool,
            &mut self.projector,
            &self.front.feat,
            mel,
            &mut self.enc,
            &mut self.positions,
            &mut timings,
        )
        .unwrap();
        let start = Instant::now();
        let projected = model
            .decoder
            .project_encoder(pool, &self.enc.x[..frames * 1024], frames);
        let tokens = model.decoder.decode(pool, &projected, frames, None);
        let ids: Vec<usize> = tokens.iter().map(|&(id, _)| id).collect();
        let text = model.tokenizer.decode(&ids).unwrap();
        timings.decoder = start.elapsed();
        Run {
            timings,
            wall_ms: total.elapsed().as_secs_f64() * 1000.0,
            text,
            tokens,
            frames,
        }
    }
}

fn measurements(run: &Run) -> Value {
    let t = run.timings;
    json!({"full_ms": run.wall_ms, "projections_ms": t.projections.as_secs_f64()*1000.0,
        "encoder_ms": (t.subsampling+t.projections+t.attention+t.elementwise).as_secs_f64()*1000.0,
        "frontend_ms": t.frontend.as_secs_f64()*1000.0, "decoder_ms": t.decoder.as_secs_f64()*1000.0,
        "attention_ms": t.attention.as_secs_f64()*1000.0, "frames": run.frames, "tokens": run.tokens.len()})
}

fn write(path: &str, value: &Value) {
    let temporary = format!("{path}.tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(value).unwrap()).unwrap();
    std::fs::rename(temporary, path).unwrap();
}

fn micro(model: &Model, pool: &ThreadPool, output: &str) {
    let mut results = Vec::new();
    for (label, w) in [
        ("ff_expand", &model.layers[0].ff1[0]),
        ("ff_contract", &model.layers[0].ff1[1]),
        ("attention", &model.layers[0].q),
    ] {
        for frames in [190, 375] {
            let x: Vec<f32> = (0..frames * w.cols())
                .map(|i| ((i * 7919 % 65521) as f32 - 32000.0) / 10000.0)
                .collect();
            for components in [2, 3] {
                let mut q = QuantizedActs::with_capacity(frames, w.cols(), components);
                q.quantize(pool, &x, frames).unwrap();
                let scratch = Scratch::new(pool.threads(), gemm_i8::scratch_bytes(w.cols()));
                let mut reference = vec![0.0; frames * w.rows()];
                gemm_i8::gemm(pool, &scratch, w, &q, &mut reference);
                for name in ["pn", "masked", "swar"] {
                    for tile in if name == "swar" {
                        vec![1, 2, 4, 8]
                    } else {
                        vec![1, 2]
                    } {
                        unsafe {
                            std::env::set_var("LF_BITPLANE_BACKEND", name);
                            std::env::set_var("LF_BITPLANE_FRAMES", tile.to_string());
                        }
                        let mut engine = Engine::from_env().unwrap();
                        let mut y = vec![0.0; reference.len()];
                        engine.gemm(pool, w, &q, &mut y, 0, false);
                        assert!(
                            y.iter()
                                .zip(&reference)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "micro output mismatch: {name}"
                        );
                        let mut times = Vec::new();
                        for _ in 0..7 {
                            let start = Instant::now();
                            engine.gemm(pool, w, &q, &mut y, 0, false);
                            times.push(start.elapsed().as_secs_f64() * 1000.0);
                            std::hint::black_box(&y);
                        }
                        times.sort_by(f64::total_cmp);
                        results.push(json!({"shape":label,"m":frames,"n":w.rows(),"k":w.cols(),"components":components,
                            "backend":name,"tile":tile,"median_ms":times[times.len()/2],"times_ms":times,"exact_vnni":true}));
                    }
                }
            }
        }
    }
    write(
        output,
        &json!({"scope":"real trained weights, synthetic activations; quantization excluded; bit-plane packing and input sums included; weight packing excluded", "cases":results}),
    );
    println!("MICRO complete cases={}", results.len());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let export = Export::open(Path::new(&args[2])).unwrap();
    let model = Model::load(&export, Storage::F16).unwrap();
    let cpus: Vec<usize> = args[4].split(',').map(|s| s.parse().unwrap()).collect();
    let pool = ThreadPool::new(&cpus);
    if args[1] == "micro" {
        micro(&model, &pool, &args[3]);
        return;
    }
    let manifest: Value = serde_json::from_slice(&std::fs::read(&args[3]).unwrap()).unwrap();
    let items: Vec<Item> = manifest
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let bytes = std::fs::read(r["pcm"].as_str().unwrap()).unwrap();
            assert_eq!(hash(&bytes), r["pcm_sha256"].as_str().unwrap());
            let samples = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            Item {
                id: r["id"].as_str().unwrap().into(),
                source: r["source"].as_str().unwrap().into(),
                samples,
            }
        })
        .collect();
    let components: usize = args[5].parse().unwrap();
    let tiles: BTreeMap<&str, usize> = args[6]
        .split(',')
        .map(|v| {
            let (name, tile) = v.split_once('=').unwrap();
            (name, tile.parse().unwrap())
        })
        .collect();
    let seconds: f64 = args[7].parse().unwrap();
    let windows: usize = args[8].parse().unwrap();
    let output = &args[9];
    let mut states: Vec<State> = ["vnni", "pn", "masked", "swar"]
        .iter()
        .map(|name| State::new(name, *tiles.get(name).unwrap_or(&2), components, cpus.len()))
        .collect();
    let longest = items.iter().max_by_key(|r| r.samples.len()).unwrap();
    for state in &mut states {
        state.run(&model, &pool, &longest.samples);
        state.run(&model, &pool, &longest.samples);
        println!("WARMED {}", state.name);
    }
    let mut expected = Vec::new();
    let mut validation = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let reference = states[0].run(&model, &pool, &item.samples);
        let fingerprint = float_hash(&states[0].enc.x[..reference.frames * 1024]);
        for state in states.iter_mut().skip(1) {
            let result = state.run(&model, &pool, &item.samples);
            assert_eq!(
                fingerprint,
                float_hash(&state.enc.x[..result.frames * 1024]),
                "encoder mismatch: {}",
                state.name
            );
            assert!(
                reference.tokens == result.tokens,
                "decoder mismatch: {}",
                state.name
            );
            assert!(
                reference.text == result.text,
                "text mismatch: {}",
                state.name
            );
        }
        validation.push(
            json!({"index":index,"id":item.id,"source":item.source,"encoder_sha256":fingerprint,
            "text_sha256":hash(reference.text.as_bytes()),"exact_all_backends":true}),
        );
        expected.push((reference.text, reference.tokens));
        println!("VALIDATED {}/{}", index + 1, items.len());
    }
    let mut records = Vec::new();
    for window in 0..windows {
        for offset in 0..states.len() {
            let arm = (offset + window) % states.len();
            let state = &mut states[arm];
            let mut samples = Vec::new();
            let start = Instant::now();
            let mut cycles = 0;
            while start.elapsed() < Duration::from_secs_f64(seconds) || cycles < 2 {
                for index in 0..items.len() {
                    let index = (index + window * 5) % items.len();
                    let run = state.run(&model, &pool, &items[index].samples);
                    assert!(run.text == expected[index].0, "timed text mismatch");
                    assert!(run.tokens == expected[index].1, "timed tokens mismatch");
                    let mut row = measurements(&run);
                    row["index"] = json!(index);
                    samples.push(row);
                }
                cycles += 1;
            }
            let elapsed = start.elapsed().as_secs_f64();
            records.push(
                json!({"backend":state.name,"window":window,"elapsed_s":elapsed,"cycles":cycles,
                "calls":samples.len(),"samples":samples}),
            );
            write(
                output,
                &json!({"status":"running","cpus":cpus,"components":components,"tiles":tiles,"validation":validation,"windows":records}),
            );
            println!(
                "WINDOW {} {} calls={} elapsed={elapsed:.2}s",
                window,
                state.name,
                cycles * items.len()
            );
        }
    }
    write(
        output,
        &json!({"status":"complete","cpus":cpus,"components":components,"tiles":tiles,"validation":validation,"windows":records,
        "scope":"warm 16kHz waveform in memory to text; full corpus cycles; decoder and quantization unchanged; bit-plane packing and input sums included; no energy counters available"}),
    );
}
