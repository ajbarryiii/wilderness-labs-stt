//! Oracle reference digests for the full-utterance exactness replays (run on NixOS, any target):
//!   oracle-digests --export DIR --format binary|ternary --captures GROUP_DIR --out DIR
//!                  [--components 2,3] [--threads 16] [--clips 0,1]
//! Writes <out>/<clip dir>.C<c>.json: per replayed call the SHA-256 of the exact dots [T][C][N] (i32 LE)
//! and of the FP32 outputs [T][N] (bits LE), computed by the shared quantizer + shared::oracle.
use std::path::{Path, PathBuf};

use serde_json::json;
use shared::bench::Args;
use shared::data::{Clip, group_clips, load_weights};
use shared::replay::{digests_to_json, oracle_digests};
use shared::schedule::Schedule;
use shared::weights::WeightFormat;

fn main() {
    let args = Args::parse_from_flags();
    let sched = Schedule::load();
    let format = match args.req("format") {
        "binary" => WeightFormat::Binary,
        "ternary" => WeightFormat::Ternary,
        f => panic!("--format {f}"),
    };
    let (mats, info) = load_weights(Path::new(args.req("export")), format, &sched).expect("weights");
    let out = PathBuf::from(args.req("out"));
    std::fs::create_dir_all(&out).unwrap();
    let threads = args.list("threads", "16")[0];
    let all = group_clips(Path::new(args.req("captures")));
    let dirs: Vec<PathBuf> = match args.get("clips") {
        None => all,
        Some(_) => args.list("clips", "").into_iter().map(|i| all[i].clone()).collect(),
    };
    for d in &dirs {
        let clip = Clip::load(d, true);
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        for &c in &args.list("components", "2,3") {
            let t0 = std::time::Instant::now();
            let dg = oracle_digests(&mats, &sched, &clip, c, threads);
            let doc = json!({"clip": clip.id(), "clip_dir": name, "clip_sha256": clip.sha256, "T": clip.t, "components": c,
                             "export": info.to_json(), "build": shared::bench::build_info(), "digests": digests_to_json(&dg)});
            std::fs::write(out.join(format!("{name}.C{c}.json")), serde_json::to_string_pretty(&doc).unwrap()).unwrap();
            println!("{}", json!({"clip": name, "T": clip.t, "components": c, "seconds": t0.elapsed().as_secs_f64()}));
        }
    }
}
