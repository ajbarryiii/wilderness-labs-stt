import BenchCore
import CoreML
import Foundation

/// parakeet-bench: macOS command-line driver of BenchCore (run on the Mac through ios/macguard).
///
///   parakeet-bench run  --models DIR --clips clips.json --pcm DIR --out results.jsonl [--mode free|replay]
///                       [--traces traces.json] [--compute-units cpuAndNeuralEngine] [--preprocessor-units cpuOnly]
///                       [--kinds natural,...] [--ids a,b] [--limit N] [--warmups 3] [--timed 10] [--emit-warmups]
///                       [--diag-dir DIR]
///   parakeet-bench plan --models DIR --out DIR [--compute-units cpuAndNeuralEngine] [--preprocessor-units cpuOnly]
///   parakeet-bench info --models DIR
///
/// run: per clip, `warmups` untimed calls, then `timed` calls (DESIGN.md "Repetition and statistics"); with
/// --diag-dir, afterwards one separate untimed diagnostic call per clip (record "diagnostic", binary arrays in
/// DIR; see BenchCore.Diagnostics). Replay validates every trace of traces.json against clips.json before any model is
/// loaded. Every output path must lie under the Mac artifact root, outside Git (ArtifactPath).
@main
struct ParakeetBenchCLI {
    static func main() async {
        do {
            var args = Array(CommandLine.arguments.dropFirst())
            guard let command = args.first else { throw BenchError.invalid(usage) }
            args.removeFirst()
            switch command {
            case "run": try await run(try Options(args, allowed: runOptions, flags: ["--emit-warmups"]))
            case "plan": try await plan(try Options(args, allowed: ["--models", "--out", "--compute-units",
                                                                    "--preprocessor-units"], flags: []))
            case "info": try info(try Options(args, allowed: ["--models"], flags: []))
            default: throw BenchError.invalid(usage)
            }
        } catch {
            FileHandle.standardError.write("parakeet-bench: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }

    static let usage = "usage: parakeet-bench run|plan|info --models DIR ... (see ParakeetBenchCLI.swift)"
    static let runOptions: Set<String> = ["--models", "--clips", "--pcm", "--traces", "--mode", "--compute-units",
                                          "--preprocessor-units", "--kinds", "--ids", "--limit", "--warmups", "--timed",
                                          "--out", "--diag-dir"]

    struct Options {
        var values: [String: String] = [:]
        var flags: Set<String> = []

        init(_ args: [String], allowed: Set<String>, flags flagNames: Set<String>) throws {
            var i = 0
            while i < args.count {
                let a = args[i]
                if flagNames.contains(a) { flags.insert(a); i += 1; continue }
                guard allowed.contains(a) else { throw BenchError.invalid("unknown option \(a)") }
                guard i + 1 < args.count else { throw BenchError.invalid("\(a) needs a value") }
                values[a] = args[i + 1]
                i += 2
            }
        }

        func string(_ key: String, _ fallback: String? = nil) throws -> String {
            guard let v = values[key] ?? fallback else { throw BenchError.invalid("missing \(key)") }
            return v
        }

        func int(_ key: String, _ fallback: Int) throws -> Int {
            guard let s = values[key] else { return fallback }
            guard let v = Int(s), v >= 0 else { throw BenchError.invalid("\(key) must be a non-negative integer") }
            return v
        }

        func url(_ key: String) throws -> URL { URL(fileURLWithPath: try string(key)) }
    }

    static func run(_ o: Options) async throws {
        // Everything that can be checked is checked before the models load.
        let modeName = try o.string("--mode", "free")
        guard DecodeMode.names.contains(modeName) else { throw BenchError.invalid("unknown --mode \(modeName)") }
        let units = try ComputeUnitsName.parse(try o.string("--compute-units", "cpuAndNeuralEngine"))
        let preUnits = try ComputeUnitsName.parse(try o.string("--preprocessor-units", "cpuOnly"))
        let manifest = try ClipManifest.load(try o.url("--clips"))
        let pcmDir = try o.url("--pcm")
        let proto = RepetitionProtocol(warmups: try o.int("--warmups", 3), timed: try o.int("--timed", 10),
                                       emitWarmups: o.flags.contains("--emit-warmups"))
        let outURL = try ArtifactPath.check(try o.url("--out"))
        let diagDir = try o.values["--diag-dir"].map { try ArtifactPath.check(URL(fileURLWithPath: $0)) }
        var clips = manifest.clips
        if let kinds = o.values["--kinds"] {
            let set = Set(kinds.split(separator: ",").map(String.init))
            clips = clips.filter { set.contains($0.kind) }
        }
        if let ids = o.values["--ids"] {
            let set = Set(ids.split(separator: ",").map(String.init))
            let known = Set(manifest.clips.map(\.id))
            guard set.isSubset(of: known) else { throw BenchError.invalid("unknown clip ids \(set.subtracting(known))") }
            clips = clips.filter { set.contains($0.id) }
        }
        if o.values["--limit"] != nil { clips = Array(clips.prefix(try o.int("--limit", clips.count))) }
        guard !clips.isEmpty else { throw BenchError.invalid("no clips selected") }
        let probe = TdtConfig()
        var traces: [String: TraceClip] = [:]
        if modeName == "replay" {
            // every trace of the manifest is validated, not only the selected clips'
            traces = try TraceFile.load(try o.url("--traces")).validated(
                for: manifest.clips, manifest: manifest, blank: probe.blankId, durations: probe.durationBins,
                maxSymbols: probe.maxSymbolsPerStep)
        }
        let pcms = try clips.map { try readPCM(directory: pcmDir, clip: $0) }  // SHA-256 checked up front
        if let diagDir { try FileManager.default.createDirectory(at: diagDir, withIntermediateDirectories: true) }
        try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let writer = try JSONLWriter(path: outURL.path)

        let footprint0 = physFootprint()
        let models = try C0Models(directory: try o.url("--models"), computeUnits: units, preprocessorUnits: preUnits)
        let footprintLoaded = physFootprint()
        let pipeline = C0Pipeline(models: models)
        try writer.writeObject([
            "record": "load", "arm": "C0", "compute_units": ComputeUnitsName.name(units),
            "preprocessor_units": ComputeUnitsName.name(preUnits), "load_ms": models.loadMs,
            "load_cache_evidence": "none: MLModel(contentsOf:) wall time only; prepare-and-cache vs cached load needs "
                + "an Instruments Core ML trace (DESIGN.md Load)",
            "phys_footprint_mb_before": Double(footprint0.current) / 1_048_576,
            "phys_footprint_mb_after_load": Double(footprintLoaded.current) / 1_048_576,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "clip_ids": clips.map(\.id), "warmups": proto.warmups, "timed": proto.timed, "mode": modeName,
            "diagnostics": diagDir != nil,
            "clips_json_sha256": manifest.fileSHA256,
        ])
        let t0 = Clock.now()
        for (clip, pcm) in zip(clips, pcms) {
            let mode: DecodeMode = modeName == "replay" ? .replay(traces[clip.id]!) : .free
            for rep in 0..<(proto.warmups + proto.timed) {
                let warm = rep < proto.warmups
                let result = try await pipeline.run(pcm, mode: mode)
                if !warm || proto.emitWarmups || rep == 0 {
                    let text = detokenize(result.tokens, vocabulary: models.vocabulary)
                    try writer.write(CallRecord(arm: "C0", mode: modeName, computeUnits: ComputeUnitsName.name(units),
                                                clip: clip, rep: rep, warmup: warm, text: text, result: result))
                }
            }
            if let diagDir {
                let diag = Diagnostics()
                let result = try await pipeline.run(pcm, mode: mode, diagnostics: diag)
                try writer.writeObject(try diagnosticRecord(clip: clip, mode: modeName, result: result, diag: diag,
                                                            trace: traces[clip.id], dir: diagDir, pipeline: pipeline,
                                                            vocabulary: models.vocabulary))
            }
        }
        let end = physFootprint()
        try writer.writeObject([
            "record": "end", "seconds": Clock.ms(t0, Clock.now()) / 1000,
            "phys_footprint_mb": Double(end.current) / 1_048_576, "phys_footprint_peak_mb": Double(end.peak) / 1_048_576,
        ])
    }

    /// The untimed diagnostic record of one clip; arrays go to <dir>/<clip>.<mode>.diag.f32.
    static func diagnosticRecord(clip: Clip, mode: String, result: CallResult, diag: Diagnostics, trace: TraceClip?,
                                 dir: URL, pipeline: C0Pipeline, vocabulary: [Int: String]) throws -> [String: Any] {
        let data = diag.binary()
        let file = dir.appendingPathComponent("\(clip.id).\(mode).diag.f32")
        try data.write(to: file)
        var record: [String: Any] = [
            "record": "diagnostic", "arm": "C0", "mode": mode, "clip": clip.id, "kind": clip.kind, "bucket": clip.bucket,
            "timed": false, "tokens": result.tokens, "text": detokenize(result.tokens, vocabulary: vocabulary),
            "encoder_length": result.encoderLength, "effective_frames": result.effectiveFrames,
            "mel_length": result.melLength, "decoder_calls": result.decoderCalls, "joint_calls": result.jointCalls,
            "steps": ["frame": diag.frames, "token_id": diag.tokenIds, "token_prob": diag.tokenProbs.map(Double.init),
                      "duration_bin": diag.durationBins, "decoder_call": diag.stepDecoderCall],
            "decoder_call_tokens": diag.decoderTokens,
            "arrays": ["file": file.lastPathComponent, "bytes": data.count, "sha256": sha256Hex(data),
                       "dtype": "float32 little-endian",
                       "layout": "encoder [encoder_frames, 1024] time-major; then per decoder call k: decoder output "
                           + "[640], h_out [2, 640], c_out [2, 640] (copied right after call k)",
                       "encoder_frames": diag.encoderFrames, "decoder_calls": diag.decoderTokens.count],
            "logits": NSNull(),
            "logits_unavailable": "C0's JointDecision.mlmodelc outputs only token_id (argmax of 1,025 token+blank "
                + "logits), token_prob (softmax prob of that token) and duration (argmax bin of 5 duration logits); "
                + "raw logits are not obtainable from the published FluidAudio v2 models",
        ]
        if let trace {
            let n = diag.tokenIds.count
            var tokenAgree = 0, durationAgree = 0
            var disagreements: [[Int]] = []
            for i in 0..<n {
                let dur = pipeline.durations[diag.durationBins[i]]
                let tok = diag.tokenIds[i] == trace.token[i], du = dur == trace.duration[i]
                if tok { tokenAgree += 1 }
                if du { durationAgree += 1 }
                if !(tok && du) { disagreements.append([i, trace.frame[i], trace.token[i], diag.tokenIds[i], trace.duration[i], dur]) }
            }
            record["replay"] = ["steps": n, "trace_steps": trace.steps, "token_agree": tokenAgree,
                                "duration_agree": durationAgree,
                                "disagreements [step, frame, trace token, C0 token, trace duration, C0 duration]": disagreements]
        }
        return record
    }

    static func plan(_ o: Options) async throws {
        let units = try ComputeUnitsName.parse(try o.string("--compute-units", "cpuAndNeuralEngine"))
        let preUnits = try ComputeUnitsName.parse(try o.string("--preprocessor-units", "cpuOnly"))
        let models = try o.url("--models")
        let out = try ArtifactPath.check(try o.url("--out"))
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        var summary: [String: Any] = [:]
        for name in C0Models.names {
            let u = name == "Preprocessor" ? preUnits : units
            var dump = try await ComputePlanDump.dump(model: models.appendingPathComponent("\(name).mlmodelc"), computeUnits: u)
            let data = try JSONSerialization.data(withJSONObject: dump, options: [.sortedKeys])
            try data.write(to: out.appendingPathComponent("\(name).\(ComputeUnitsName.name(u)).computeplan.json"))
            dump.removeValue(forKey: "ops")
            summary[name] = dump
        }
        let data = try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys, .prettyPrinted])
        try data.write(to: out.appendingPathComponent("summary.\(ComputeUnitsName.name(units)).json"))
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    }

    static func info(_ o: Options) throws {
        let models = try C0Models(directory: try o.url("--models"))
        for (name, model) in [("Preprocessor", models.preprocessor), ("Encoder", models.encoder),
                              ("Decoder", models.decoder), ("JointDecision", models.joint)] {
            let d = model.modelDescription
            print(name, "load_ms", models.loadMs[name] ?? 0)
            for (k, v) in d.inputDescriptionsByName.sorted(by: { $0.key < $1.key }) { print("  in ", k, v) }
            for (k, v) in d.outputDescriptionsByName.sorted(by: { $0.key < $1.key }) { print("  out", k, v) }
        }
    }
}
