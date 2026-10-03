import BenchCore
import CoreML
import Foundation

/// parakeet-bench: macOS command-line driver of BenchCore (run on the Mac through ios/macguard).
///
///   parakeet-bench run  --clips clips.json --pcm DIR --out results.jsonl [--mode free|replay] [--traces traces.json]
///                       [--kinds natural,...] [--ids a,b] [--limit N] [--warmups 3] [--timed 10] [--emit-warmups]
///                       [--diag-dir DIR] [--compute-units cpuAndNeuralEngine] [--preprocessor-units cpuOnly]
///                       ARM
///     ARM = --arm c0 --models C0DIR                                     (C0: FluidAudio 0.7.8's loop)
///         | --arm custom [--arm-name NAME] --decode f0|f1|f2|f1native
///                 ( --frontend vdsp --frontend-constants DIR | --frontend c0pre --models C0DIR )
///                 --encoder PATH(.mlmodelc|.mlpackage) --encoder-variant fixed15|multifunction|enumerated
///               | --encoder-input DIR                                   (gate mode: <id>.f32 [T, 1024])
///                 f0: --decoder-models DIR (Decoder.mlmodelc + JointDecision.mlmodelc, C0's contract)
///                 f1: --decoder-models DIR (DecoderJoint.mlmodelc or .mlpackage)
///                 f2, f1native: --native-weights DIR (native.py weights)
///                 [--vocab parakeet_vocab.json]  (default: C0DIR/parakeet_vocab.json if --models is given)
///                 --eligibility MODEL:ARM  (required with --encoder: a passing revision-8 PIPELINE record
///                 <ios>/results/eligibility/pipelines/<model>-<arm>-<variant>-<backend>-<frontend>-<decode>.json whose
///                 component SHA-256s (encoder package, front-end constants or preprocessor, F2 weights or decoder
///                 models + precision) and compute units equal the loaded ones, plus the arm's WP3 record;
///                 ios = --ios or the directory of clips.json; C0 is exempt)
///                 [--pair-c0 C0DIR --c0-out PATH]  (C0 and the arm interleaved clip by clip in this process;
///                 C0 runs as shipped on --c0-compute-units, default cpuAndNeuralEngine, whatever the arm's units)
///   parakeet-bench features --frontend-constants DIR --clips clips.json --pcm DIR --out DIR [--kinds/--ids]
///                       front end A on every clip: <id>.mel.f32 [128, N // 160 + 1] and features.jsonl
///   parakeet-bench encode --encoder PATH --encoder-variant fixed15 --clips clips.json --features DIR --tags vdsp,ref64
///                       --out DIR [--compute-units cpuAndNeuralEngine]
///                       the encoder on given features (<DIR>/<id>.mel.f32 for tag vdsp, <id>.<tag>.mel.f32 otherwise,
///                       [128, N // 160 + 1]) zero-padded to the bucket, mel_length = N // 160: <id>.<tag>.enc.f32
///                       [encoder_length, 1024] time-major + encode.jsonl (gate 5, rev. 5)
///   parakeet-bench gate --eligibility MODEL:ARM --encoder PATH --encoder-variant V --compute-units U
///                       --frontend-constants DIR --native-weights DIR --decoder-models DIR --decodes f2,f0,f1
///                       --clips clips.json --pcm DIR --traces traces.json --out DIR
///                       deployed-pipeline gate run (untimed): per clip front end A -> encoder (own bucket) once, then
///                       per decode loop free decoding and forced replay of the trace; gate.jsonl (+ enc/<id>.f32) with
///                       the components of every pipeline, for ios/pipegate.py
///   parakeet-bench plan --models DIR --out DIR [--compute-units cpuAndNeuralEngine] [--preprocessor-units cpuOnly]
///   parakeet-bench info --models DIR
///
/// run: per clip, `warmups` untimed calls, then `timed` calls (DESIGN.md "Repetition and statistics"); with
/// --diag-dir, afterwards one separate untimed diagnostic call per clip (record "diagnostic", binary arrays in
/// DIR). Replay validates every trace of traces.json against clips.json before any model is loaded. Every output
/// path must lie under the Mac artifact root, outside Git (ArtifactPath).
@main
struct ParakeetBenchCLI {
    static func main() async {
        do {
            var args = Array(CommandLine.arguments.dropFirst())
            guard let command = args.first else { throw BenchError.invalid(usage) }
            args.removeFirst()
            switch command {
            case "run": try await run(try Options(args, allowed: runOptions, flags: ["--emit-warmups"]))
            case "features": try await features(try Options(args, allowed: ["--frontend-constants", "--clips", "--pcm",
                                                                             "--out", "--kinds", "--ids"], flags: []))
            case "gate": try await gate(try Options(args, allowed: ["--clips", "--pcm", "--traces", "--eligibility", "--encoder",
                                                                     "--encoder-variant", "--compute-units", "--frontend-constants",
                                                                     "--native-weights", "--decoder-models", "--decodes", "--out",
                                                                     "--ids"], flags: []))
            case "encode": try await encode(try Options(args, allowed: ["--encoder", "--encoder-variant", "--compute-units",
                                                                         "--clips", "--features", "--tags", "--out", "--kinds",
                                                                         "--ids"], flags: []))
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
                                          "--out", "--diag-dir", "--arm", "--arm-name", "--decode", "--frontend",
                                          "--frontend-constants", "--encoder", "--encoder-variant", "--encoder-input",
                                          "--decoder-models", "--native-weights", "--vocab", "--eligibility", "--ios",
                                          "--pair-c0", "--c0-out", "--c0-compute-units"]

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

    static func selectClips(_ o: Options, _ manifest: ClipManifest) throws -> [Clip] {
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
        return clips
    }

    /// The arm to run: C0 (C0Pipeline) or a custom ArmPipeline, with its load record fields.
    enum Runner {
        case c0(C0Pipeline, C0Models)
        case custom(ArmPipeline)
    }

    static func run(_ o: Options) async throws {
        // Everything that can be checked is checked before the models load.
        let modeName = try o.string("--mode", "free")
        guard DecodeMode.names.contains(modeName) else { throw BenchError.invalid("unknown --mode \(modeName)") }
        let armKind = try o.string("--arm", "c0")
        guard ["c0", "custom"].contains(armKind) else { throw BenchError.invalid("unknown --arm \(armKind)") }
        let decodeName = try o.string("--decode", armKind == "c0" ? "c0" : "f2")
        if armKind == "custom" {
            guard ["f0", "f1", "f2", "f1native"].contains(decodeName) else { throw BenchError.invalid("unknown --decode \(decodeName)") }
            if o.values["--encoder-input"] == nil {
                guard let fe = o.values["--frontend"], FrontEndKind(rawValue: fe) != nil else {
                    throw BenchError.invalid("--frontend vdsp|c0pre is required (or --encoder-input)")
                }
                guard let v = o.values["--encoder-variant"], LengthVariant(rawValue: v) != nil, o.values["--encoder"] != nil else {
                    throw BenchError.invalid("--encoder PATH and --encoder-variant fixed15|multifunction|enumerated are required")
                }
            }
        } else {
            for key in ["--decode", "--frontend", "--frontend-constants", "--encoder", "--encoder-variant", "--encoder-input",
                        "--decoder-models", "--native-weights"] where o.values[key] != nil {
                throw BenchError.invalid("\(key) applies to --arm custom only")
            }
        }
        let units = try ComputeUnitsName.parse(try o.string("--compute-units", "cpuAndNeuralEngine"))
        let preUnits = try ComputeUnitsName.parse(try o.string("--preprocessor-units", "cpuOnly"))
        let c0Units = try ComputeUnitsName.parse(try o.string("--c0-compute-units", "cpuAndNeuralEngine"))
        let clipsURL = try o.url("--clips")
        let iosDir = o.values["--ios"].map { URL(fileURLWithPath: $0) } ?? clipsURL.deletingLastPathComponent()
        // Timing gate (DESIGN.md revision 8): a Core ML encoder arm needs a passing eligibility record.
        var eligibility: Any = "exempt: C0 is the product baseline"
        if armKind == "custom" {
            if let enc = o.values["--encoder"] {
                guard let id = o.values["--eligibility"], id.split(separator: ":").count == 2 else {
                    throw BenchError.invalid("--eligibility MODEL:ARM is required to time an encoder arm")
                }
                let parts = id.split(separator: ":").map(String.init)
                let spec = PipelineSpec(model: parts[0], arm: parts[1],
                                        variant: LengthVariant(rawValue: try o.string("--encoder-variant"))!,
                                        computeUnits: ComputeUnitsName.name(units), frontEnd: try o.string("--frontend"),
                                        decode: decodeName, encoderPath: URL(fileURLWithPath: enc),
                                        frontendConstants: o.values["--frontend-constants"].map { URL(fileURLWithPath: $0) },
                                        c0Dir: o.values["--models"].map { URL(fileURLWithPath: $0) },
                                        nativeWeights: o.values["--native-weights"].map { URL(fileURLWithPath: $0) },
                                        decoderModels: o.values["--decoder-models"].map { URL(fileURLWithPath: $0) })
                // the exact deployed combination needs its pipeline record; every component is hashed and compared
                eligibility = try PipelineEligibility.check(iosDir: iosDir, spec: spec)
            } else {
                eligibility = "n/a: gate harness (external encoder input), not an arm timing"
            }
        } else if o.values["--eligibility"] != nil || o.values["--pair-c0"] != nil {
            throw BenchError.invalid("--eligibility and --pair-c0 apply to --arm custom only")
        }
        guard (o.values["--pair-c0"] == nil) == (o.values["--c0-out"] == nil) else {
            throw BenchError.invalid("--pair-c0 C0DIR and --c0-out PATH go together")
        }
        let c0OutURL = try o.values["--c0-out"].map { try ArtifactPath.check(URL(fileURLWithPath: $0)) }
        let manifest = try ClipManifest.load(clipsURL)
        let pcmDir = try o.url("--pcm")
        let proto = RepetitionProtocol(warmups: try o.int("--warmups", 3), timed: try o.int("--timed", 10),
                                       emitWarmups: o.flags.contains("--emit-warmups"))
        let outURL = try ArtifactPath.check(try o.url("--out"))
        let diagDir = try o.values["--diag-dir"].map { try ArtifactPath.check(URL(fileURLWithPath: $0)) }
        let clips = try selectClips(o, manifest)
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
        let runner: Runner
        var load: [String: Any] = [:]
        var vocabulary: [Int: String] = [:]
        var armName = "C0"
        if armKind == "c0" {
            let models = try C0Models(directory: try o.url("--models"), computeUnits: units, preprocessorUnits: preUnits)
            runner = .c0(C0Pipeline(models: models), models)
            vocabulary = models.vocabulary
            load = ["load_ms": models.loadMs, "decode": "c0 (FluidAudio 0.7.8 loop, per-step Decoder + JointDecision)"]
        } else {
            let arm = try await buildArm(o, decodeName: decodeName, units: units, preUnits: preUnits, load: &load)
            armName = arm.name
            runner = .custom(arm)
            if let v = o.values["--vocab"] ?? o.values["--models"].map({ $0 + "/parakeet_vocab.json" }) {
                vocabulary = try loadVocabulary(URL(fileURLWithPath: v))
            }
        }
        var paired: (C0Pipeline, C0Models, JSONLWriter)? = nil
        if let c0Dir = o.values["--pair-c0"], let c0OutURL {
            let models = try C0Models(directory: URL(fileURLWithPath: c0Dir), computeUnits: c0Units, preprocessorUnits: preUnits)
            let w = try JSONLWriter(path: c0OutURL.path)
            paired = (C0Pipeline(models: models), models, w)
        }
        let footprintLoaded = physFootprint()
        let pairing: Any = paired == nil ? NSNull() : [
            "paired_with": armKind == "custom" ? "C0" : armName,
            "order": "per clip, C0 block and arm block (warm-ups + timed each) in the same process; C0 first on even "
                + "clip indices, the arm first on odd ones",
            "c0_out": c0OutURL?.lastPathComponent ?? "", "arm_out": outURL.lastPathComponent,
            "arm": armName, "session": UUID().uuidString]  // session: one id per process, shared by both files
        load["eligibility"] = eligibility
        load["pairing"] = pairing
        if let (_, models, w) = paired {
            try w.writeObject([
                "record": "load", "arm": "C0", "compute_units": ComputeUnitsName.name(c0Units),
                "preprocessor_units": ComputeUnitsName.name(preUnits), "load_ms": models.loadMs,
                "decode": "c0 (FluidAudio 0.7.8 loop, per-step Decoder + JointDecision)",
                "eligibility": "exempt: C0 is the product baseline", "pairing": pairing,
                "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "clip_ids": clips.map(\.id), "warmups": proto.warmups, "timed": proto.timed, "mode": modeName,
                "diagnostics": false, "clips_json_sha256": manifest.fileSHA256,
                "note": "loaded in the same process as the paired arm; phys_footprint covers both"])
        }
        try writer.writeObject(load.merging([
            "record": "load", "arm": armName, "compute_units": ComputeUnitsName.name(units),
            "preprocessor_units": ComputeUnitsName.name(preUnits),
            "load_cache_evidence": "none: MLModel(contentsOf:) wall time only; prepare-and-cache vs cached load needs "
                + "an Instruments Core ML trace (DESIGN.md Load)",
            "phys_footprint_mb_before": Double(footprint0.current) / 1_048_576,
            "phys_footprint_mb_after_load": Double(footprintLoaded.current) / 1_048_576,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "clip_ids": clips.map(\.id), "warmups": proto.warmups, "timed": proto.timed, "mode": modeName,
            "diagnostics": diagDir != nil,
            "clips_json_sha256": manifest.fileSHA256,
        ]) { _, new in new })
        let t0 = Clock.now()
        for (index, (clip, pcm)) in zip(clips, pcms).enumerated() {
            let mode: DecodeMode = modeName == "replay" ? .replay(traces[clip.id]!) : .free
            func block(_ r: Runner, _ w: JSONLWriter, _ name: String, _ vocab: [Int: String], _ cu: MLComputeUnits) async throws {
                for rep in 0..<(proto.warmups + proto.timed) {
                    let warm = rep < proto.warmups
                    let result: CallResult
                    switch r {
                    case .c0(let pipeline, _): result = try await pipeline.run(pcm, mode: mode)
                    case .custom(let arm): result = try await arm.run(clip: clip, pcm: pcm, mode: mode)
                    }
                    if !warm || proto.emitWarmups || rep == 0 {
                        try w.write(CallRecord(arm: name, mode: modeName, computeUnits: ComputeUnitsName.name(cu),
                                               clip: clip, rep: rep, warmup: warm, text: detokenize(result.tokens, vocabulary: vocab),
                                               result: result))
                    }
                }
            }
            if let (pipeline, models, w) = paired {
                if index % 2 == 0 {
                    try await block(.c0(pipeline, models), w, "C0", models.vocabulary, c0Units)
                    try await block(runner, writer, armName, vocabulary, units)
                } else {
                    try await block(runner, writer, armName, vocabulary, units)
                    try await block(.c0(pipeline, models), w, "C0", models.vocabulary, c0Units)
                }
            } else {
                try await block(runner, writer, armName, vocabulary, units)
            }
            if let diagDir {
                switch runner {
                case .c0(let pipeline, let models):
                    let diag = Diagnostics()
                    let result = try await pipeline.run(pcm, mode: mode, diagnostics: diag)
                    try writer.writeObject(try diagnosticRecord(clip: clip, mode: modeName, result: result, diag: diag,
                                                                trace: traces[clip.id], dir: diagDir, pipeline: pipeline,
                                                                vocabulary: models.vocabulary))
                case .custom(let arm):
                    let sink = DiagSink()
                    let result = try await arm.run(clip: clip, pcm: pcm, mode: mode, diag: sink)
                    try writer.writeObject(try customDiagnosticRecord(clip: clip, arm: arm, mode: modeName, result: result,
                                                                      sink: sink, trace: traces[clip.id], dir: diagDir,
                                                                      vocabulary: vocabulary))
                }
            }
        }
        let end = physFootprint()
        let endRecord: [String: Any] = [
            "record": "end", "seconds": Clock.ms(t0, Clock.now()) / 1000,
            "phys_footprint_mb": Double(end.current) / 1_048_576, "phys_footprint_peak_mb": Double(end.peak) / 1_048_576,
            "phys_footprint_note": paired == nil ? "this arm only" : "process with C0 and the arm loaded"]
        try writer.writeObject(endRecord)
        if let (_, _, w) = paired { try w.writeObject(endRecord) }
    }

    static func buildArm(_ o: Options, decodeName: String, units: MLComputeUnits, preUnits: MLComputeUnits,
                         load: inout [String: Any]) async throws -> ArmPipeline {
        let engine: DecodeEngine
        var t0 = Clock.now()
        switch decodeName {
        case "f2", "f1native":
            let weights = try NativeWeights(directory: try o.url("--native-weights"))
            engine = decodeName == "f2" ? NativeEngine(weights: weights) : NativeFusedEngine(weights: weights)
            load["native_weights"] = ["sha256": weights.manifest["sha256"] ?? "", "provenance": weights.manifest["provenance"] ?? [:]]
        case "f0":
            let dir = try o.url("--decoder-models")
            let config = C0Models.configuration(units)
            engine = try CoreMLStepEngine(decoder: try MLModel(contentsOf: dir.appendingPathComponent("Decoder.mlmodelc"), configuration: config),
                                          joint: try MLModel(contentsOf: dir.appendingPathComponent("JointDecision.mlmodelc"), configuration: config))
            load["decoder_models"] = dir.path
        default:  // f1
            let dir = try o.url("--decoder-models")
            var url = dir.appendingPathComponent("DecoderJoint.mlmodelc")
            if !FileManager.default.fileExists(atPath: url.path) {
                let pkg = dir.appendingPathComponent("DecoderJoint.mlpackage")
                let compiled = try await MLModel.compileModel(at: pkg)
                url = compiled
            }
            engine = try CoreMLFusedEngine(model: try MLModel(contentsOf: url, configuration: C0Models.configuration(units)))
            load["decoder_models"] = dir.path
        }
        load["decode_load_ms"] = Clock.ms(t0, Clock.now())
        var mel: MelInput? = nil
        var encoder: EncoderModel? = nil
        var external: URL? = nil
        var parts: [String] = []
        if let ext = o.values["--encoder-input"] {
            external = URL(fileURLWithPath: ext)
            parts = ["external-encoder"]
        } else {
            let fe = FrontEndKind(rawValue: try o.string("--frontend"))!
            mel = try MelInput(kind: fe, constantsDir: o.values["--frontend-constants"].map { URL(fileURLWithPath: $0) },
                               c0Dir: o.values["--models"].map { URL(fileURLWithPath: $0) }, preprocessorUnits: preUnits)
            let variant = LengthVariant(rawValue: try o.string("--encoder-variant"))!
            t0 = Clock.now()
            encoder = try await EncoderModel(url: try o.url("--encoder"), variant: variant, computeUnits: units,
                                             compiledDir: try ArtifactPath.check(ArtifactPath.macRoot.appendingPathComponent("compiled")))
            load["encoder"] = ["path": encoder!.url.path, "variant": variant.rawValue, "load_ms": encoder!.loadMs,
                               "compile_ms": encoder!.compileMs as Any, "total_ms": Clock.ms(t0, Clock.now())]
            parts = [fe.rawValue, (try o.string("--encoder") as NSString).lastPathComponent, variant.rawValue]
        }
        let name = o.values["--arm-name"] ?? (parts + [decodeName]).joined(separator: "+")
        load["arm_spec"] = ["front_end": mel?.kind.rawValue ?? "external", "decode": decodeName,
                            "encoder_input": external?.path as Any]
        return try ArmPipeline(name: name, mel: mel, encoder: encoder, externalEncoderDir: external, engine: engine)
    }

    /// Untimed diagnostic record of a custom arm; float sections go to <dir>/<clip>.<arm>.<mode>.diag.f32.
    static func customDiagnosticRecord(clip: Clip, arm: ArmPipeline, mode: String, result: CallResult, sink: DiagSink,
                                       trace: TraceClip?, dir: URL, vocabulary: [Int: String]) throws -> [String: Any] {
        let (data, index) = sink.binary()
        let safe = arm.name.replacingOccurrences(of: "/", with: "_")
        let file = dir.appendingPathComponent("\(clip.id).\(safe).\(mode).diag.f32")
        try data.write(to: file)
        var steps: [String: Any] = sink.ints
        for (k, v) in sink.floats { steps[k] = v }
        var record: [String: Any] = [
            "record": "diagnostic", "arm": arm.name, "decode": arm.engine.name, "mode": mode, "clip": clip.id,
            "kind": clip.kind, "bucket": result.bucket, "timed": false, "tokens": result.tokens,
            "text": detokenize(result.tokens, vocabulary: vocabulary), "encoder_length": result.encoderLength,
            "mel_length": result.melLength, "physical_calls": result.physicalCalls, "steps": steps,
            "arrays": ["file": file.lastPathComponent, "bytes": data.count, "sha256": sha256Hex(data),
                       "dtype": "float32 little-endian", "sections": index],
        ]
        if arm.engine is NativeEngine {
            record["logits"] = "section logits [steps, 1030] (raw joint output: 1,025 token+blank, 5 duration)"
        } else {
            record["logits"] = NSNull()
            record["logits_unavailable"] = "Core ML per-step models with C0's contract output only argmax decisions"
        }
        if let trace, let tok = sink.ints["argmax_token"], let dur = sink.ints["argmax_duration"] {
            var ta = 0, da = 0
            for i in 0..<min(tok.count, trace.steps) {
                if tok[i] == trace.token[i] { ta += 1 }
                if dur[i] == trace.duration[i] { da += 1 }
            }
            record["replay"] = ["steps": tok.count, "trace_steps": trace.steps, "token_agree": ta, "duration_agree": da]
        }
        return record
    }

    /// Deployed-pipeline gate run (WP7): untimed; see pipegate.py for the conditions.
    static func gate(_ o: Options) async throws {
        let units = try ComputeUnitsName.parse(try o.string("--compute-units", "cpuAndNeuralEngine"))
        guard let variant = LengthVariant(rawValue: try o.string("--encoder-variant")) else {
            throw BenchError.invalid("unknown --encoder-variant")
        }
        let id = try o.string("--eligibility").split(separator: ":").map(String.init)
        guard id.count == 2 else { throw BenchError.invalid("--eligibility MODEL:ARM") }
        let clipsURL = try o.url("--clips")
        let manifest = try ClipManifest.load(clipsURL)
        let clips = try selectClips(o, manifest)
        let probe = TdtConfig()
        let traces = try TraceFile.load(try o.url("--traces")).validated(
            for: manifest.clips, manifest: manifest, blank: probe.blankId, durations: probe.durationBins,
            maxSymbols: probe.maxSymbolsPerStep)
        let out = try ArtifactPath.check(try o.url("--out"))
        try FileManager.default.createDirectory(at: out.appendingPathComponent("enc"), withIntermediateDirectories: true)
        let decodes = try o.string("--decodes", "f2,f0,f1").split(separator: ",").map(String.init)
        let encURL = try o.url("--encoder")
        var components: [String: Any] = [:]
        var engines: [(String, DecodeEngine)] = []
        let config = C0Models.configuration(units)
        for d in decodes {
            let spec = PipelineSpec(model: id[0], arm: id[1], variant: variant, computeUnits: ComputeUnitsName.name(units),
                                    frontEnd: "vdsp", decode: d, encoderPath: encURL,
                                    frontendConstants: try o.url("--frontend-constants"), c0Dir: nil,
                                    nativeWeights: o.values["--native-weights"].map { URL(fileURLWithPath: $0) },
                                    decoderModels: o.values["--decoder-models"].map { URL(fileURLWithPath: $0) })
            components[d] = ["record": spec.recordName, "components": try spec.components()]
            switch d {
            case "f2": engines.append((d, NativeEngine(weights: try NativeWeights(directory: try o.url("--native-weights")))))
            case "f0":
                let dir = try o.url("--decoder-models")
                engines.append((d, try CoreMLStepEngine(decoder: try MLModel(contentsOf: dir.appendingPathComponent("Decoder.mlmodelc"), configuration: config),
                                                        joint: try MLModel(contentsOf: dir.appendingPathComponent("JointDecision.mlmodelc"), configuration: config))))
            case "f1":
                let dir = try o.url("--decoder-models")
                engines.append((d, try CoreMLFusedEngine(model: try MLModel(contentsOf: dir.appendingPathComponent("DecoderJoint.mlmodelc"), configuration: config))))
            default: throw BenchError.invalid("unknown decode \(d)")
            }
        }
        let mel = try MelInput(kind: .vdsp, constantsDir: try o.url("--frontend-constants"), c0Dir: nil, preprocessorUnits: .cpuOnly)
        let t0 = Clock.now()
        let encoder = try await EncoderModel(url: encURL, variant: variant, computeUnits: units, compiledDir: nil)
        let loadMs = Clock.ms(t0, Clock.now())
        let writer = try JSONLWriter(path: out.appendingPathComponent("gate.jsonl").path)
        try writer.writeObject(["record": "header", "model": id[0], "arm": id[1], "variant": Eligibility.variants[variant] ?? "",
                                "compute_units": ComputeUnitsName.name(units), "front_end": "vdsp", "decodes": decodes,
                                "pipelines": components, "encoder_load_ms": loadMs, "clip_ids": clips.map(\.id),
                                "clips_json_sha256": manifest.fileSHA256, "os": ProcessInfo.processInfo.operatingSystemVersionString])
        let loop = LabelLoop()
        let options = MLPredictionOptions()
        for clip in clips {
            let pcm = try readPCM(directory: try o.url("--pcm"), clip: clip)
            let bucket = try encoder.bucket(forSamples: pcm.count)
            let (melArray, melLength) = try await mel.mel(pcm, bucket: bucket)
            let (encOut, length) = try await encoder.predict(mel: melArray, melLength: melLength, bucket: bucket, options: options)
            let frames = try EncoderFrames(encOut, validLength: length)
            let enc = try frames.timeMajor()
            let data = enc.withUnsafeBufferPointer { Data(buffer: $0) }
            try data.write(to: out.appendingPathComponent("enc/\(clip.id).f32"))
            var rec: [String: Any] = ["record": "clip", "clip": clip.id, "kind": clip.kind, "bucket": bucket,
                                      "mel_length": melLength, "encoder_length": frames.count,
                                      "finite": enc.allSatisfy { $0.isFinite }, "enc_sha256": sha256Hex(data)]
            let trace = traces[clip.id]!
            for (d, engine) in engines {
                var r: [String: Any] = [:]
                try engine.begin(frames: frames, length: frames.count, diag: nil)
                let free = try loop.free(engine, length: frames.count, diag: nil)
                r["tokens"] = free.tokens
                r["free_steps"] = free.steps
                if frames.count == trace.numFrames {
                    let sink = DiagSink()
                    try engine.begin(frames: frames, length: frames.count, diag: nil)
                    let rep = try loop.replay(engine, trace: trace, diag: sink)
                    r["argmax_token"] = sink.ints["argmax_token"] ?? []
                    r["argmax_duration"] = sink.ints["argmax_duration"] ?? []
                    r["replay_steps"] = rep.steps
                    r["replay_predictions"] = rep.predictions
                } else {
                    r["replay_error"] = "encoder frames \(frames.count) != trace \(trace.numFrames)"
                }
                rec[d] = r
            }
            try writer.writeObject(rec)
        }
        try writer.writeObject(["record": "end", "seconds": Clock.ms(t0, Clock.now()) / 1000])
    }

    /// Gate 5 (rev. 5), encoder part: the encoder on externally computed features (untimed).
    static func encode(_ o: Options) async throws {
        let units = try ComputeUnitsName.parse(try o.string("--compute-units", "cpuAndNeuralEngine"))
        guard let variant = LengthVariant(rawValue: try o.string("--encoder-variant", "fixed15")) else {
            throw BenchError.invalid("unknown --encoder-variant")
        }
        let manifest = try ClipManifest.load(try o.url("--clips"))
        let clips = try selectClips(o, manifest)
        let featDir = try o.url("--features")
        let tags = try o.string("--tags", "vdsp").split(separator: ",").map(String.init)
        let out = try ArtifactPath.check(try o.url("--out"))
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let encoder = try await EncoderModel(url: try o.url("--encoder"), variant: variant, computeUnits: units,
                                             compiledDir: try ArtifactPath.check(ArtifactPath.macRoot.appendingPathComponent("compiled")))
        let writer = try JSONLWriter(path: out.appendingPathComponent("encode.jsonl").path)
        let options = MLPredictionOptions()
        for clip in clips {
            let bucket = try encoder.bucket(forSamples: clip.length)
            let fB = Buckets.melFrames(bucket), frames = clip.length / 160 + 1
            for tag in tags {
                let name = tag == "vdsp" ? "\(clip.id).mel.f32" : "\(clip.id).\(tag).mel.f32"
                let data = try Data(contentsOf: featDir.appendingPathComponent(name))
                guard data.count == 128 * frames * 4, frames <= fB else { throw BenchError.invalid("\(name): size") }
                let mel = try MLMultiArray(shape: [1, 128, NSNumber(value: fB)], dataType: .float32)
                let dst = mel.dataPointer.bindMemory(to: Float.self, capacity: 128 * fB)
                dst.initialize(repeating: 0, count: 128 * fB)
                data.withUnsafeBytes { raw in
                    let src = raw.bindMemory(to: Float.self).baseAddress!
                    for c in 0..<128 { (dst + c * fB).update(from: src + c * frames, count: frames) }
                }
                let (enc, length) = try await encoder.predict(mel: mel, melLength: clip.melFrames, bucket: bucket, options: options)
                let view = try EncoderFrames(enc, validLength: length)
                let values = try view.timeMajor()
                let bytes = values.withUnsafeBufferPointer { Data(buffer: $0) }
                try bytes.write(to: out.appendingPathComponent("\(clip.id).\(tag).enc.f32"))
                try writer.writeObject(["id": "\(clip.id).\(tag)", "clip": clip.id, "tag": tag, "bucket": bucket,
                                        "mel_length": clip.melFrames, "encoder_length": view.count,
                                        "sha256": sha256Hex(bytes)])
            }
        }
    }

    /// Front end A on every selected clip, for the gate against reference.Featurizer (native.py gate-frontend).
    static func features(_ o: Options) async throws {
        let fe = try VDSPFrontEnd(constantsDir: try o.url("--frontend-constants"))
        let manifest = try ClipManifest.load(try o.url("--clips"))
        let clips = try selectClips(o, manifest)
        let out = try ArtifactPath.check(try o.url("--out"))
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let writer = try JSONLWriter(path: out.appendingPathComponent("features.jsonl").path)
        let pcmDir = try o.url("--pcm")
        for clip in clips {
            let pcm = try readPCM(directory: pcmDir, clip: clip)
            let t0 = Clock.now()
            let (features, frames, valid) = fe.compute(pcm)
            let ms = Clock.ms(t0, Clock.now())
            let data = features.withUnsafeBufferPointer { Data(buffer: $0) }
            try data.write(to: out.appendingPathComponent("\(clip.id).mel.f32"))
            try writer.writeObject(["clip": clip.id, "frames": frames, "mel_length": valid, "ms": ms,
                                    "sha256": sha256Hex(data)])
        }
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
