import BenchCore
import CoreML
import Foundation

/// parakeet-bench: macOS command-line driver of BenchCore (run on the Mac through ios/macguard).
///
///   parakeet-bench run  --models DIR --clips clips.json --pcm DIR [--traces traces.json] [--mode free|replay]
///                       [--compute-units cpuAndNeuralEngine] [--preprocessor-units cpuOnly] [--kinds natural,...]
///                       [--ids a,b] [--limit N] [--warmups 3] [--timed 10] [--emit-warmups] [--no-steps]
///                       [--out results.jsonl]
///   parakeet-bench plan --models DIR [--compute-units cpuAndNeuralEngine] [--preprocessor-units cpuOnly] --out DIR
///   parakeet-bench info --models DIR
@main
struct ParakeetBenchCLI {
    static func main() async {
        do {
            var args = Array(CommandLine.arguments.dropFirst())
            guard let command = args.first else { throw BenchError.invalid(usage) }
            args.removeFirst()
            let options = try Options(args)
            switch command {
            case "run": try await run(options)
            case "plan": try await plan(options)
            case "info": try info(options)
            default: throw BenchError.invalid(usage)
            }
        } catch {
            FileHandle.standardError.write("parakeet-bench: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }

    static let usage = "usage: parakeet-bench run|plan|info --models DIR ... (see ParakeetBenchCLI.swift)"

    struct Options {
        var values: [String: String] = [:]
        var flags: Set<String> = []
        static let flagNames: Set<String> = ["--emit-warmups", "--no-steps"]

        init(_ args: [String]) throws {
            var i = 0
            while i < args.count {
                let a = args[i]
                if Self.flagNames.contains(a) { flags.insert(a); i += 1; continue }
                guard a.hasPrefix("--"), i + 1 < args.count else { throw BenchError.invalid("bad argument \(a)") }
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
            guard let v = Int(s) else { throw BenchError.invalid("\(key) must be an integer") }
            return v
        }

        func url(_ key: String) throws -> URL { URL(fileURLWithPath: try string(key)) }
    }

    static func run(_ o: Options) async throws {
        let units = try ComputeUnitsName.parse(try o.string("--compute-units", "cpuAndNeuralEngine"))
        let preUnits = try ComputeUnitsName.parse(try o.string("--preprocessor-units", "cpuOnly"))
        let manifest = try ClipManifest.load(try o.url("--clips"))
        let pcmDir = try o.url("--pcm")
        let modeName = try o.string("--mode", "free")
        let traces: [String: TraceClip] = modeName == "replay" ? try TraceFile.load(try o.url("--traces")) : [:]
        let proto = RepetitionProtocol(warmups: try o.int("--warmups", 3), timed: try o.int("--timed", 10),
                                       emitWarmups: o.flags.contains("--emit-warmups"))
        var clips = manifest.clips
        if let kinds = o.values["--kinds"] {
            let set = Set(kinds.split(separator: ",").map(String.init))
            clips = clips.filter { set.contains($0.kind) }
        }
        if let ids = o.values["--ids"] {
            let set = Set(ids.split(separator: ",").map(String.init))
            clips = clips.filter { set.contains($0.id) }
        }
        if let limit = o.values["--limit"], let n = Int(limit) { clips = Array(clips.prefix(n)) }
        let writer = try JSONLWriter(path: o.values["--out"])

        let footprint0 = physFootprint()
        let models = try C0Models(directory: try o.url("--models"), computeUnits: units, preprocessorUnits: preUnits)
        let footprintLoaded = physFootprint()
        let pipeline = C0Pipeline(models: models)
        pipeline.recordSteps = !o.flags.contains("--no-steps")
        try writer.writeObject([
            "record": "load", "arm": "C0", "compute_units": ComputeUnitsName.name(units),
            "preprocessor_units": ComputeUnitsName.name(preUnits), "load_ms": models.loadMs,
            "phys_footprint_mb_before": Double(footprint0.current) / 1_048_576,
            "phys_footprint_mb_after_load": Double(footprintLoaded.current) / 1_048_576,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "clips": clips.count, "warmups": proto.warmups, "timed": proto.timed, "mode": modeName,
        ])
        let t0 = Clock.now()
        for clip in clips {
            let pcm = try readPCM(directory: pcmDir, clip: clip)
            let mode: DecodeMode
            if modeName == "replay" {
                guard let trace = traces[clip.id] else { throw BenchError.invalid("no trace for \(clip.id)") }
                mode = .replay(trace)
            } else {
                mode = .free
            }
            for rep in 0..<(proto.warmups + proto.timed) {
                let warm = rep < proto.warmups
                let result = try await pipeline.run(pcm, mode: mode)
                if !warm || proto.emitWarmups || rep == 0 {
                    let text = detokenize(result.tokens, vocabulary: models.vocabulary)
                    try writer.write(CallRecord(arm: "C0", mode: modeName, computeUnits: ComputeUnitsName.name(units),
                                                clip: clip, rep: rep, warmup: warm, text: text, result: result))
                }
            }
        }
        let end = physFootprint()
        try writer.writeObject([
            "record": "end", "seconds": Clock.ms(t0, Clock.now()) / 1000,
            "phys_footprint_mb": Double(end.current) / 1_048_576, "phys_footprint_peak_mb": Double(end.peak) / 1_048_576,
        ])
    }

    static func plan(_ o: Options) async throws {
        let units = try ComputeUnitsName.parse(try o.string("--compute-units", "cpuAndNeuralEngine"))
        let preUnits = try ComputeUnitsName.parse(try o.string("--preprocessor-units", "cpuOnly"))
        let models = try o.url("--models")
        let out = try o.url("--out")
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
