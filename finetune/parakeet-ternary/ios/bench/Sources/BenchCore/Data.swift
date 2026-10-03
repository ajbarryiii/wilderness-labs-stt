import CryptoKit
import Foundation

public enum BenchError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self {
        case .invalid(let message): return message
        }
    }
}

public func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

/// One entry of clips.json (written by ios/clips.py).
public struct Clip: Decodable, Sendable {
    public let id: String
    public let kind: String
    public let bucket: Int
    public let length: Int
    public let melFrames: Int
    public let encoderFrames: Int
    public let fluidaudioFrames: Int
    public let transcript: String?
    public let sha256: String

    enum CodingKeys: String, CodingKey {
        case id, kind, bucket, length, transcript, sha256
        case melFrames = "mel_frames", encoderFrames = "encoder_frames", fluidaudioFrames = "fluidaudio_frames"
    }
}

public struct ClipManifest: Decodable, Sendable {
    public let clips: [Clip]
    /// SHA-256 of the clips.json bytes this manifest was read from (traces.json records the one it was made with).
    public private(set) var fileSHA256 = ""

    enum CodingKeys: String, CodingKey { case clips }

    public static func load(_ url: URL) throws -> ClipManifest {
        let data = try Data(contentsOf: url)
        var manifest = try JSONDecoder().decode(ClipManifest.self, from: data)
        manifest.fileSHA256 = sha256Hex(data)
        return manifest
    }
}

/// 16 kHz float32 PCM of a clip (<dir>/<id>.f32, written by `clips.py materialize`), SHA-256 checked.
public func readPCM(directory: URL, clip: Clip) throws -> [Float] {
    let data = try Data(contentsOf: directory.appendingPathComponent("\(clip.id).f32"))
    guard sha256Hex(data) == clip.sha256, data.count == clip.length * 4 else {
        throw BenchError.invalid("\(clip.id): PCM SHA-256 or length differs from clips.json")
    }
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

/// One clip of traces.json (ios/traces.py): the B0 FP32 reference's complete greedy TDT trace.
public struct TraceClip: Decodable, Sendable {
    public let id: String
    public let sha256: String
    public let samples: Int
    public let melFrames: Int
    public let numFrames: Int
    public let steps: Int
    public let frame: [Int]
    public let predInput: [Int]
    public let token: [Int]
    public let duration: [Int]
    public let emitted: [Int]
    public let predUpdated: [Int]
    public let symbolsAtFrame: [Int]
    public let forcedAdvance: [Int]
    public let advance: [Int]
    public let tokens: [Int]
    public let text: String

    enum CodingKeys: String, CodingKey {
        case id, sha256, samples, steps, frame, token, duration, emitted, advance, tokens, text
        case melFrames = "mel_frames", numFrames = "num_frames", predInput = "pred_input", predUpdated = "pred_updated"
        case symbolsAtFrame = "symbols_at_frame", forcedAdvance = "forced_advance"
    }
}

public struct TraceDecoding: Decodable, Sendable {
    public let maxSymbols: Int
    public let durations: [Int]
    public let blank: Int
    enum CodingKeys: String, CodingKey { case durations, blank; case maxSymbols = "max_symbols" }
}

public struct TraceFile: Decodable, Sendable {
    public let clipsJSONSHA256: String
    public let decoding: TraceDecoding
    public let clips: [TraceClip]

    enum CodingKeys: String, CodingKey { case decoding, clips; case clipsJSONSHA256 = "clips_json_sha256" }

    public static func load(_ url: URL) throws -> TraceFile {
        try JSONDecoder().decode(TraceFile.self, from: Data(contentsOf: url))
    }

    /// Traces of `clips`, each validated by `validate` against this manifest; throws on the first problem.
    public func validated(for clips: [Clip], manifest: ClipManifest, blank: Int, durations: [Int],
                          maxSymbols: Int) throws -> [String: TraceClip] {
        guard clipsJSONSHA256 == manifest.fileSHA256 else {
            throw BenchError.invalid("traces.json was made from clips.json \(clipsJSONSHA256), not \(manifest.fileSHA256)")
        }
        guard decoding.blank == blank, decoding.durations == durations, decoding.maxSymbols == maxSymbols else {
            throw BenchError.invalid("traces.json decoding constants differ from the harness's")
        }
        let byID = Dictionary(uniqueKeysWithValues: self.clips.map { ($0.id, $0) })
        var out: [String: TraceClip] = [:]
        for clip in clips {
            guard let trace = byID[clip.id] else { throw BenchError.invalid("no trace for \(clip.id)") }
            try Self.validate(trace, clip: clip, blank: blank, durations: durations, maxSymbols: maxSymbols)
            out[clip.id] = trace
        }
        return out
    }

    /// Provenance, lengths and every invariant of the reference's label-looping step sequence
    /// (reference.run_steps): the trace must be exactly what that loop produces from its own decisions.
    public static func validate(_ t: TraceClip, clip: Clip, blank: Int, durations: [Int], maxSymbols: Int) throws {
        func fail(_ what: String) -> BenchError { BenchError.invalid("trace \(t.id): \(what)") }
        guard t.sha256 == clip.sha256, t.samples == clip.length, t.melFrames == clip.melFrames,
              t.numFrames == clip.encoderFrames else { throw fail("provenance/length differs from clips.json") }
        let n = t.steps
        for (name, a) in [("frame", t.frame), ("pred_input", t.predInput), ("token", t.token), ("duration", t.duration),
                          ("emitted", t.emitted), ("pred_updated", t.predUpdated), ("symbols_at_frame", t.symbolsAtFrame),
                          ("forced_advance", t.forcedAdvance), ("advance", t.advance)] where a.count != n {
            throw fail("\(name) has \(a.count) entries, steps = \(n)")
        }
        let length = t.numFrames
        guard length > 0 else { throw fail("no frames") }
        var time = 0, lastNB = -1, lasts = 0, predInput = blank
        var tokens: [Int] = []
        for i in 0..<n {
            guard time < length else { throw fail("step \(i) after the last frame") }
            guard t.frame[i] == time else { throw fail("step \(i): frame \(t.frame[i]) != \(time)") }
            guard t.predInput[i] == predInput else { throw fail("step \(i): pred_input") }
            guard t.token[i] >= 0 && t.token[i] <= blank else { throw fail("step \(i): token out of range") }
            guard durations.contains(t.duration[i]) else { throw fail("step \(i): duration \(t.duration[i])") }
            let emitted = t.token[i] != blank
            guard t.emitted[i] == (emitted ? 1 : 0), t.predUpdated[i] == t.emitted[i] else {
                throw fail("step \(i): emitted/pred_updated")
            }
            var advance = (!emitted && t.duration[i] == 0) ? 1 : t.duration[i]
            var forced = false
            if emitted {
                lasts = lastNB == time ? lasts + 1 : 1
                lastNB = time
                if time + advance < length && lasts >= maxSymbols && lastNB == time + advance {
                    advance += 1
                    forced = true
                }
                tokens.append(t.token[i])
                predInput = t.token[i]
            }
            guard t.symbolsAtFrame[i] == lasts, t.forcedAdvance[i] == (forced ? 1 : 0), t.advance[i] == advance else {
                throw fail("step \(i): symbols_at_frame/forced_advance/advance")
            }
            time += advance
        }
        guard time >= length else { throw fail("ends at frame \(time) before \(length)") }
        guard tokens == t.tokens else { throw fail("tokens differ from the emitted steps") }
    }
}

/// parakeet_vocab.json ({"id": "piece"}), as FluidAudio's AsrModels.loadVocabulary reads it.
public func loadVocabulary(_ url: URL) throws -> [Int: String] {
    guard let dict = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: String] else {
        throw BenchError.invalid("vocabulary is not a {String: String} object")
    }
    var vocab: [Int: String] = [:]
    for (key, value) in dict { if let id = Int(key) { vocab[id] = value } }
    return vocab
}

/// FluidAudio 0.7.8 AsrManager.convertTokensWithExistingTimings: concatenate pieces, "▁" -> " ", trim.
public func detokenize(_ tokens: [Int], vocabulary: [Int: String]) -> String {
    let pieces = tokens.compactMap { id -> String? in
        guard let piece = vocabulary[id], !piece.isEmpty else { return nil }
        return piece
    }
    return pieces.joined().replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
}

/// Where the macOS CLI may write (mirrors ios/artifacts.py check()): under the Mac artifact root and
/// outside any Git work tree. Audio, weights and weight excerpts never land in the repository.
public enum ArtifactPath {
    public static let macRoot = URL(fileURLWithPath: "/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios")

    public static func check(_ url: URL) throws -> URL {
        let resolved = url.standardizedFileURL.resolvingSymlinksInPath()
        let root = macRoot.standardizedFileURL.resolvingSymlinksInPath().path
        guard resolved.path == root || resolved.path.hasPrefix(root + "/") else {
            throw BenchError.invalid("refusing to write \(resolved.path): artifacts must live under \(root)")
        }
        var dir = resolved
        while dir.path != "/" {
            if FileManager.default.fileExists(atPath: dir.appendingPathComponent(".git").path) {
                throw BenchError.invalid("refusing to write \(resolved.path): inside a Git work tree")
            }
            dir = dir.deletingLastPathComponent()
        }
        return resolved
    }
}

/// Timing gate (DESIGN.md revision 8; WP3's mil/eligibility.py check()): an encoder arm may be timed only with a
/// passing eligibility record results/eligibility/<model>-<arm>-<variant>-<backend>.json of the required design
/// revision whose input files are unchanged (SHA-256 recomputed here). C0 is the product baseline and exempt.
public enum Eligibility {
    public static let requiredRevision = 8
    public static let variants: [LengthVariant: String] = [.fixed15: "fixed", .multifunction: "multi", .enumerated: "enum"]

    public static func backend(_ units: String) -> String? {
        ["cpuAndNeuralEngine": "ane", "cpuOnly": "cpu", "cpuAndGPU": "gpu"][units]
    }

    /// The record (as a dictionary) or an error explaining why the arm may not be timed.
    public static func check(iosDir: URL, model: String, arm: String, variant: LengthVariant, computeUnits: String,
                             encoderPath: URL) throws -> [String: Any] {
        guard let v = variants[variant], let b = backend(computeUnits) else {
            throw BenchError.invalid("no eligibility backend for compute units \(computeUnits)")
        }
        let name = "\(model)-\(arm)-\(v)-\(b).json"
        let url = iosDir.appendingPathComponent("results/eligibility/\(name)")
        guard let data = try? Data(contentsOf: url),
              let rec = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw BenchError.invalid("refusing to time \(model)/\(arm)/\(v)/\(b): no eligibility record \(name)")
        }
        guard (rec["design_revision"] as? Int) == requiredRevision else {
            throw BenchError.invalid("refusing to time: \(name) is for design revision \(rec["design_revision"] ?? "?"), not \(requiredRevision)")
        }
        guard rec["model"] as? String == model, rec["arm"] as? String == arm, rec["variant"] as? String == v,
              rec["backend"] as? String == b, rec["compute_units"] as? String == computeUnits else {
            throw BenchError.invalid("refusing to time: \(name) does not describe this arm/backend")
        }
        let expected = "/\(model)/\(arm)/\(v).mlmodelc"
        guard encoderPath.standardizedFileURL.path.hasSuffix(expected) else {
            throw BenchError.invalid("refusing to time: encoder \(encoderPath.path) is not the record's model (*\(expected))")
        }
        var stale: [String] = []
        for (file, digest) in (rec["inputs"] as? [String: String]) ?? [:] {
            let d = try? Data(contentsOf: iosDir.appendingPathComponent(file))
            if d == nil || sha256Hex(d!) != digest { stale.append(file) }
        }
        guard stale.isEmpty else { throw BenchError.invalid("refusing to time: \(name) inputs changed: \(stale.prefix(5))") }
        guard rec["timing_allowed"] as? Bool == true else {
            let reasons = (rec["reasons"] as? [String])?.joined(separator: "; ") ?? "?"
            throw BenchError.invalid("refusing to time \(model)/\(arm)/\(v)/\(b): not eligible: \(reasons)")
        }
        return ["record": name, "design_revision": requiredRevision, "eligible": rec["eligible"] ?? NSNull(),
                "selection_eligible": rec["selection_eligible"] ?? NSNull(), "timing_allowed": true,
                "built": rec["built"] ?? NSNull()]
    }
}
