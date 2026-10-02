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

    public static func load(_ url: URL) throws -> ClipManifest {
        try JSONDecoder().decode(ClipManifest.self, from: Data(contentsOf: url))
    }
}

/// 16 kHz float32 PCM of a clip (<dir>/<id>.f32, written by `clips.py materialize`), SHA-256 checked.
public func readPCM(directory: URL, clip: Clip) throws -> [Float] {
    let data = try Data(contentsOf: directory.appendingPathComponent("\(clip.id).f32"))
    let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    guard digest == clip.sha256, data.count == clip.length * 4 else {
        throw BenchError.invalid("\(clip.id): PCM SHA-256 or length differs from clips.json")
    }
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

/// One clip of traces.json (ios/traces.py): the B0 FP32 reference's complete greedy TDT trace.
public struct TraceClip: Decodable, Sendable {
    public let id: String
    public let numFrames: Int
    public let steps: Int
    public let frame: [Int]
    public let predInput: [Int]
    public let token: [Int]
    public let duration: [Int]
    public let emitted: [Int]
    public let predUpdated: [Int]
    public let tokens: [Int]
    public let text: String

    enum CodingKeys: String, CodingKey {
        case id, steps, frame, token, duration, emitted, tokens, text
        case numFrames = "num_frames", predInput = "pred_input", predUpdated = "pred_updated"
    }
}

public struct TraceFile: Decodable, Sendable {
    public let clips: [TraceClip]

    public static func load(_ url: URL) throws -> [String: TraceClip] {
        let file = try JSONDecoder().decode(TraceFile.self, from: Data(contentsOf: url))
        return Dictionary(uniqueKeysWithValues: file.clips.map { ($0.id, $0) })
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
