import Foundation

/// DESIGN.md "Repetition and statistics": per clip and arm, `warmups` untimed calls, then `timed` calls
/// (10 per clip on the Mac, 5 on the phone). Statistics (median of per-clip medians, Harrell-Davis p95) are
/// computed offline from the emitted records.
public struct RepetitionProtocol: Sendable {
    public var warmups: Int
    public var timed: Int
    public var emitWarmups: Bool

    public init(warmups: Int = 3, timed: Int = 10, emitWarmups: Bool = false) {
        self.warmups = warmups; self.timed = timed; self.emitWarmups = emitWarmups
    }
}

/// One JSON line per call.
public struct CallRecord: Encodable {
    public let arm: String
    public let mode: String
    public let computeUnits: String
    public let clip: String
    public let kind: String
    public let bucket: Int
    public let samples: Int
    public let rep: Int
    public let warmup: Bool
    public let text: String
    public let result: CallResult
    public let physFootprintMB: Double

    enum CodingKeys: String, CodingKey {
        case arm, mode, clip, kind, bucket, samples, rep, warmup, text, result
        case computeUnits = "compute_units", physFootprintMB = "phys_footprint_mb"
    }

    public init(arm: String, mode: String, computeUnits: String, clip: Clip, rep: Int, warmup: Bool, text: String,
                result: CallResult) {
        self.arm = arm; self.mode = mode; self.computeUnits = computeUnits; self.clip = clip.id; self.kind = clip.kind
        self.bucket = clip.bucket; self.samples = clip.length; self.rep = rep; self.warmup = warmup; self.text = text
        self.result = result
        self.physFootprintMB = Double(physFootprint().current) / 1_048_576
    }
}

/// Writes JSON lines to a file (or stdout).
public final class JSONLWriter {
    let handle: FileHandle
    let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        e.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return e
    }()

    public init(path: String?) throws {
        if let path {
            FileManager.default.createFile(atPath: path, contents: nil)
            handle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
        } else {
            handle = FileHandle.standardOutput
        }
    }

    public func write<T: Encodable>(_ value: T) throws {
        var data = try encoder.encode(value)
        data.append(0x0A)
        handle.write(data)
    }

    public func writeObject(_ object: [String: Any]) throws {
        var data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
        data.append(0x0A)
        handle.write(data)
    }
}
