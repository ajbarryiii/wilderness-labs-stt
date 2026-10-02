import Accelerate
import CoreML
import Foundation

/// Float storage that outlives the views onto it (external encoder outputs, native buffers).
public final class FloatBuffer {
    public let pointer: UnsafeMutablePointer<Float>
    public let count: Int
    public init(count: Int) {
        self.count = count
        pointer = .allocate(capacity: max(count, 1))
        pointer.initialize(repeating: 0, count: max(count, 1))
    }
    public convenience init(_ values: [Float]) {
        self.init(count: values.count)
        values.withUnsafeBufferPointer { pointer.update(from: $0.baseAddress!, count: values.count) }
    }
    deinit { pointer.deallocate() }
}

/// Stride-aware view of encoder frames: the Core ML output [1, 1024, T] or [1, T, 1024] (FluidAudio 0.7.8
/// EncoderFrameView), or an external time-major [T, 1024] buffer (the reference's encoder output, F2 gate).
public struct EncoderFrames {
    public let count: Int
    public let hiddenStride: Int
    public let timeStride: Int
    public let base: UnsafeMutablePointer<Float>
    let owner: AnyObject

    public init(_ output: MLMultiArray, validLength: Int) throws {
        let shape = output.intShape, strides = output.intStrides
        guard shape.count == 3, shape[0] == 1, shape[1] == 1024 || shape[2] == 1024, output.dataType == .float32 else {
            throw BenchError.invalid("unexpected encoder output \(shape) \(output.dataType.rawValue)")
        }
        let hiddenAxis = shape[1] == 1024 ? 1 : 2
        let timeAxis = 3 - hiddenAxis
        hiddenStride = strides[hiddenAxis]
        timeStride = strides[timeAxis]
        count = min(validLength, shape[timeAxis])
        guard count > 0, timeStride > 0 else { throw BenchError.invalid("encoder output has no frames") }
        owner = output
        base = output.dataPointer.bindMemory(to: Float.self, capacity: output.count)
    }

    public init(timeMajor buffer: FloatBuffer, frames: Int) throws {
        guard frames > 0, buffer.count == frames * 1024 else { throw BenchError.invalid("external encoder output size") }
        count = frames; hiddenStride = 1; timeStride = 1024; base = buffer.pointer; owner = buffer
    }

    public func copyFrame(_ t: Int, into dest: UnsafeMutablePointer<Float>, destStride: Int) throws {
        guard t >= 0 && t < count else { throw BenchError.invalid("encoder frame \(t) out of range \(count)") }
        let src = base.advanced(by: t * timeStride)
        if hiddenStride == 1 && destStride == 1 {
            dest.update(from: src, count: 1024)
        } else {
            cblas_scopy(1024, src, Int32(hiddenStride), dest, Int32(destStride))
        }
    }

    /// Time-major copy of the valid frames [count, 1024] (diagnostics).
    public func timeMajor() throws -> [Float] {
        var out = [Float](repeating: 0, count: count * 1024)
        try out.withUnsafeMutableBufferPointer { buf in
            for t in 0..<count { try copyFrame(t, into: buf.baseAddress! + t * 1024, destStride: 1) }
        }
        return out
    }
}

/// Bucketed encoder input contract (WP3 ios/mil/contract.json): buckets of 2, 4, 8 and 15 s; mel input
/// [1, 128, F_b] with F_b = 100 b + 1 (201, 401, 801, 1501); outputs encoder [1, 1024, T_b], encoder_length [1].
public enum Buckets {
    public static let seconds = [2, 4, 8, 15]
    public static func melFrames(_ b: Int) -> Int { 16000 * b / 160 + 1 }
    /// Smallest bucket holding `samples` (DESIGN.md: valid duration in (0, 2], (2, 4], (4, 8], (8, 15] s).
    public static func bucket(forSamples samples: Int) throws -> Int {
        guard let b = seconds.first(where: { samples <= 16000 * $0 }) else {
            throw BenchError.invalid("\(samples) samples exceed 15 s")
        }
        return b
    }
}

public enum LengthVariant: String, CaseIterable, Sendable {
    case fixed15, multifunction, enumerated
}

/// An encoder arm's Core ML model(s): fixed 15 s window (one function), a multifunction model with functions
/// b2, b4, b8, b15 (each loaded through MLModelConfiguration.functionName), or one model with enumerated shapes.
/// A .mlpackage is compiled once into <artifacts>/compiled/ (compile time recorded).
public final class EncoderModel {
    public let variant: LengthVariant
    public let models: [Int: MLModel]   // bucket -> model (fixed15 and enumerated: one model under every bucket)
    public let loadMs: [String: Double]
    public let compileMs: Double?
    public let url: URL

    public init(url: URL, variant: LengthVariant, computeUnits: MLComputeUnits, compiledDir: URL?) async throws {
        self.variant = variant
        var compiled = url
        var compileMs: Double? = nil
        if url.pathExtension == "mlpackage" {
            guard let compiledDir else { throw BenchError.invalid("a .mlpackage needs a compiled-model directory") }
            let dest = compiledDir.appendingPathComponent(url.deletingPathExtension().lastPathComponent + ".mlmodelc")
            let fm = FileManager.default
            let srcDate = (try? fm.attributesOfItem(atPath: url.appendingPathComponent("Manifest.json").path)[.modificationDate]) as? Date
            let dstDate = (try? fm.attributesOfItem(atPath: dest.path)[.modificationDate]) as? Date
            if dstDate == nil || (srcDate != nil && srcDate! > dstDate!) {
                let t0 = Clock.now()
                let tmp = try await MLModel.compileModel(at: url)
                compileMs = Clock.ms(t0, Clock.now())
                try fm.createDirectory(at: compiledDir, withIntermediateDirectories: true)
                if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
                try fm.moveItem(at: tmp, to: dest)
            }
            compiled = dest
        }
        self.url = compiled
        self.compileMs = compileMs
        var models: [Int: MLModel] = [:], times: [String: Double] = [:]
        switch variant {
        case .fixed15, .enumerated:
            let t0 = Clock.now()
            let m = try MLModel(contentsOf: compiled, configuration: C0Models.configuration(computeUnits))
            times["main"] = Clock.ms(t0, Clock.now())
            for b in (variant == .fixed15 ? [15] : Buckets.seconds) { models[b] = m }
        case .multifunction:
            for b in Buckets.seconds {
                let config = C0Models.configuration(computeUnits)
                config.functionName = "b\(b)"
                let t0 = Clock.now()
                models[b] = try MLModel(contentsOf: compiled, configuration: config)
                times["b\(b)"] = Clock.ms(t0, Clock.now())
            }
        }
        self.models = models
        loadMs = times
    }

    public func bucket(forSamples n: Int) throws -> Int {
        if variant == .fixed15 {
            guard n <= 240_000 else { throw BenchError.invalid("\(n) samples exceed the 15 s window") }
            return 15
        }
        return try Buckets.bucket(forSamples: n)
    }

    /// One encoder call: mel [1, 128, F_b] + mel_length -> (encoder output, encoder_length).
    public func predict(mel: MLMultiArray, melLength: Int, bucket: Int,
                        options: MLPredictionOptions) async throws -> (MLMultiArray, Int) {
        guard let model = models[bucket] else { throw BenchError.invalid("no encoder function for bucket \(bucket)") }
        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = NSNumber(value: melLength)
        let input = try MLDictionaryFeatureProvider(dictionary: ["mel": MLFeatureValue(multiArray: mel),
                                                                 "mel_length": MLFeatureValue(multiArray: length)])
        let out = try await model.prediction(from: input, options: options)
        guard let enc = out.featureValue(for: "encoder")?.multiArrayValue,
              let len = out.featureValue(for: "encoder_length")?.multiArrayValue else {
            throw BenchError.invalid("encoder output missing")
        }
        return (enc, len[0].intValue)
    }
}
