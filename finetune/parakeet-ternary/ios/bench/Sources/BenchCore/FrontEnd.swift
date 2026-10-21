import Accelerate
import CoreML
import Foundation

/// A float32 blob + JSON manifest written by ios/native.py (tensors with shapes and byte offsets; file SHA-256).
public struct NativeBlob {
    public let manifest: [String: Any]
    public let tensors: [String: (shape: [Int], values: [Float])]

    public init(directory: URL, stem: String) throws {
        let json = try Data(contentsOf: directory.appendingPathComponent("\(stem).json"))
        guard let m = try JSONSerialization.jsonObject(with: json) as? [String: Any],
              let file = m["file"] as? String, let sha = m["sha256"] as? String,
              let list = m["tensors"] as? [[String: Any]] else {
            throw BenchError.invalid("\(stem).json is not a native blob manifest")
        }
        let data = try Data(contentsOf: directory.appendingPathComponent(file))
        guard sha256Hex(data) == sha else { throw BenchError.invalid("\(file): SHA-256 differs from \(stem).json") }
        var out: [String: (shape: [Int], values: [Float])] = [:]
        try data.withUnsafeBytes { raw in
            for t in list {
                guard let name = t["name"] as? String, let shape = t["shape"] as? [Int], let offset = t["offset"] as? Int,
                      let bytes = t["bytes"] as? Int, bytes == shape.reduce(1, *) * 4, offset + bytes <= data.count else {
                    throw BenchError.invalid("\(stem).json: bad tensor entry")
                }
                let floats = raw.baseAddress!.advanced(by: offset).assumingMemoryBound(to: Float.self)
                out[name] = (shape, Array(UnsafeBufferPointer(start: floats, count: bytes / 4)))
            }
        }
        manifest = m
        tensors = out
    }

    public func tensor(_ name: String, _ shape: [Int]) throws -> [Float] {
        guard let t = tensors[name] else { throw BenchError.invalid("missing tensor \(name)") }
        guard t.shape == shape else { throw BenchError.invalid("\(name): shape \(t.shape) != \(shape)") }
        return t.values
    }
}

/// Front end A: NeMo's FilterbankFeatures contract at inference on the CPU with Accelerate (reference.py
/// Featurizer): dither off; pre-emphasis 0.97 within the valid samples; centred STFT (n_fft 512, hop 160,
/// constant zero padding of 256 each side, the stored symmetric Hann window of 400 centred in 512);
/// |X|^2 as sqrt then square; the stored Slaney mel filterbank [128, 257]; log(x + 2^-24); per-feature
/// normalization over the valid frames M = N // 160 (mean, unbiased std + 1e-5); frames >= M set to 0.
/// Output [128, N // 160 + 1] feature-major. The window and filterbank are the model's stored values,
/// exported by `native.py frontend`.
public final class VDSPFrontEnd {
    public static let nFFT = 512, hop = 160, winLength = 400, features = 128, bins = 257
    public let window512: [Float]        // stored window centred in n_fft (zeros outside)
    public let fb: [Float]               // [128, 257]
    let preemph: Float = 0.97
    let logGuard: Float = Float(pow(2.0, -24.0))
    let normConstant: Float = 1e-5
    let dft: vDSP_DFT_Setup
    let dftScale: Float

    public convenience init(constantsDir: URL) throws {
        let blob = try NativeBlob(directory: constantsDir, stem: "frontend")
        try self.init(window: try blob.tensor("window", [Self.winLength]), fb: try blob.tensor("fb", [Self.features, Self.bins]))
    }

    public init(window: [Float], fb: [Float]) throws {
        guard window.count == Self.winLength, fb.count == Self.features * Self.bins else {
            throw BenchError.invalid("front-end constants have the wrong size")
        }
        self.fb = fb
        var w = [Float](repeating: 0, count: Self.nFFT)
        let offset = (Self.nFFT - Self.winLength) / 2
        for i in 0..<Self.winLength { w[offset + i] = window[i] }
        window512 = w
        guard let setup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(Self.nFFT), .FORWARD) else {
            throw BenchError.invalid("vDSP_DFT_zrop_CreateSetup failed")
        }
        dft = setup
        // Calibrate vDSP's packed real-DFT scaling on a unit impulse (true DFT = 1 in every bin).
        var re = [Float](repeating: 0, count: Self.nFFT / 2), im = re, outR = re, outI = re
        re[0] = 1
        vDSP_DFT_Execute(setup, &re, &im, &outR, &outI)
        dftScale = 1 / outR[1]
    }

    deinit { vDSP_DFT_DestroySetup(dft) }

    /// Real DFT of 512 samples -> re[257], im[257] (true DFT values; vDSP packs DC and Nyquist into element 0).
    public func dft512(_ x: UnsafePointer<Float>, re: UnsafeMutablePointer<Float>, im: UnsafeMutablePointer<Float>) {
        let half = Self.nFFT / 2
        var even = [Float](repeating: 0, count: half), odd = even, outR = even, outI = even
        for k in 0..<half { even[k] = x[2 * k]; odd[k] = x[2 * k + 1] }
        vDSP_DFT_Execute(dft, &even, &odd, &outR, &outI)
        re[0] = outR[0] * dftScale; im[0] = 0
        re[half] = outI[0] * dftScale; im[half] = 0
        for k in 1..<half { re[k] = outR[k] * dftScale; im[k] = outI[k] * dftScale }
    }

    /// Features [128, T] (T = N // 160 + 1) and the valid frame count M = N // 160.
    public func compute(_ pcm: [Float]) -> (features: [Float], frames: Int, valid: Int) {
        let n = pcm.count, nFFT = Self.nFFT, pad = nFFT / 2
        let valid = n / Self.hop, frames = n / Self.hop + 1
        // pre-emphasis within the valid samples, then constant zero padding of n_fft / 2 on both sides
        var padded = [Float](repeating: 0, count: n + 2 * pad)
        if n > 0 {
            padded[pad] = pcm[0]
            for i in 1..<n { padded[pad + i] = pcm[i] - preemph * pcm[i - 1] }
        }
        var power = [Float](repeating: 0, count: Self.bins * frames)   // [257, T]
        var frame = [Float](repeating: 0, count: nFFT)
        var re = [Float](repeating: 0, count: Self.bins), im = re
        padded.withUnsafeBufferPointer { p in
            for f in 0..<frames {
                vDSP_vmul(p.baseAddress! + f * Self.hop, 1, window512, 1, &frame, 1, vDSP_Length(nFFT))
                frame.withUnsafeBufferPointer { fr in dft512(fr.baseAddress!, re: &re, im: &im) }
                for k in 0..<Self.bins {
                    let mag = (re[k] * re[k] + im[k] * im[k]).squareRoot()
                    power[k * frames + f] = mag * mag
                }
            }
        }
        var mel = [Float](repeating: 0, count: Self.features * frames)   // [128, T] = fb [128, 257] x power [257, T]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(Self.features), Int32(frames), Int32(Self.bins),
                    1, fb, Int32(Self.bins), power, Int32(frames), 0, &mel, Int32(frames))
        for i in 0..<mel.count { mel[i] = log(mel[i] + logGuard) }
        for c in 0..<Self.features {
            let row = c * frames
            var sum = 0.0
            for f in 0..<valid { sum += Double(mel[row + f]) }
            let mean = Float(sum / Double(valid))
            var sq = 0.0
            for f in 0..<valid { let d = Double(mel[row + f] - mean); sq += d * d }
            var std = Float((sq / Double(valid - 1)).squareRoot())
            if std.isNaN { std = 0 }
            std += normConstant
            for f in 0..<frames { mel[row + f] = f < valid ? (mel[row + f] - mean) / std : 0 }
        }
        return (mel, frames, valid)
    }
}

/// Where an arm's mel input comes from.
public enum FrontEndKind: String, Sendable {
    /// Front end A on the CPU (VDSPFrontEnd).
    case vdsp
    /// C0's Core ML Preprocessor with C0's contract: audio zero-padded to 240,000 samples plus audio_length; its
    /// output [1, 128, 1501] is cut to the bucket's F_b frames and its own mel_length (N // 160 + 1, one more
    /// than NeMo) is passed on unchanged. For comparison only.
    case c0pre
}

/// Builds the encoder's mel input [1, 128, F_b] + mel_length for a clip (DESIGN.md masking contract: features
/// of the valid audio, normalized over the valid frames, padded frames 0, zero-padded to F_b).
public final class MelInput {
    public let kind: FrontEndKind
    let vdsp: VDSPFrontEnd?
    let preprocessor: MLModel?
    let options: MLPredictionOptions = { let o = MLPredictionOptions(); o.outputBackings = [:]; return o }()

    public init(kind: FrontEndKind, constantsDir: URL?, c0Dir: URL?, preprocessorUnits: MLComputeUnits) throws {
        self.kind = kind
        switch kind {
        case .vdsp:
            guard let constantsDir else { throw BenchError.invalid("--frontend vdsp needs --frontend-constants") }
            vdsp = try VDSPFrontEnd(constantsDir: constantsDir); preprocessor = nil
        case .c0pre:
            guard let c0Dir else { throw BenchError.invalid("--frontend c0pre needs --models (C0 directory)") }
            vdsp = nil
            preprocessor = try MLModel(contentsOf: c0Dir.appendingPathComponent("Preprocessor.mlmodelc"),
                                       configuration: C0Models.configuration(preprocessorUnits))
        }
    }

    public func mel(_ pcm: [Float], bucket: Int) async throws -> (MLMultiArray, Int) {
        let fB = Buckets.melFrames(bucket)
        let out = try MLMultiArray(shape: [1, 128, NSNumber(value: fB)], dataType: .float32)
        let dst = out.dataPointer.bindMemory(to: Float.self, capacity: 128 * fB)
        dst.initialize(repeating: 0, count: 128 * fB)
        switch kind {
        case .vdsp:
            let (features, frames, valid) = vdsp!.compute(pcm)
            guard frames <= fB else { throw BenchError.invalid("\(frames) frames exceed bucket \(bucket) s") }
            features.withUnsafeBufferPointer { src in
                for c in 0..<128 { (dst + c * fB).update(from: src.baseAddress! + c * frames, count: frames) }
            }
            return (out, valid)
        case .c0pre:
            guard pcm.count <= 240_000 else { throw BenchError.invalid("clip longer than 15 s") }
            let audio = try AlignedArray.make(shape: [1, 240_000], dataType: .float32, zero: true)
            _ = pcm.withUnsafeBufferPointer { memcpy(audio.dataPointer, $0.baseAddress!, pcm.count * 4) }
            let length = try MLMultiArray(shape: [1], dataType: .int32)
            length[0] = NSNumber(value: pcm.count)
            let res = try await preprocessor!.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "audio_signal": MLFeatureValue(multiArray: audio), "audio_length": MLFeatureValue(multiArray: length)]),
                options: options)
            guard let mel = res.featureValue(for: "mel")?.multiArrayValue,
                  let len = res.featureValue(for: "mel_length")?.multiArrayValue else {
                throw BenchError.invalid("preprocessor output missing")
            }
            let shape = mel.intShape, strides = mel.intStrides
            guard shape.count == 3, shape[1] == 128, shape[2] >= fB else { throw BenchError.invalid("mel shape \(shape)") }
            let src = mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count)
            for c in 0..<128 { for f in 0..<fB { dst[c * fB + f] = src[c * strides[1] + f * strides[2]] } }
            return (out, len[0].intValue)
        }
    }
}
