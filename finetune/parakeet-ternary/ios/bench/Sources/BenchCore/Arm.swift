import Accelerate
import CoreML
import Foundation

/// A benchmark arm = {front end, encoder package + length variant, decode-loop variant, compute units}
/// (DESIGN.md "Arms"); C0 itself is C0Pipeline (FluidAudio's own loop). Every other arm decodes with LabelLoop.
///
/// Timing boundary (DESIGN.md "Measurements"): from the 16 kHz PCM buffer in memory to the final token IDs. Stages:
/// preprocess (front end, incl. bucket padding), encoder (Core ML call, output materialized), preprojection
/// (the engine's per-utterance set-up: F2's encoder-side joint projection of all frames; ~0 for F0/F1), decode
/// (the loop), total. Bookkeeping in timed calls is the same for every arm: tokens, timestamps, counters.
public final class ArmPipeline {
    public let name: String
    public let mel: MelInput?
    public let encoder: EncoderModel?
    /// Gate mode: the encoder output is read from <dir>/<clip id>.f32 (time-major [T, 1024], e.g. the FP32
    /// reference's, native.py reference) before the clock starts; front end and encoder are skipped.
    public let externalEncoderDir: URL?
    public let engine: DecodeEngine
    public var loop = LabelLoop()
    let options: MLPredictionOptions = { let o = MLPredictionOptions(); o.outputBackings = [:]; return o }()

    public init(name: String, mel: MelInput?, encoder: EncoderModel?, externalEncoderDir: URL?, engine: DecodeEngine) throws {
        guard (externalEncoderDir != nil) != (encoder != nil && mel != nil) else {
            throw BenchError.invalid("an arm needs a front end and an encoder, or an external encoder input")
        }
        self.name = name; self.mel = mel; self.encoder = encoder; self.externalEncoderDir = externalEncoderDir
        self.engine = engine
    }

    /// Frames C0-contract front ends may add relative to the trace (c0pre's mel_length = N // 160 + 1).
    var allowedExtraFrames: Int { mel?.kind == .c0pre ? 1 : 0 }

    func external(_ clip: Clip) throws -> EncoderFrames {
        let url = externalEncoderDir!.appendingPathComponent("\(clip.id).f32")
        let data = try Data(contentsOf: url)
        guard data.count % 4096 == 0, data.count > 0 else { throw BenchError.invalid("\(url.lastPathComponent): size") }
        let buffer = FloatBuffer(count: data.count / 4)
        _ = data.withUnsafeBytes { memcpy(buffer.pointer, $0.baseAddress!, data.count) }
        return try EncoderFrames(timeMajor: buffer, frames: data.count / 4096)
    }

    public func run(clip: Clip, pcm: [Float], mode: DecodeMode, diag: DiagSink? = nil) async throws -> CallResult {
        var r = CallResult()
        let pre = try externalEncoderDir.map { _ in try external(clip) }
        let callID = Signposts.poi.makeSignpostID()
        let callState = Signposts.poi.beginInterval("call", id: callID, "\(self.name) \(mode.name)")
        let t0 = Clock.now()
        var frames: EncoderFrames
        var t1 = t0
        if let pre {
            frames = pre
            r.encoderLength = pre.count
            r.bucket = 0
        } else {
            let sPre = Signposts.stages.beginInterval("preprocess", id: callID)
            let bucket = try encoder!.bucket(forSamples: pcm.count)
            let (melArray, melLength) = try await mel!.mel(pcm, bucket: bucket)
            r.bucket = bucket; r.melLength = melLength; r.preprocessorCalls = mel!.kind == .c0pre ? 1 : 0
            t1 = Clock.now()
            Signposts.stages.endInterval("preprocess", sPre)
            let sEnc = Signposts.stages.beginInterval("encoder", id: callID)
            let (out, length) = try await encoder!.predict(mel: melArray, melLength: melLength, bucket: bucket, options: options)
            r.encoderCalls = 1
            frames = try EncoderFrames(out, validLength: length)
            r.encoderLength = frames.count
            var touch: Float = 0
            vDSP_sve(frames.base, 1, &touch, vDSP_Length(min(out.count, 1024)))
            _ = touch
            Signposts.stages.endInterval("encoder", sEnc)
        }
        let t2 = Clock.now()
        let length = frames.count
        r.effectiveFrames = length
        let sDec = Signposts.stages.beginInterval("decode", id: callID)
        try engine.begin(frames: frames, length: length, diag: diag)
        let t2b = Clock.now()
        let out: LabelLoop.Output
        switch mode {
        case .free:
            out = try loop.free(engine, length: length, diag: diag)
        case .replay(let trace):
            let extra = length - trace.numFrames
            guard extra >= 0 && extra <= allowedExtraFrames else {
                throw BenchError.invalid("\(clip.id): encoder frames \(length) vs trace \(trace.numFrames)")
            }
            out = try loop.replay(engine, trace: trace, diag: diag)
        }
        let t3 = Clock.now()
        Signposts.stages.endInterval("decode", sDec)
        Signposts.poi.endInterval("call", callState)
        r.tokens = out.tokens
        r.timestamps = out.timestamps
        var times: [String: Double] = ["preprocess": pre == nil ? Clock.ms(t0, t1) : 0,
                                       "encoder": pre == nil ? Clock.ms(t1, t2) : 0,
                                       "preprojection": Clock.ms(t2, t2b), "decode": Clock.ms(t2b, t3),
                                       "total": Clock.ms(t0, t3)]
        for (component, v) in engine.accounting() where component != "preprojection" {
            r.physicalCalls[component] = v.calls
            times[component] = v.ms
        }
        r.decoderCalls = r.physicalCalls["decoder_model"] ?? r.physicalCalls["native_predict"] ?? 0
        r.jointCalls = r.physicalCalls["joint_model"] ?? r.physicalCalls["native_joint"] ?? r.physicalCalls["fused_model"] ?? 0
        r.timesMs = times
        if let diag, pre == nil {
            diag.append("encoder", rowShape: [1024], try frames.timeMajor())
        }
        return r
    }
}
