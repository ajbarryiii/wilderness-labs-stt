import Accelerate
import CoreML
import Foundation

/// Named float sections and integer/float step fields of an untimed diagnostic pass (one binary file + JSON).
public final class DiagSink {
    public private(set) var sections: [(name: String, rowShape: [Int], data: [Float])] = []
    public var ints: [String: [Int]] = [:]
    public var floats: [String: [Double]] = [:]
    public init() {}

    public func append(_ name: String, rowShape: [Int], _ values: UnsafeBufferPointer<Float>) {
        if let i = sections.firstIndex(where: { $0.name == name }) {
            sections[i].data.append(contentsOf: values)
        } else {
            sections.append((name, rowShape, Array(values)))
        }
    }

    public func append(_ name: String, rowShape: [Int], _ values: [Float]) {
        values.withUnsafeBufferPointer { append(name, rowShape: rowShape, $0) }
    }

    public func int(_ name: String, _ v: Int) { ints[name, default: []].append(v) }
    public func float(_ name: String, _ v: Double) { floats[name, default: []].append(v) }

    /// Concatenated little-endian float32 sections and their [{name, shape, offset, bytes}] index.
    public func binary() -> (Data, [[String: Any]]) {
        var data = Data(), index: [[String: Any]] = []
        for s in sections {
            let rowSize = s.rowShape.reduce(1, *)
            let rows = rowSize > 0 ? s.data.count / rowSize : 0
            index.append(["name": s.name, "shape": [rows] + s.rowShape, "offset": data.count, "bytes": s.data.count * 4])
            s.data.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        }
        return (data, index)
    }
}

/// One decode-loop implementation (DESIGN.md arm F). The loop logic is shared (LabelLoop); an engine supplies the
/// prediction network (predict: run it on a token, or note the token for a fused call) and the joint (one
/// decision at an encoder frame). Physical calls and their time are accumulated per component.
public protocol DecodeEngine: AnyObject {
    var name: String { get }
    /// Per utterance: reset the state; F2 also projects the encoder side of the joint for all valid frames here.
    func begin(frames: EncoderFrames, length: Int, diag: DiagSink?) throws
    func predict(_ token: Int, diag: DiagSink?) throws
    func joint(_ t: Int, diag: DiagSink?) throws -> (token: Int, durationBin: Int)
    /// component -> (physical calls, ms) since begin; "preprojection" is reported as its own stage.
    func accounting() -> [String: (calls: Int, ms: Double)]
}

/// NeMo greedy_batch label-looping TDT semantics (reference.run_steps), shared by every arm except C0 (which keeps
/// FluidAudio's own loop): zero state and the blank/SOS input, reset per utterance; token = argmax over
/// tokens+blank, duration = durations[argmax]; blank with duration 0 advances 1; after an emission the prediction
/// network runs on it, and max_symbols emissions at a frame force an advance; stop when the frame reaches L.
public struct LabelLoop {
    public var blank = 1024
    public var durations = [0, 1, 2, 3, 4]
    public var maxSymbols = 10
    public init() {}

    public struct Output {
        public var tokens: [Int] = []
        public var timestamps: [Int] = []
        /// trace of the free decode (recordTrace; the unit tests validate it with TraceFile.validate)
        public var frame: [Int] = [], token: [Int] = [], duration: [Int] = [], predInput: [Int] = []
        public var symbolsAtFrame: [Int] = [], forcedAdvance: [Int] = [], advance: [Int] = []
    }

    public func free(_ e: DecodeEngine, length: Int, diag: DiagSink?, recordTrace: Bool = false) throws -> Output {
        var out = Output()
        try e.predict(blank, diag: diag)
        var t = 0, lastNB = -1, lasts = 0, predInput = blank
        while t < length {
            let (tok, bin) = try e.joint(t, diag: diag)
            guard bin >= 0 && bin < durations.count else { throw BenchError.invalid("duration bin \(bin)") }
            let dur = durations[bin]
            let emitted = tok != blank
            var advance = (!emitted && dur == 0) ? 1 : dur
            var forced = false
            if emitted {
                lasts = lastNB == t ? lasts + 1 : 1
                lastNB = t
                out.tokens.append(tok); out.timestamps.append(t)
                try e.predict(tok, diag: diag)
                if t + advance < length && lasts >= maxSymbols && lastNB == t + advance { advance += 1; forced = true }
            }
            if recordTrace {
                out.frame.append(t); out.token.append(tok); out.duration.append(dur); out.predInput.append(predInput)
                out.symbolsAtFrame.append(lasts); out.forcedAdvance.append(forced ? 1 : 0); out.advance.append(advance)
            }
            if emitted { predInput = tok }
            t += advance
        }
        return out
    }

    /// The trace's decisions drive the prediction network and the frames (validated trace; same logical work).
    public func replay(_ e: DecodeEngine, trace: TraceClip, diag: DiagSink?) throws -> Output {
        var out = Output()
        try e.predict(blank, diag: diag)
        for i in 0..<trace.steps {
            let (tok, bin) = try e.joint(trace.frame[i], diag: diag)
            diag?.int("argmax_token", tok)
            diag?.int("argmax_duration", bin >= 0 && bin < durations.count ? durations[bin] : -1)
            if trace.predUpdated[i] != 0 {
                out.tokens.append(trace.token[i]); out.timestamps.append(trace.frame[i])
                try e.predict(trace.token[i], diag: diag)
            }
        }
        return out
    }
}

// MARK: - F2: native CPU loop (Accelerate), FP32

/// Decoder + joint weights of a benchmark model (native.py weights: decoder_joint.json + .f32bin).
public final class NativeWeights {
    public static let hidden = 640, layers = 2, dModel = 1024, outputs = 1030, vocabPlusBlank = 1025
    public let manifest: [String: Any]
    let embed, wih0, whh0, bias0, w1cat, bias1, encW, encB, predW, predB, outW, outB: [Float]

    public convenience init(directory: URL) throws {
        let blob = try NativeBlob(directory: directory, stem: "decoder_joint")
        try self.init(tensors: blob.tensors.mapValues { $0.values }, manifest: blob.manifest)
    }

    /// From NeMo-named tensors (sizes checked; shapes as in native.py DECODER_KEYS).
    public init(tensors: [String: [Float]], manifest: [String: Any] = [:]) throws {
        let blob = NamedTensors(tensors)
        let H = Self.hidden, G = 4 * H
        let p = "decoder.prediction."
        embed = try blob.tensor(p + "embed.weight", [Self.vocabPlusBlank, H])
        wih0 = try blob.tensor(p + "dec_rnn.lstm.weight_ih_l0", [G, H])
        whh0 = try blob.tensor(p + "dec_rnn.lstm.weight_hh_l0", [G, H])
        let bih0 = try blob.tensor(p + "dec_rnn.lstm.bias_ih_l0", [G]), bhh0 = try blob.tensor(p + "dec_rnn.lstm.bias_hh_l0", [G])
        let wih1 = try blob.tensor(p + "dec_rnn.lstm.weight_ih_l1", [G, H]), whh1 = try blob.tensor(p + "dec_rnn.lstm.weight_hh_l1", [G, H])
        let bih1 = try blob.tensor(p + "dec_rnn.lstm.bias_ih_l1", [G]), bhh1 = try blob.tensor(p + "dec_rnn.lstm.bias_hh_l1", [G])
        bias0 = zip(bih0, bhh0).map { $0 + $1 }
        bias1 = zip(bih1, bhh1).map { $0 + $1 }
        var cat = [Float](repeating: 0, count: G * 2 * H)  // [W_ih1 | W_hh1], [2560, 1280]
        for r in 0..<G {
            for c in 0..<H { cat[r * 2 * H + c] = wih1[r * H + c]; cat[r * 2 * H + H + c] = whh1[r * H + c] }
        }
        w1cat = cat
        encW = try blob.tensor("joint.enc.weight", [H, Self.dModel]); encB = try blob.tensor("joint.enc.bias", [H])
        predW = try blob.tensor("joint.pred.weight", [H, H]); predB = try blob.tensor("joint.pred.bias", [H])
        outW = try blob.tensor("joint.joint_net.2.weight", [Self.outputs, H]); outB = try blob.tensor("joint.joint_net.2.bias", [Self.outputs])
        guard embed[(Self.vocabPlusBlank - 1) * H..<Self.vocabPlusBlank * H].allSatisfy({ $0 == 0 }) else {
            throw BenchError.invalid("the blank embedding row must be zero (start symbol)")
        }
        self.manifest = manifest
    }
}

struct NamedTensors {
    let values: [String: [Float]]
    init(_ values: [String: [Float]]) { self.values = values }
    func tensor(_ name: String, _ shape: [Int]) throws -> [Float] {
        guard let v = values[name] else { throw BenchError.invalid("missing tensor \(name)") }
        guard v.count == shape.reduce(1, *) else { throw BenchError.invalid("\(name): \(v.count) values for \(shape)") }
        return v
    }
}

/// Shared native math: the 2-layer LSTM step (PyTorch gate order i, f, g, o), the pred projection and the joint.
final class NativeMath {
    let w: NativeWeights
    let H = NativeWeights.hidden
    /// layer-0 input contribution per token: embed[token] W_ih0^T + b_ih0 + b_hh0, [1025, 2560] (one sgemm at load)
    let table0: [Float]
    var gates = [Float](repeating: 0, count: 2560)
    var xcat = [Float](repeating: 0, count: 1280)

    init(_ w: NativeWeights) {
        self.w = w
        let G = 4 * NativeWeights.hidden, V = NativeWeights.vocabPlusBlank
        var t = [Float](repeating: 0, count: V * G)
        for v in 0..<V { for g in 0..<G { t[v * G + g] = w.bias0[g] } }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(V), Int32(G), Int32(NativeWeights.hidden), 1,
                    w.embed, Int32(NativeWeights.hidden), w.wih0, Int32(NativeWeights.hidden), 1, &t, Int32(G))
        table0 = t
    }

    @inline(__always) static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    /// One LSTM cell update in place: h, c [640] with gates [2560] (already holding the input + bias part).
    func cell(_ gates: UnsafeMutablePointer<Float>, h: UnsafeMutablePointer<Float>, c: UnsafeMutablePointer<Float>) {
        for k in 0..<H {
            let i = Self.sigmoid(gates[k]), f = Self.sigmoid(gates[H + k])
            let g = tanh(gates[2 * H + k]), o = Self.sigmoid(gates[3 * H + k])
            let cn = f * c[k] + i * g
            c[k] = cn
            h[k] = o * tanh(cn)
        }
    }

    /// Prediction network on `token` from state (h, c) [2, 640] each, updated in place; writes the projected output
    /// g = W_pred h_top + b_pred [640].
    func predict(_ token: Int, h: UnsafeMutablePointer<Float>, c: UnsafeMutablePointer<Float>, g: UnsafeMutablePointer<Float>) {
        let G = 4 * H
        gates.withUnsafeMutableBufferPointer { gp in
            // layer 0: gates = table0[token] + W_hh0 h0
            table0.withUnsafeBufferPointer { gp.baseAddress!.update(from: $0.baseAddress! + token * G, count: G) }
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(G), Int32(H), 1, w.whh0, Int32(H), h, 1, 1, gp.baseAddress!, 1)
            cell(gp.baseAddress!, h: h, c: c)
            // layer 1: gates = [W_ih1 | W_hh1] [h0'; h1] + b_ih1 + b_hh1
            xcat.withUnsafeMutableBufferPointer { x in
                x.baseAddress!.update(from: h, count: H)
                (x.baseAddress! + H).update(from: h + H, count: H)
                w.bias1.withUnsafeBufferPointer { gp.baseAddress!.update(from: $0.baseAddress!, count: G) }
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(G), Int32(2 * H), 1, w.w1cat, Int32(2 * H), x.baseAddress!, 1, 1,
                            gp.baseAddress!, 1)
            }
            cell(gp.baseAddress!, h: h + H, c: c + H)
        }
        w.predB.withUnsafeBufferPointer { g.update(from: $0.baseAddress!, count: H) }
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(H), Int32(H), 1, w.predW, Int32(H), h + H, 1, 1, g, 1)
    }

    /// Encoder-side joint projection of frames 0..<length: f [length, 640] = E W_enc^T + b_enc (one sgemm).
    func project(_ frames: EncoderFrames, length: Int, into f: UnsafeMutablePointer<Float>) {
        for t in 0..<length { w.encB.withUnsafeBufferPointer { (f + t * H).update(from: $0.baseAddress!, count: H) } }
        if frames.hiddenStride == 1 {  // time-major [T, 1024]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(length), Int32(H), 1024, 1, frames.base,
                        Int32(frames.timeStride), w.encW, 1024, 1, f, Int32(H))
        } else {                       // hidden-major [1024, T_alloc] with unit time stride
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, Int32(length), Int32(H), 1024, 1, frames.base,
                        Int32(frames.hiddenStride), w.encW, 1024, 1, f, Int32(H))
        }
    }

    /// Joint logits [1030] = W_out relu(f_t + g) + b_out; returns (argmax token over 1025, argmax duration bin).
    func joint(f: UnsafePointer<Float>, g: UnsafePointer<Float>, z: UnsafeMutablePointer<Float>,
               logits: UnsafeMutablePointer<Float>) -> (Int, Int) {
        vDSP_vadd(f, 1, g, 1, z, 1, vDSP_Length(H))
        var zero: Float = 0
        vDSP_vthr(z, 1, &zero, z, 1, vDSP_Length(H))
        w.outB.withUnsafeBufferPointer { logits.update(from: $0.baseAddress!, count: NativeWeights.outputs) }
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(NativeWeights.outputs), Int32(H), 1, w.outW, Int32(H), z, 1, 1, logits, 1)
        return (argmax(logits, 1025), argmax(logits + 1025, 5))
    }

    @inline(__always) func argmax(_ p: UnsafePointer<Float>, _ n: Int) -> Int {
        var best = 0
        for i in 1..<n where p[i] > p[best] { best = i }  // first maximum, as torch.max
        return best
    }
}

/// F2: native CPU decode loop. Joint encoder side projected for all frames in one sgemm (stage "preprojection");
/// prediction network run only after a non-blank emission (its projected output cached); joint per step as one
/// sgemv plus the two-head argmax. FP32 throughout. Diagnostics: per step logits [1030] and the LSTM state (h, c)
/// [2, 640] that produced the step's prediction output (as reference.StepOutputs).
public final class NativeEngine: DecodeEngine {
    public let name = "f2"
    let math: NativeMath
    let H = NativeWeights.hidden
    var h = [Float](repeating: 0, count: 1280), c = [Float](repeating: 0, count: 1280), g = [Float](repeating: 0, count: 640)
    var f = FloatBuffer(count: 0)
    var z = [Float](repeating: 0, count: 640), logits = [Float](repeating: 0, count: 1030)
    var predictAcc = Accumulator(), jointAcc = Accumulator(), projectMs = 0.0
    public private(set) var lastPreprojectionMs = 0.0

    public init(weights: NativeWeights) { math = NativeMath(weights) }

    public func begin(frames: EncoderFrames, length: Int, diag: DiagSink?) throws {
        for i in 0..<1280 { h[i] = 0; c[i] = 0 }
        predictAcc = Accumulator(); jointAcc = Accumulator()
        let t0 = Clock.now()
        if f.count < length * H { f = FloatBuffer(count: length * H) }
        math.project(frames, length: length, into: f.pointer)
        lastPreprojectionMs = Clock.ms(t0, Clock.now())
    }

    public func predict(_ token: Int, diag: DiagSink?) throws {
        guard token >= 0 && token < NativeWeights.vocabPlusBlank else { throw BenchError.invalid("token \(token)") }
        predictAcc.measure {
            h.withUnsafeMutableBufferPointer { hp in c.withUnsafeMutableBufferPointer { cp in g.withUnsafeMutableBufferPointer { gp in
                math.predict(token, h: hp.baseAddress!, c: cp.baseAddress!, g: gp.baseAddress!)
            } } }
        }
    }

    public func joint(_ t: Int, diag: DiagSink?) throws -> (token: Int, durationBin: Int) {
        let r = jointAcc.measure { () -> (Int, Int) in
            g.withUnsafeBufferPointer { gp in z.withUnsafeMutableBufferPointer { zp in logits.withUnsafeMutableBufferPointer { lp in
                math.joint(f: f.pointer + t * H, g: gp.baseAddress!, z: zp.baseAddress!, logits: lp.baseAddress!)
            } } }
        }
        if let diag {
            diag.append("logits", rowShape: [1030], logits)
            diag.append("h", rowShape: [2, 640], h)
            diag.append("c", rowShape: [2, 640], c)
            diag.int("frame", t); diag.int("token_id", r.0); diag.int("duration_bin", r.1)
        }
        return (r.0, r.1)
    }

    public func accounting() -> [String: (calls: Int, ms: Double)] {
        ["native_predict": (predictAcc.count, predictAcc.totalMs), "native_joint": (jointAcc.count, jointAcc.totalMs),
         "preprojection": (1, lastPreprojectionMs)]
    }
}

/// Native fused decoder+joint (test double of F1's call structure on F2's math): every joint step re-runs the
/// prediction network on the pending token from the state before it, as a fused DecoderJoint call does.
public final class NativeFusedEngine: DecodeEngine {
    public let name = "f1-native"
    let math: NativeMath
    let H = NativeWeights.hidden
    var pending = 1024
    var hBefore = [Float](repeating: 0, count: 1280), cBefore = [Float](repeating: 0, count: 1280)
    var hOut = [Float](repeating: 0, count: 1280), cOut = [Float](repeating: 0, count: 1280)
    var g = [Float](repeating: 0, count: 640), z = [Float](repeating: 0, count: 640), logits = [Float](repeating: 0, count: 1030)
    var f = FloatBuffer(count: 0)
    var calls = Accumulator(), started = false

    public init(weights: NativeWeights) { math = NativeMath(weights) }

    public func begin(frames: EncoderFrames, length: Int, diag: DiagSink?) throws {
        for i in 0..<1280 { hBefore[i] = 0; cBefore[i] = 0 }
        pending = 1024; started = false; calls = Accumulator()
        if f.count < length * H { f = FloatBuffer(count: length * H) }
        math.project(frames, length: length, into: f.pointer)
    }

    public func predict(_ token: Int, diag: DiagSink?) throws {
        if started { hBefore = hOut; cBefore = cOut }  // the last fused call consumed the previous pending token
        pending = token
        started = true
    }

    public func joint(_ t: Int, diag: DiagSink?) throws -> (token: Int, durationBin: Int) {
        calls.measure { () -> (Int, Int) in
            hOut = hBefore; cOut = cBefore
            hOut.withUnsafeMutableBufferPointer { hp in cOut.withUnsafeMutableBufferPointer { cp in g.withUnsafeMutableBufferPointer { gp in
                math.predict(pending, h: hp.baseAddress!, c: cp.baseAddress!, g: gp.baseAddress!)
            } } }
            return g.withUnsafeBufferPointer { gp in z.withUnsafeMutableBufferPointer { zp in logits.withUnsafeMutableBufferPointer { lp in
                math.joint(f: f.pointer + t * H, g: gp.baseAddress!, z: zp.baseAddress!, logits: lp.baseAddress!)
            } } }
        }
    }

    public func accounting() -> [String: (calls: Int, ms: Double)] { ["native_fused": (calls.count, calls.totalMs)] }
}

// MARK: - F0 and F1: Core ML per-step models

/// F0: separate per-step Core ML Decoder and JointDecision calls with C0's contract (Decoder: targets [1, 1],
/// target_length [1], h_in/c_in [2, 1, 640] -> decoder [1, 640, 1], h_out, c_out; JointDecision: encoder_step
/// [1, 1024, 1], decoder_step [1, 640, 1] -> token_id, token_prob, duration [1, 1, 1]) under LabelLoop. Buffers
/// and in-place state handling as FluidAudio's TdtDecoderV3 (h_out/c_out written into h_in/c_in).
public final class CoreMLStepEngine: DecodeEngine {
    public let name = "f0"
    let decoder: MLModel, jointModel: MLModel
    let b: C0Pipeline.StepBuffers
    var state: DecoderState
    var frames: EncoderFrames?
    var decAcc = Accumulator(), jointAcc = Accumulator()

    public init(decoder: MLModel, joint: MLModel) throws {
        self.decoder = decoder; jointModel = joint
        b = try C0Pipeline.StepBuffers()
        state = try DecoderState.make()
    }

    public func begin(frames: EncoderFrames, length: Int, diag: DiagSink?) throws {
        state.hidden.fill(0); state.cell.fill(0)
        self.frames = frames
        decAcc = Accumulator(); jointAcc = Accumulator()
    }

    public func predict(_ token: Int, diag: DiagSink?) throws {
        b.target[0] = NSNumber(value: token)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: b.target), "target_length": MLFeatureValue(multiArray: b.targetLength),
            "h_in": MLFeatureValue(multiArray: state.hidden), "c_in": MLFeatureValue(multiArray: state.cell)])
        b.options.outputBackings = ["h_out": state.hidden, "c_out": state.cell]
        let out = try decAcc.measure { try decoder.prediction(from: input, options: b.options) }
        state.update(from: out)
        guard let proj = out.featureValue(for: "decoder")?.multiArrayValue else { throw BenchError.invalid("decoder output") }
        try copyStep(proj, into: b.decoderStep, hidden: 640)
        if let diag {
            diag.append("decoder_out", rowShape: [640], copyFloats(proj))
            diag.append("h", rowShape: [2, 640], copyFloats(state.hidden))
            diag.append("c", rowShape: [2, 640], copyFloats(state.cell))
            diag.int("decoder_call_token", token)
        }
    }

    public func joint(_ t: Int, diag: DiagSink?) throws -> (token: Int, durationBin: Int) {
        try frames!.copyFrame(t, into: b.encDest, destStride: b.encDestStride)
        b.options.outputBackings = ["token_id": b.tokenId, "token_prob": b.tokenProb, "duration": b.duration]
        let out = try jointAcc.measure { try jointModel.prediction(from: b.jointInput, options: b.options) }
        let d = try readDecision(out)
        if let diag {
            diag.int("frame", t); diag.int("token_id", d.token); diag.int("duration_bin", d.bin)
            diag.float("token_prob", Double(d.prob)); diag.int("decoder_call", decAcc.count - 1)
        }
        return (d.token, d.bin)
    }

    public func accounting() -> [String: (calls: Int, ms: Double)] {
        ["decoder_model": (decAcc.count, decAcc.totalMs), "joint_model": (jointAcc.count, jointAcc.totalMs)]
    }
}

/// F1: one fused Core ML DecoderJoint call per step (WP3 contract: targets [1, 1], h_in/c_in [2, 1, 640],
/// encoder_step [1, 1024, 1] -> token_id, token_prob, duration, h_out, c_out). The call runs the prediction network
/// on `targets` from h_in/c_in and the joint on its output; the loop passes the pending token with the state before
/// it, and adopts h_out/c_out as the new "before" state when the next token is emitted (ping-pong buffers).
public final class CoreMLFusedEngine: DecodeEngine {
    public let name = "f1"
    let model: MLModel
    let options = MLPredictionOptions()
    let target: MLMultiArray, encoderStep: MLMultiArray
    let encDest: UnsafeMutablePointer<Float>, encDestStride: Int
    var before: (MLMultiArray, MLMultiArray), after: (MLMultiArray, MLMultiArray)
    let tokenId: MLMultiArray, tokenProb: MLMultiArray, duration: MLMultiArray
    var frames: EncoderFrames?
    var pending = 1024, started = false
    var acc = Accumulator()

    public init(model: MLModel) throws {
        self.model = model
        target = try MLMultiArray(shape: [1, 1], dataType: .int32)
        encoderStep = try AlignedArray.make(shape: [1, 1024, 1], dataType: .float32)
        encDest = encoderStep.dataPointer.bindMemory(to: Float.self, capacity: 1024)
        encDestStride = encoderStep.intStrides[1]
        before = (try AlignedArray.make(shape: [2, 1, 640], dataType: .float32), try AlignedArray.make(shape: [2, 1, 640], dataType: .float32))
        after = (try AlignedArray.make(shape: [2, 1, 640], dataType: .float32), try AlignedArray.make(shape: [2, 1, 640], dataType: .float32))
        tokenId = try MLMultiArray(shape: [1, 1, 1], dataType: .int32)
        tokenProb = try MLMultiArray(shape: [1, 1, 1], dataType: .float32)
        duration = try MLMultiArray(shape: [1, 1, 1], dataType: .int32)
    }

    public func begin(frames: EncoderFrames, length: Int, diag: DiagSink?) throws {
        before.0.fill(0); before.1.fill(0)
        pending = 1024; started = false; self.frames = frames; acc = Accumulator()
    }

    public func predict(_ token: Int, diag: DiagSink?) throws {
        if started { swap(&before, &after) }  // the last call's output state = state after the previous pending token
        pending = token
        started = true
    }

    public func joint(_ t: Int, diag: DiagSink?) throws -> (token: Int, durationBin: Int) {
        try frames!.copyFrame(t, into: encDest, destStride: encDestStride)
        target[0] = NSNumber(value: pending)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: target), "h_in": MLFeatureValue(multiArray: before.0),
            "c_in": MLFeatureValue(multiArray: before.1), "encoder_step": MLFeatureValue(multiArray: encoderStep)])
        options.outputBackings = ["token_id": tokenId, "token_prob": tokenProb, "duration": duration,
                                  "h_out": after.0, "c_out": after.1]
        let out = try acc.measure { try model.prediction(from: input, options: options) }
        if let h = out.featureValue(for: "h_out")?.multiArrayValue, h !== after.0 { try copyStep(h, into: after.0, hidden: 1280) }
        if let c = out.featureValue(for: "c_out")?.multiArrayValue, c !== after.1 { try copyStep(c, into: after.1, hidden: 1280) }
        let d = try readDecision(out)
        if let diag {
            diag.int("frame", t); diag.int("token_id", d.token); diag.int("duration_bin", d.bin)
            diag.float("token_prob", Double(d.prob))
            diag.append("h_out", rowShape: [2, 640], copyFloats(after.0))
            diag.append("c_out", rowShape: [2, 640], copyFloats(after.1))
        }
        return (d.token, d.bin)
    }

    public func accounting() -> [String: (calls: Int, ms: Double)] { ["fused_model": (acc.count, acc.totalMs)] }
}

func readDecision(_ out: MLFeatureProvider) throws -> (token: Int, prob: Float, bin: Int) {
    guard let tid = out.featureValue(for: "token_id")?.multiArrayValue,
          let tp = out.featureValue(for: "token_prob")?.multiArrayValue,
          let dur = out.featureValue(for: "duration")?.multiArrayValue,
          tid.count == 1, tp.count == 1, dur.count == 1 else { throw BenchError.invalid("joint decision output missing") }
    return (Int(tid.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]),
            tp.dataPointer.bindMemory(to: Float.self, capacity: 1)[0],
            Int(dur.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]))
}

/// Logical-order copy of a float32 array of `count` elements into `out` (which may be strided).
func copyStep(_ src: MLMultiArray, into out: MLMultiArray, hidden: Int) throws {
    guard src.dataType == .float32, src.count == hidden, out.count == hidden else {
        throw BenchError.invalid("step array size \(src.count) != \(hidden)")
    }
    let values = copyFloats(src)
    let shape = out.intShape, strides = out.intStrides
    let dst = out.dataPointer.bindMemory(to: Float.self, capacity: 1)
    var index = [Int](repeating: 0, count: shape.count)
    for i in 0..<hidden {
        var offset = 0
        for d in 0..<shape.count { offset += index[d] * strides[d] }
        dst[offset] = values[i]
        var d = shape.count - 1
        while d >= 0 { index[d] += 1; if index[d] < shape[d] { break }; index[d] = 0; d -= 1 }
    }
}
