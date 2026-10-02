import Accelerate
import CoreML
import Foundation

// C0: FluidInference's parakeet-tdt-0.6b-v2-coreml driven exactly as FluidAudio 0.7.8 drives it for v2
// (tag v0.7.8 = commit 8136bd0642e7c5ce1f6f5b2931890266aeecb08c). Sources replicated, all under
// Sources/FluidAudio/ at that tag:
// - ASR/AsrModels.swift: preprocessor loaded .cpuOnly; encoder, decoder, joint with the configuration's
//   compute units (default .cpuAndNeuralEngine); allowLowPrecisionAccumulationOnGPU = true (DownloadUtils.swift).
// - ASR/AsrTranscription.swift transcribeWithState / executeMLInferenceWithTimings: audio of <= 240,000
//   samples zero-padded to 240,000; preprocessor input audio_signal [1, 240000] + audio_length = original
//   length; the preprocessor's output provider is the encoder's input; actualAudioFrames = ceil(N / 1280);
//   decode with contextFrameAdjustment 0, isLastChunk false, globalFrameOffset 0.
// - ASR/TDT/TdtDecoderV2.swift (blank 1024) -> ASR/TDT/TdtDecoderV3.swift decodeWithTimings: the loop below.
// - ASR/TDT/TdtDecoderState.swift, ASR/TDT/EncoderFrameView.swift, Shared/ANEMemoryUtils.swift.
// Not replicated (outside the PCM -> tokens boundary): AsrManager.transcribe's resetDecoderState after each
// utterance (two extra decoder calls on token 0 whose state the next utterance zeroes again), text
// post-processing beyond detokenization, chunking of > 15 s audio.

public struct TdtConfig: Sendable {
    public var blankId = 1024            // TdtDecoderV2.adaptConfigForV2
    public var maxSymbolsPerStep = 10    // TdtConfig default
    public var durationBins = [0, 1, 2, 3, 4]
    public var maxTokensPerChunk = 150
    public init() {}
}

public enum ComputeUnitsName {
    public static func parse(_ s: String) throws -> MLComputeUnits {
        switch s {
        case "cpuOnly": return .cpuOnly
        case "cpuAndGPU": return .cpuAndGPU
        case "cpuAndNeuralEngine": return .cpuAndNeuralEngine
        case "all": return .all
        default: throw BenchError.invalid("unknown compute units \(s)")
        }
    }

    public static func name(_ u: MLComputeUnits) -> String {
        switch u {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        case .all: return "all"
        @unknown default: return "unknown(\(u.rawValue))"
        }
    }
}

public final class C0Models {
    public static let names = ["Preprocessor", "Encoder", "Decoder", "JointDecision"]
    public let preprocessor: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let joint: MLModel
    public let vocabulary: [Int: String]
    public let loadMs: [String: Double]
    public let computeUnits: MLComputeUnits
    public let preprocessorUnits: MLComputeUnits

    public static func configuration(_ units: MLComputeUnits) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = units
        config.allowLowPrecisionAccumulationOnGPU = true
        return config
    }

    public init(directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                preprocessorUnits: MLComputeUnits = .cpuOnly) throws {
        self.computeUnits = computeUnits
        self.preprocessorUnits = preprocessorUnits
        var models: [String: MLModel] = [:]
        var times: [String: Double] = [:]
        for name in Self.names {  // FluidAudio order: preprocessor, encoder, then decoder and joint
            let units = name == "Preprocessor" ? preprocessorUnits : computeUnits
            let state = Signposts.poi.beginInterval("load", id: Signposts.poi.makeSignpostID(), "\(name)")
            let t0 = Clock.now()
            models[name] = try MLModel(contentsOf: directory.appendingPathComponent("\(name).mlmodelc"),
                                       configuration: Self.configuration(units))
            times[name] = Clock.ms(t0, Clock.now())
            Signposts.poi.endInterval("load", state)
        }
        preprocessor = models["Preprocessor"]!
        encoder = models["Encoder"]!
        decoder = models["Decoder"]!
        joint = models["JointDecision"]!
        loadMs = times
        vocabulary = try loadVocabulary(directory.appendingPathComponent("parakeet_vocab.json"))
    }
}

/// Decoder LSTM state (TdtDecoderState): h and c [2, 1, 640] float32. A struct of references, so copies alias
/// the same arrays, as in FluidAudio (decoder calls write h_out/c_out in place through outputBackings).
struct DecoderState {
    var hidden: MLMultiArray
    var cell: MLMultiArray
    var lastToken: Int?
    var predictorOutput: MLMultiArray?

    static func make() throws -> DecoderState {
        let h = try AlignedArray.make(shape: [2, 1, 640], dataType: .float32)
        let c = try AlignedArray.make(shape: [2, 1, 640], dataType: .float32)
        h.fill(0); c.fill(0)
        return DecoderState(hidden: h, cell: c, lastToken: nil, predictorOutput: nil)
    }

    mutating func update(from output: MLFeatureProvider) {
        hidden = output.featureValue(for: "h_out")?.multiArrayValue ?? hidden
        cell = output.featureValue(for: "c_out")?.multiArrayValue ?? cell
    }
}

public enum DecodeMode: Sendable {
    case free
    case replay(TraceClip)

    public static let names = ["free", "replay"]

    public var name: String {
        switch self {
        case .free: return "free"
        case .replay: return "replay"
        }
    }
}

/// One end-to-end call: PCM in memory -> token IDs. Timed calls carry only this minimal bookkeeping, identical in
/// both modes (token and timestamp per emission, call counters); everything else is in `Diagnostics`, which only
/// the untimed diagnostic pass collects.
public struct CallResult: Encodable, Sendable {
    public var bucket = 15
    public var tokens: [Int] = []
    public var timestamps: [Int] = []
    public var encoderLength = 0
    public var actualAudioFrames = 0
    public var effectiveFrames = 0
    public var melLength = 0
    public var jointCalls = 0
    public var decoderCalls = 0
    public var preprocessorCalls = 0
    public var encoderCalls = 0
    /// Physical calls per decode-loop component (e.g. decoder_model, joint_model, fused_model, native_predict,
    /// native_joint); the logical work (steps, prediction-net runs) is the trace's or the loop's.
    public var physicalCalls: [String: Int] = [:]
    /// Stages: preprocess (front end), encoder (incl. output materialization), preprojection (joint encoder-side
    /// projection of all frames; 0 where the joint projects per step), decode, total, plus per-component model time.
    public var timesMs: [String: Double] = [:]
}

/// Untimed diagnostic outputs of one call (DESIGN.md "Replay trace": replay returns logits and LSTM states).
///
/// Limitation of C0 (the product baseline, not a gated arm): the published JointDecision.mlmodelc computes the
/// 1,030 joint logits internally and outputs only token_id (argmax over the 1,025 token+blank logits), token_prob
/// (softmax probability of that token over the same 1,025) and duration (argmax bin of the 5 duration logits);
/// see its model.mil. Raw token and duration logits are therefore not obtainable from the four models FluidAudio
/// 0.7.8 uses, and C0 cannot be put through the logit/margin parts of gate 4. What is exposed per joint step:
/// frame, token id, token probability, duration bin, and the index of the decoder call whose output fed it; per
/// decoder call: input token, the `decoder` output (prediction-net output, 640) and copies of h_out and c_out
/// [2, 1, 640] taken right after the call (the state that produced that output, as reference.StepOutputs); and
/// the encoder output's frames [encoderLength, 1024].
public final class Diagnostics {
    public var frames: [Int] = []
    public var tokenIds: [Int] = []
    public var tokenProbs: [Float] = []
    public var durationBins: [Int] = []
    public var stepDecoderCall: [Int] = []
    public var decoderTokens: [Int] = []
    public var decoderOut: [Float] = []   // [calls, 640]
    public var h: [Float] = []            // [calls, 2, 640]
    public var c: [Float] = []            // [calls, 2, 640]
    public var encoder: [Float] = []      // [encoderFrames, 1024]
    public var encoderFrames = 0
    public init() {}

    func step(_ t: Int, _ d: C0Pipeline.Decision, decoderCall: Int) {
        frames.append(t); tokenIds.append(d.token); tokenProbs.append(d.probability); durationBins.append(d.durationBin)
        stepDecoderCall.append(decoderCall)
    }

    func decoderCall(token: Int, output: MLMultiArray, state: DecoderState) {
        decoderTokens.append(token)
        decoderOut += copyFloats(output)
        h += copyFloats(state.hidden)
        c += copyFloats(state.cell)
    }

    /// Little-endian float32: encoder [encoderFrames, 1024], then per decoder call decoder[640], h[1280], c[1280].
    public func binary() -> Data {
        var data = Data()
        encoder.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        for k in 0..<decoderTokens.count {
            decoderOut[(k * 640)..<((k + 1) * 640)].withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
            h[(k * 1280)..<((k + 1) * 1280)].withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
            c[(k * 1280)..<((k + 1) * 1280)].withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        }
        return data
    }
}

/// Logical-order float copy of any float32 MLMultiArray (honours strides).
func copyFloats(_ a: MLMultiArray) -> [Float] {
    let shape = a.intShape, strides = a.intStrides
    let base = a.dataPointer.bindMemory(to: Float.self, capacity: 1)
    var out = [Float](repeating: 0, count: shape.reduce(1, *))
    var index = [Int](repeating: 0, count: shape.count)
    for i in 0..<out.count {
        var offset = 0
        for d in 0..<shape.count { offset += index[d] * strides[d] }
        out[i] = base[offset]
        var d = shape.count - 1
        while d >= 0 {
            index[d] += 1
            if index[d] < shape[d] { break }
            index[d] = 0
            d -= 1
        }
    }
    return out
}

public final class C0Pipeline {
    let models: C0Models
    let tdt = TdtConfig()
    /// AsrManager.predictionOptions (AsrModels.optimizedPredictionOptions(): outputBackings [:]).
    let managerOptions: MLPredictionOptions = {
        let o = MLPredictionOptions(); o.outputBackings = [:]; return o
    }()
    public static let windowSamples = 240_000
    public var blank: Int { tdt.blankId }
    public var durations: [Int] { tdt.durationBins }
    public var maxSymbols: Int { tdt.maxSymbolsPerStep }

    public init(models: C0Models) { self.models = models }

    /// AsrManager.preparePreprocessorInput on the zero-padded window (ANE-aligned [1, 240000] array).
    func preprocessorInput(_ pcm: [Float]) throws -> MLFeatureProvider {
        guard pcm.count <= Self.windowSamples else { throw BenchError.invalid("clip longer than 15 s") }
        let padded = pcm + [Float](repeating: 0, count: Self.windowSamples - pcm.count)  // padAudioIfNeeded
        let audio = try AlignedArray.make(shape: [1, Self.windowSamples], dataType: .float32)
        _ = padded.withUnsafeBufferPointer { buf in
            memcpy(audio.dataPointer, buf.baseAddress!, Self.windowSamples * MemoryLayout<Float>.stride)
        }
        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = NSNumber(value: pcm.count)
        return try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: audio), "audio_length": MLFeatureValue(multiArray: length),
        ])
    }

    /// One call. `diagnostics` nil = the timed path; non-nil only in the untimed diagnostic pass.
    public func run(_ pcm: [Float], mode: DecodeMode, diagnostics: Diagnostics? = nil) async throws -> CallResult {
        var r = CallResult()
        let callID = Signposts.poi.makeSignpostID()
        let callState = Signposts.poi.beginInterval("call", id: callID, "\(mode.name)")
        let t0 = Clock.now()

        // Front end (Core ML preprocessor).
        let sPre = Signposts.stages.beginInterval("preprocess", id: callID)
        let preIn = try preprocessorInput(pcm)
        let preOut = try await models.preprocessor.prediction(from: preIn, options: managerOptions)
        r.preprocessorCalls = 1
        guard let melLength = preOut.featureValue(for: "mel_length")?.multiArrayValue else {
            throw BenchError.invalid("preprocessor output lacks mel_length")
        }
        r.melLength = melLength[0].intValue
        let t1 = Clock.now()
        Signposts.stages.endInterval("preprocess", sPre)

        // Encoder; its output is materialized (dataPointer bound and read).
        let sEnc = Signposts.stages.beginInterval("encoder", id: callID)
        let encIn = try encoderInput(preOut, fallback: preIn)
        let encOut = try await models.encoder.prediction(from: encIn, options: managerOptions)
        r.encoderCalls = 1
        guard let encoder = encOut.featureValue(for: "encoder")?.multiArrayValue,
              let encoderLength = encOut.featureValue(for: "encoder_length")?.multiArrayValue else {
            throw BenchError.invalid("encoder output missing")
        }
        r.encoderLength = encoderLength[0].intValue
        r.actualAudioFrames = (pcm.count + 1279) / 1280  // ASRConstants.calculateEncoderFrames
        r.effectiveFrames = min(r.encoderLength, r.actualAudioFrames)
        let frames = try EncoderFrames(encoder, validLength: r.encoderLength)
        var touch: Float = 0
        vDSP_sve(frames.base, 1, &touch, vDSP_Length(min(encoder.count, 1024)))
        let t2 = Clock.now()
        Signposts.stages.endInterval("encoder", sEnc)

        // Decode loop.
        let sDec = Signposts.stages.beginInterval("decode", id: callID)
        var decoderAcc = Accumulator(), jointAcc = Accumulator()
        switch mode {
        case .free:
            try decodeFree(frames: frames, result: &r, decoderAcc: &decoderAcc, jointAcc: &jointAcc, diag: diagnostics)
        case .replay(let trace):
            try decodeReplay(frames: frames, trace: trace, result: &r, decoderAcc: &decoderAcc, jointAcc: &jointAcc,
                             diag: diagnostics)
        }
        let t3 = Clock.now()
        Signposts.stages.endInterval("decode", sDec)
        Signposts.poi.endInterval("call", callState)

        r.decoderCalls = decoderAcc.count
        r.jointCalls = jointAcc.count
        r.physicalCalls = ["decoder_model": decoderAcc.count, "joint_model": jointAcc.count]
        r.timesMs = ["preprocess": Clock.ms(t0, t1), "encoder": Clock.ms(t1, t2), "preprojection": 0, "decode": Clock.ms(t2, t3),
                     "decoder_model": decoderAcc.totalMs, "joint_model": jointAcc.totalMs, "total": Clock.ms(t0, t3)]
        _ = touch
        if let diagnostics {  // after the clock stopped
            diagnostics.encoderFrames = frames.count
            diagnostics.encoder.reserveCapacity(frames.count * 1024)
            var row = [Float](repeating: 0, count: 1024)
            for t in 0..<frames.count {
                try row.withUnsafeMutableBufferPointer { try frames.copyFrame(t, into: $0.baseAddress!, destStride: 1) }
                diagnostics.encoder += row
            }
        }
        return r
    }

    /// AsrTranscription.prepareEncoderInput: the preprocessor output provider if it has every encoder input.
    func encoderInput(_ pre: MLFeatureProvider, fallback: MLFeatureProvider) throws -> MLFeatureProvider {
        let names = models.encoder.modelDescription.inputDescriptionsByName.keys
        if names.allSatisfy({ pre.featureValue(for: $0) != nil }) { return pre }
        var features: [String: MLFeatureValue] = [:]
        for name in names {
            guard let v = pre.featureValue(for: name) ?? fallback.featureValue(for: name) else {
                throw BenchError.invalid("missing encoder input \(name)")
            }
            features[name] = v
        }
        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    // MARK: - per-step model calls (TdtDecoderV3.runDecoder / runJointPrepared)

    final class JointInput: NSObject, MLFeatureProvider {
        let encoderStep: MLMultiArray
        let decoderStep: MLMultiArray
        init(encoderStep: MLMultiArray, decoderStep: MLMultiArray) {
            self.encoderStep = encoderStep; self.decoderStep = decoderStep
        }
        var featureNames: Set<String> { ["encoder_step", "decoder_step"] }
        func featureValue(for featureName: String) -> MLFeatureValue? {
            switch featureName {
            case "encoder_step": return MLFeatureValue(multiArray: encoderStep)
            case "decoder_step": return MLFeatureValue(multiArray: decoderStep)
            default: return nil
            }
        }
    }

    /// Per-utterance buffers of TdtDecoderV3.decodeWithTimings.
    final class StepBuffers {
        let options = MLPredictionOptions()  // TdtDecoderV3.predictionOptions (shared by decoder and joint calls)
        let target: MLMultiArray
        let targetLength: MLMultiArray
        let encoderStep: MLMultiArray
        let decoderStep: MLMultiArray
        let jointInput: JointInput
        let encDest: UnsafeMutablePointer<Float>
        let encDestStride: Int
        let tokenId: MLMultiArray
        let tokenProb: MLMultiArray
        let duration: MLMultiArray

        init() throws {
            options.outputBackings = [:]
            target = try MLMultiArray(shape: [1, 1], dataType: .int32)
            targetLength = try MLMultiArray(shape: [1], dataType: .int32)
            targetLength[0] = 1
            encoderStep = try AlignedArray.make(shape: [1, 1024, 1], dataType: .float32)
            decoderStep = try AlignedArray.make(shape: [1, 640, 1], dataType: .float32)
            jointInput = JointInput(encoderStep: encoderStep, decoderStep: decoderStep)
            encDestStride = encoderStep.intStrides[1]
            encDest = encoderStep.dataPointer.bindMemory(to: Float.self, capacity: 1024)
            tokenId = try MLMultiArray(shape: [1, 1, 1], dataType: .int32)
            tokenProb = try MLMultiArray(shape: [1, 1, 1], dataType: .float32)
            duration = try MLMultiArray(shape: [1, 1, 1], dataType: .int32)
        }
    }

    struct Decision { let token: Int; let probability: Float; let durationBin: Int }

    func runDecoder(token: Int, state: DecoderState, buffers b: StepBuffers, acc: inout Accumulator,
                    diag: Diagnostics?) throws -> (output: MLFeatureProvider, newState: DecoderState) {
        b.target[0] = NSNumber(value: token)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: b.target), "target_length": MLFeatureValue(multiArray: b.targetLength),
            "h_in": MLFeatureValue(multiArray: state.hidden), "c_in": MLFeatureValue(multiArray: state.cell),
        ])
        b.options.outputBackings = ["h_out": state.hidden, "c_out": state.cell]
        let output = try acc.measure { try models.decoder.prediction(from: input, options: b.options) }
        var newState = state
        newState.update(from: output)
        if let diag, let out = output.featureValue(for: "decoder")?.multiArrayValue {
            diag.decoderCall(token: token, output: out, state: newState)  // copies: h/c are overwritten in place later
        }
        return (output, newState)
    }

    func runJoint(frames: EncoderFrames, t: Int, buffers b: StepBuffers, acc: inout Accumulator) throws -> Decision {
        try frames.copyFrame(t, into: b.encDest, destStride: b.encDestStride)
        _ = b.encoderStep[0]; _ = b.encoderStep[b.encoderStep.count - 1]   // ANEOptimizer.prefetchToNeuralEngine
        _ = b.decoderStep[0]; _ = b.decoderStep[b.decoderStep.count - 1]
        b.options.outputBackings = ["token_id": b.tokenId, "token_prob": b.tokenProb, "duration": b.duration]
        let out = try acc.measure { try models.joint.prediction(from: b.jointInput, options: b.options) }
        guard let tid = out.featureValue(for: "token_id")?.multiArrayValue,
              let tp = out.featureValue(for: "token_prob")?.multiArrayValue,
              let dur = out.featureValue(for: "duration")?.multiArrayValue,
              tid.count == 1, tp.count == 1, dur.count == 1 else {
            throw BenchError.invalid("joint decision output missing")
        }
        return Decision(token: Int(tid.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]),
                        probability: tp.dataPointer.bindMemory(to: Float.self, capacity: 1)[0],
                        durationBin: Int(dur.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]))
    }

    /// TdtDecoderV3.populatePreparedDecoderProjection: copy the decoder output [1, 640, 1] or [1, 1, 640] into the
    /// prepared [1, 640, 1] (stride-16) step array.
    func populateDecoderStep(_ projection: MLMultiArray, into out: MLMultiArray) throws {
        let shape = projection.intShape, strides = projection.intStrides
        guard shape.count == 3, shape[0] == 1, projection.dataType == .float32 else {
            throw BenchError.invalid("decoder projection \(shape)")
        }
        let hiddenAxis = shape[2] == 640 ? 2 : (shape[1] == 640 ? 1 : -1)
        guard hiddenAxis > 0 else { throw BenchError.invalid("decoder projection hidden size \(shape)") }
        let src = projection.dataPointer.bindMemory(to: Float.self, capacity: projection.count)
        let dst = out.dataPointer.bindMemory(to: Float.self, capacity: 640)
        let srcStride = strides[hiddenAxis], dstStride = out.intStrides[1]
        if srcStride == 1 && dstStride == 1 {
            dst.update(from: src, count: 640)
        } else {
            cblas_scopy(640, src, Int32(srcStride), dst, Int32(dstStride))
        }
    }

    func durationValue(_ bin: Int) throws -> Int {
        guard bin >= 0 && bin < tdt.durationBins.count else { throw BenchError.invalid("duration bin \(bin)") }
        return tdt.durationBins[bin]
    }

    // MARK: - free decoding: TdtDecoderV3.decodeWithTimings for one fresh utterance (first and only chunk)

    func decodeFree(frames: EncoderFrames, result r: inout CallResult, decoderAcc: inout Accumulator,
                    jointAcc: inout Accumulator, diag: Diagnostics?) throws {
        let encoderSequenceLength = r.encoderLength
        guard encoderSequenceLength > 1 else { return }  // "Early exit for very short audio"
        let blank = tdt.blankId
        var decoderState = try DecoderState.make()        // fresh state, lastToken nil, predictorOutput nil
        var hypDecState: DecoderState? = decoderState     // TdtHypothesis(decState: decoderState)
        var hypLastToken: Int? = decoderState.lastToken

        var timeIndices = 0                                // first chunk, contextFrameAdjustment 0
        let effective = r.effectiveFrames                  // min(encoderSequenceLength, actualAudioFrames)
        var safeTimeIndices = min(timeIndices, effective - 1)
        var timeIndicesCurrentLabels = timeIndices
        var activeMask = timeIndices < effective
        let lastTimestep = effective - 1
        if timeIndices >= effective { return }

        let b = try StepBuffers()
        // decoderState.lastToken == nil && predictorOutput == nil: zero the state (already zero).
        // Prime with SOS = blank.
        let primed = try runDecoder(token: blank, state: decoderState, buffers: b, acc: &decoderAcc, diag: diag)
        guard let primedProj = primed.output.featureValue(for: "decoder")?.multiArrayValue else {
            throw BenchError.invalid("decoder output missing")
        }
        decoderState.predictorOutput = primedProj
        hypDecState = primed.newState

        var lastEmissionTimestamp = -1
        var emissionsAtThisTimestamp = 0
        var tokensProcessedThisChunk = 0

        while activeMask {
            var label = hypLastToken ?? blank
            let stateToUse = hypDecState ?? decoderState
            let decoderResult: (output: MLFeatureProvider, newState: DecoderState)
            if let cached = decoderState.predictorOutput {
                decoderResult = (try MLDictionaryFeatureProvider(dictionary: ["decoder": MLFeatureValue(multiArray: cached)]),
                                 stateToUse)
            } else {
                decoderResult = try runDecoder(token: label, state: stateToUse, buffers: b, acc: &decoderAcc, diag: diag)
            }
            guard let projection = decoderResult.output.featureValue(for: "decoder")?.multiArrayValue else {
                throw BenchError.invalid("decoder output missing")
            }
            try populateDecoderStep(projection, into: b.decoderStep)

            let decision = try runJoint(frames: frames, t: safeTimeIndices, buffers: b, acc: &jointAcc)
            diag?.step(safeTimeIndices, decision, decoderCall: decoderAcc.count - 1)
            label = decision.token
            var duration = try durationValue(decision.durationBin)
            var blankMask = label == blank
            if blankMask && duration == 0 { duration = 1 }

            timeIndicesCurrentLabels = timeIndices
            timeIndices += duration
            safeTimeIndices = min(timeIndices, lastTimestep)
            activeMask = timeIndices < effective
            var advanceMask = activeMask && blankMask

            while advanceMask {  // inner blank loop: decoder output reused
                timeIndicesCurrentLabels = timeIndices
                let inner = try runJoint(frames: frames, t: safeTimeIndices, buffers: b, acc: &jointAcc)
                diag?.step(safeTimeIndices, inner, decoderCall: decoderAcc.count - 1)
                label = inner.token
                duration = try durationValue(inner.durationBin)
                blankMask = label == blank
                if blankMask && duration == 0 { duration = 1 }
                timeIndices += duration
                safeTimeIndices = min(timeIndices, lastTimestep)
                activeMask = timeIndices < effective
                advanceMask = activeMask && blankMask
            }

            if activeMask && label != blank {
                tokensProcessedThisChunk += 1
                if tokensProcessedThisChunk > tdt.maxTokensPerChunk { break }
                r.tokens.append(label)
                r.timestamps.append(timeIndicesCurrentLabels)
                hypLastToken = label
                let step = try runDecoder(token: label, state: decoderResult.newState, buffers: b, acc: &decoderAcc,
                                          diag: diag)
                hypDecState = step.newState
                decoderState.predictorOutput = step.output.featureValue(for: "decoder")?.multiArrayValue
                if timeIndicesCurrentLabels == lastEmissionTimestamp {
                    emissionsAtThisTimestamp += 1
                } else {
                    lastEmissionTimestamp = timeIndicesCurrentLabels
                    emissionsAtThisTimestamp = 1
                }
                if emissionsAtThisTimestamp >= tdt.maxSymbolsPerStep {  // force-blank mechanism
                    timeIndices = min(timeIndices + 1, lastTimestep)
                    safeTimeIndices = min(timeIndices, lastTimestep)
                    emissionsAtThisTimestamp = 0
                    lastEmissionTimestamp = -1
                }
            }
            activeMask = timeIndices < effective
        }
    }

    // MARK: - replay: the trace's decisions drive the same per-step decoder and joint calls

    /// The trace (validated by TraceFile.validated before any call) is B0's: num_frames = ceil((N // 160) / 8).
    /// C0's documented frame-count difference is allowed explicitly: its effective frames
    /// min(encoder_length, ceil(N / 1280)) may exceed num_frames by exactly 0 or 1 (its preprocessor reports
    /// mel_length = N // 160 + 1, and FluidAudio's ceil(N / 1280)); replay follows the trace's frames, so the
    /// logical work is the trace's. Anything else is an error.
    func decodeReplay(frames: EncoderFrames, trace: TraceClip, result r: inout CallResult,
                      decoderAcc: inout Accumulator, jointAcc: inout Accumulator, diag: Diagnostics?) throws {
        let extra = r.effectiveFrames - trace.numFrames
        guard extra == 0 || extra == 1 else {
            throw BenchError.invalid("\(trace.id): C0 effective frames \(r.effectiveFrames) vs trace \(trace.numFrames)")
        }
        let b = try StepBuffers()
        var current = try runDecoder(token: tdt.blankId, state: try DecoderState.make(), buffers: b, acc: &decoderAcc,
                                     diag: diag)
        guard let primedProj = current.output.featureValue(for: "decoder")?.multiArrayValue else {
            throw BenchError.invalid("decoder output missing")
        }
        try populateDecoderStep(primedProj, into: b.decoderStep)
        for i in 0..<trace.steps {
            let d = try runJoint(frames: frames, t: trace.frame[i], buffers: b, acc: &jointAcc)
            diag?.step(trace.frame[i], d, decoderCall: decoderAcc.count - 1)
            if trace.predUpdated[i] != 0 {
                r.tokens.append(trace.token[i])
                r.timestamps.append(trace.frame[i])
                current = try runDecoder(token: trace.token[i], state: current.newState, buffers: b, acc: &decoderAcc,
                                         diag: diag)
                guard let proj = current.output.featureValue(for: "decoder")?.multiArrayValue else {
                    throw BenchError.invalid("decoder output missing")
                }
                try populateDecoderStep(proj, into: b.decoderStep)
            }
        }
    }
}
