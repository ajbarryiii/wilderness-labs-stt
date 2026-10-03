import XCTest
@testable import BenchCore

/// Unit tests of the parts that need no model file: the vDSP DFT, the native LSTM/joint math against a naive
/// double-precision implementation, and the decode loops (F2 vs the fused call structure, free decode trace
/// validity, replay of a free decode's own trace). Seeded, deterministic.
final class BenchCoreTests: XCTestCase {
    struct LCG {  // deterministic uniform/normal generator
        var s: UInt64
        mutating func uniform() -> Double { s = s &* 6364136223846793005 &+ 1442695040888963407; return Double(s >> 11) / Double(1 << 53) }
        mutating func normal() -> Float { Float((-2 * log(max(uniform(), 1e-12))).squareRoot() * cos(2 * Double.pi * uniform())) }
    }

    func randomWeights(seed: UInt64, scale: Float = 0.08) throws -> [String: [Float]] {
        var g = LCG(s: seed)
        let H = 640, G = 2560
        func t(_ n: Int, _ s: Float = scale) -> [Float] { (0..<n).map { _ in g.normal() * s } }
        let p = "decoder.prediction."
        var embed = t(1025 * H, 1)
        for k in 0..<H { embed[1024 * H + k] = 0 }  // blank row (padding_idx) is zero
        return [p + "embed.weight": embed,
                p + "dec_rnn.lstm.weight_ih_l0": t(G * H, 0.05), p + "dec_rnn.lstm.weight_hh_l0": t(G * H, 0.05),
                p + "dec_rnn.lstm.bias_ih_l0": t(G), p + "dec_rnn.lstm.bias_hh_l0": t(G),
                p + "dec_rnn.lstm.weight_ih_l1": t(G * H, 0.05), p + "dec_rnn.lstm.weight_hh_l1": t(G * H, 0.05),
                p + "dec_rnn.lstm.bias_ih_l1": t(G), p + "dec_rnn.lstm.bias_hh_l1": t(G),
                "joint.enc.weight": t(H * 1024, 0.03), "joint.enc.bias": t(H),
                "joint.pred.weight": t(H * H, 0.05), "joint.pred.bias": t(H),
                "joint.joint_net.2.weight": t(1030 * H, 0.1), "joint.joint_net.2.bias": t(1030, 0.5)]
    }

    func testDFTMatchesNaive() throws {
        var g = LCG(s: 1)
        let window = (0..<400).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / 399)) }
        let fe = try VDSPFrontEnd(window: window, fb: [Float](repeating: 0, count: 128 * 257))
        let x = (0..<512).map { _ in g.normal() }
        var re = [Float](repeating: 0, count: 257), im = re
        x.withUnsafeBufferPointer { fe.dft512($0.baseAddress!, re: &re, im: &im) }
        var maxErr = 0.0, maxRef = 0.0
        for k in 0..<257 {
            var r = 0.0, i = 0.0
            for n in 0..<512 { let a = -2 * Double.pi * Double(k * n) / 512; r += Double(x[n]) * cos(a); i += Double(x[n]) * sin(a) }
            maxErr = max(maxErr, abs(Double(re[k]) - r), abs(Double(im[k]) - i)); maxRef = max(maxRef, abs(r), abs(i))
        }
        XCTAssertLessThan(maxErr / maxRef, 1e-5, "vDSP real DFT vs naive DFT")
    }

    func testNativePredictAndJointMatchNaive() throws {
        let tensors = try randomWeights(seed: 7)
        let w = try NativeWeights(tensors: tensors)
        let math = NativeMath(w)
        var h = [Float](repeating: 0, count: 1280), c = h, g = [Float](repeating: 0, count: 640)
        var hd = [Double](repeating: 0, count: 1280), cd = hd
        let p = "decoder.prediction.dec_rnn.lstm."
        func naiveLayer(_ l: Int, x: [Double]) {
            let wih = tensors[p + "weight_ih_l\(l)"]!, whh = tensors[p + "weight_hh_l\(l)"]!
            let bih = tensors[p + "bias_ih_l\(l)"]!, bhh = tensors[p + "bias_hh_l\(l)"]!
            var gates = [Double](repeating: 0, count: 2560)
            for r in 0..<2560 {
                var s = Double(bih[r]) + Double(bhh[r])
                for k in 0..<640 { s += Double(wih[r * 640 + k]) * x[k] + Double(whh[r * 640 + k]) * hd[l * 640 + k] }
                gates[r] = s
            }
            func sig(_ v: Double) -> Double { 1 / (1 + exp(-v)) }
            for k in 0..<640 {
                let cn = sig(gates[640 + k]) * cd[l * 640 + k] + sig(gates[k]) * tanh(gates[1280 + k])
                cd[l * 640 + k] = cn
                hd[l * 640 + k] = sig(gates[1920 + k]) * tanh(cn)
            }
        }
        for token in [1024, 17, 900, 3] {
            h.withUnsafeMutableBufferPointer { hp in c.withUnsafeMutableBufferPointer { cp in g.withUnsafeMutableBufferPointer { gp in
                math.predict(token, h: hp.baseAddress!, c: cp.baseAddress!, g: gp.baseAddress!)
            } } }
            let emb = tensors["decoder.prediction.embed.weight"]!
            naiveLayer(0, x: (0..<640).map { Double(emb[token * 640 + $0]) })
            naiveLayer(1, x: Array(hd[0..<640]))
            let err = zip(h, hd).map { abs(Double($0) - $1) }.max()!
            XCTAssertLessThan(err, 1e-5, "LSTM state after token \(token)")
        }
        // joint against naive
        var gen = LCG(s: 3)
        let enc = (0..<1024 * 3).map { _ in gen.normal() }
        let frames = try EncoderFrames(timeMajor: FloatBuffer(enc), frames: 3)
        var f = [Float](repeating: 0, count: 3 * 640), z = [Float](repeating: 0, count: 640), logits = [Float](repeating: 0, count: 1030)
        f.withUnsafeMutableBufferPointer { math.project(frames, length: 3, into: $0.baseAddress!) }
        _ = f.withUnsafeBufferPointer { fp in g.withUnsafeBufferPointer { gp in z.withUnsafeMutableBufferPointer { zp in
            logits.withUnsafeMutableBufferPointer { lp in math.joint(f: fp.baseAddress! + 640, g: gp.baseAddress!, z: zp.baseAddress!, logits: lp.baseAddress!) }
        } } }
        let encW = tensors["joint.enc.weight"]!, encB = tensors["joint.enc.bias"]!
        let outW = tensors["joint.joint_net.2.weight"]!, outB = tensors["joint.joint_net.2.bias"]!
        var zd = [Double](repeating: 0, count: 640)
        for j in 0..<640 {
            var s = Double(encB[j])
            for k in 0..<1024 { s += Double(encW[j * 1024 + k]) * Double(enc[1024 + k]) }
            zd[j] = max(0, s + Double(g[j]))
        }
        var maxErr = 0.0, rms = 0.0
        for o in 0..<1030 {
            var s = Double(outB[o])
            for j in 0..<640 { s += Double(outW[o * 640 + j]) * zd[j] }
            maxErr = max(maxErr, abs(Double(logits[o]) - s)); rms += s * s
        }
        XCTAssertLessThan(maxErr / (rms / 1030).squareRoot(), 1e-4, "joint logits")
    }

    func makeTrace(_ out: LabelLoop.Output, length: Int, id: String) -> TraceClip {
        TraceClip(id: id, sha256: "x", samples: 0, melFrames: 0, numFrames: length, steps: out.frame.count, frame: out.frame,
                  predInput: out.predInput, token: out.token, duration: out.duration,
                  emitted: out.token.map { $0 != 1024 ? 1 : 0 }, predUpdated: out.token.map { $0 != 1024 ? 1 : 0 },
                  symbolsAtFrame: out.symbolsAtFrame, forcedAdvance: out.forcedAdvance, advance: out.advance,
                  tokens: out.tokens, text: "")
    }

    func testLoopsAgreeAndTracesValidate() throws {
        var emitted = 0, forced = 0
        for seed in [11, 12, 13] as [UInt64] {
            let w = try NativeWeights(tensors: try randomWeights(seed: seed))
            var gen = LCG(s: seed &* 31)
            let length = 40
            let enc = (0..<1024 * length).map { _ in gen.normal() }
            let frames = try EncoderFrames(timeMajor: FloatBuffer(enc), frames: length)
            let loop = LabelLoop()
            let f2 = NativeEngine(weights: w), fused = NativeFusedEngine(weights: w)
            try f2.begin(frames: frames, length: length, diag: nil)
            let a = try loop.free(f2, length: length, diag: nil, recordTrace: true)
            try fused.begin(frames: frames, length: length, diag: nil)
            let b = try loop.free(fused, length: length, diag: nil, recordTrace: true)
            XCTAssertEqual(a.frame, b.frame); XCTAssertEqual(a.token, b.token); XCTAssertEqual(a.duration, b.duration)
            // physical work: F2 runs the prediction net once per emission (+ SOS); the fused call once per step
            XCTAssertEqual(f2.accounting()["native_predict"]!.calls, a.tokens.count + 1)
            XCTAssertEqual(fused.accounting()["native_fused"]!.calls, a.frame.count)
            let clip = Clip(id: "t\(seed)", kind: "test", bucket: 2, length: 0, melFrames: 0, encoderFrames: length,
                            fluidaudioFrames: 0, transcript: nil, sha256: "x")
            let trace = makeTrace(a, length: length, id: clip.id)
            XCTAssertNoThrow(try TraceFile.validate(trace, clip: clip, blank: 1024, durations: [0, 1, 2, 3, 4], maxSymbols: 10))
            // replay of its own trace reproduces every decision and the same logits
            let d1 = DiagSink(), d2 = DiagSink()
            try f2.begin(frames: frames, length: length, diag: nil)
            _ = try loop.free(f2, length: length, diag: d1)
            try f2.begin(frames: frames, length: length, diag: nil)
            let r = try loop.replay(f2, trace: trace, diag: d2)
            XCTAssertEqual(r.tokens, a.tokens)
            XCTAssertEqual(d2.ints["argmax_token"]!, trace.token)
            XCTAssertEqual(d2.ints["argmax_duration"]!, trace.duration)
            XCTAssertEqual(d1.sections.first { $0.name == "logits" }!.data, d2.sections.first { $0.name == "logits" }!.data)
            emitted += a.tokens.count; forced += a.forcedAdvance.reduce(0, +)
        }
        XCTAssertGreaterThan(emitted, 0, "random models should emit tokens")
        print("loop test: \(emitted) emissions, \(forced) forced advances")
    }

    func testMutatedTraceIsRejected() throws {
        let w = try NativeWeights(tensors: try randomWeights(seed: 11))
        var gen = LCG(s: 5)
        let frames = try EncoderFrames(timeMajor: FloatBuffer((0..<1024 * 20).map { _ in gen.normal() }), frames: 20)
        let f2 = NativeEngine(weights: w)
        try f2.begin(frames: frames, length: 20, diag: nil)
        let out = try LabelLoop().free(f2, length: 20, diag: nil, recordTrace: true)
        let clip = Clip(id: "m", kind: "test", bucket: 2, length: 0, melFrames: 0, encoderFrames: 20, fluidaudioFrames: 0,
                        transcript: nil, sha256: "x")
        var bad = out
        bad.advance[0] += 1
        XCTAssertThrowsError(try TraceFile.validate(makeTrace(bad, length: 20, id: "m"), clip: clip, blank: 1024,
                                                    durations: [0, 1, 2, 3, 4], maxSymbols: 10))
        bad = out
        bad.frame.removeLast(); bad.token.removeLast(); bad.duration.removeLast(); bad.predInput.removeLast()
        bad.symbolsAtFrame.removeLast(); bad.forcedAdvance.removeLast(); bad.advance.removeLast()
        XCTAssertThrowsError(try TraceFile.validate(makeTrace(bad, length: 20, id: "m"), clip: clip, blank: 1024,
                                                    durations: [0, 1, 2, 3, 4], maxSymbols: 10))
    }

    // MARK: identity binding (review WP7 r1 findings 2 and 3)

    func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("benchcore-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    func writeBlob(_ dir: URL, stem: String, tensors: [(String, Int)], swapFirstTwo: Bool = false) throws {
        var floats: [Float] = []
        var entries: [[String: Any]] = []
        for (name, n) in tensors {
            entries.append(["name": name, "shape": [n], "offset": floats.count * 4, "bytes": n * 4])
            floats += (0..<n).map { Float($0) + Float(floats.count) }
        }
        if swapFirstTwo {  // same bytes, two equal-sized tensors' offsets exchanged
            let o0 = entries[0]["offset"]!, o1 = entries[1]["offset"]!
            entries[0]["offset"] = o1; entries[1]["offset"] = o0
        }
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        try data.write(to: dir.appendingPathComponent("\(stem).f32bin"))
        let m: [String: Any] = ["file": "\(stem).f32bin", "sha256": sha256Hex(data), "tensors": entries]
        try JSONSerialization.data(withJSONObject: m, options: [.sortedKeys]).write(to: dir.appendingPathComponent("\(stem).json"))
    }

    func testComponentsBindNativeManifests() throws {
        let root = try tempDir()
        let enc = root.appendingPathComponent("mp2/C4/multi.mlmodelc")
        try FileManager.default.createDirectory(at: enc, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: enc.appendingPathComponent("model.mil"))
        let native = root.appendingPathComponent("native")
        try FileManager.default.createDirectory(at: native, withIntermediateDirectories: true)
        try writeBlob(native, stem: "frontend", tensors: [("window", 4), ("fb", 8)])
        try writeBlob(native, stem: "decoder_joint", tensors: [("a", 6), ("b", 6)])
        func comps() throws -> String {
            PipelineEligibility.canonical(try PipelineSpec(
                model: "mp2", arm: "C4", variant: .multifunction, computeUnits: "cpuAndNeuralEngine", frontEnd: "vdsp",
                decode: "f2", encoderPath: enc, frontendConstants: native, c0Dir: nil, nativeWeights: native,
                decoderModels: nil).components())
        }
        let before = try comps()
        XCTAssertEqual(before, try comps())
        try writeBlob(native, stem: "decoder_joint", tensors: [("a", 6), ("b", 6)], swapFirstTwo: true)
        XCTAssertNotEqual(before, try comps(), "swapping equal-sized tensor offsets must change the identity")
        try writeBlob(native, stem: "decoder_joint", tensors: [("a", 6), ("b", 6)])
        XCTAssertEqual(before, try comps())
        try Data([1, 2, 4]).write(to: enc.appendingPathComponent("model.mil"))
        XCTAssertNotEqual(before, try comps(), "a changed encoder file must change the identity")
    }

    func testC0IdentityRejectsTamperingAndExtraFiles() throws {
        let root = try tempDir()
        let dir = root.appendingPathComponent("c0")
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("Encoder.mlmodelc"), withIntermediateDirectories: true)
        let payload = Data("weights".utf8), vocab = Data("{}".utf8)
        try payload.write(to: dir.appendingPathComponent("Encoder.mlmodelc/weight.bin"))
        try vocab.write(to: dir.appendingPathComponent("parakeet_vocab.json"))
        let pinned: [String: Any] = ["repo": "r", "revision": "v", "files": [
            ["path": "Encoder.mlmodelc/weight.bin", "bytes": payload.count, "sha256": sha256Hex(payload)],
            ["path": "parakeet_vocab.json", "bytes": vocab.count, "sha256": sha256Hex(vocab)]]]
        let pin = root.appendingPathComponent("c0.json")
        try JSONSerialization.data(withJSONObject: pinned).write(to: pin)
        XCTAssertEqual(try C0Identity.verify(directory: dir, pinned: pin)["files"] as? Int, 2)
        try Data("weightz".utf8).write(to: dir.appendingPathComponent("Encoder.mlmodelc/weight.bin"))
        XCTAssertThrowsError(try C0Identity.verify(directory: dir, pinned: pin))
        try payload.write(to: dir.appendingPathComponent("Encoder.mlmodelc/weight.bin"))
        try Data("x".utf8).write(to: dir.appendingPathComponent("Encoder.mlmodelc/extra.bin"))
        XCTAssertThrowsError(try C0Identity.verify(directory: dir, pinned: pin))
        try FileManager.default.removeItem(at: dir.appendingPathComponent("Encoder.mlmodelc/extra.bin"))
        try Data("[]".utf8).write(to: dir.appendingPathComponent("parakeet_vocab.json"))
        XCTAssertThrowsError(try C0Identity.verify(directory: dir, pinned: pin), "a different vocabulary must be refused")
    }

    func testDiagSinkCountsNonfinite() {
        let sink = DiagSink()
        sink.append("x", rowShape: [3], [1, .nan, 2])
        sink.float("p", .infinity)
        sink.float("p", 0.5)
        XCTAssertEqual(sink.valueCount, 5)
        XCTAssertEqual(sink.nonfiniteCount, 2)
    }

    /// A terminal prediction (no joint step follows it) with non-finite outputs leaves every decision unchanged, so
    /// only the per-prediction capture can catch it (review WP7 r2 finding 2).
    func testTerminalNonfinitePredictionIsCaptured() throws {
        let clean = try randomWeights(seed: 21)
        var bad = clean
        let token = 7
        var embed = bad["decoder.prediction.embed.weight"]!
        for k in 0..<640 { embed[token * 640 + k] = .nan }
        bad["decoder.prediction.embed.weight"] = embed
        var gen = LCG(s: 5)
        let enc = (0..<1024 * 4).map { _ in gen.normal() }
        let frames = try EncoderFrames(timeMajor: FloatBuffer(enc), frames: 4)
        func run(_ t: [String: [Float]]) throws -> ((Int, Int), DiagSink) {
            let e = NativeEngine(weights: try NativeWeights(tensors: t))
            let sink = DiagSink()
            try e.begin(frames: frames, length: 4, diag: sink)
            try e.predict(1024, diag: sink)
            let d = try e.joint(0, diag: sink)
            try e.predict(token, diag: sink)   // terminal: nothing reads its output
            return ((d.token, d.durationBin), sink)
        }
        let (d0, s0) = try run(clean), (d1, s1) = try run(bad)
        XCTAssertEqual(d0.0, d1.0); XCTAssertEqual(d0.1, d1.1)
        XCTAssertEqual(s0.nonfiniteCount, 0)
        XCTAssertGreaterThan(s1.nonfiniteCount, 0, "the terminal prediction's non-finite g/h/c must be captured")
        XCTAssertEqual(s1.sections.first { $0.name == "pred_g" }!.data.count, 2 * 640)
    }
}
