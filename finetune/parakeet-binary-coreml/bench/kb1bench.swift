// kb1bench: measurement tool of the kea-b1 encoder benchmark. Compiled by kb1bench.py together with LocalFlow's
// own Sources/Parakeet/*.swift and Sources/Dictionary/*.swift (path given to the driver), so `service` and
// `prepare` time the deployed LocalParakeetService, and the other modes reuse its front end, chunking, silence
// compaction, decoder and transcript assembly. Synthetic `say` speech and public LibriSpeech only; prints JSON
// lines (timings, memory, transcripts of that public/synthetic audio) and writes features/encoder outputs only
// where told. Never opens the app, microphone, settings, history or a provider.
//
// Modes (flags are --name value):
//   fixtures --speech AIFF --out DIR --lengths 2,4,... --offsets 0,9.3,...  cut 16 kHz 16-bit WAV clips
//   features --bundle B --list TSV --out DIR          deployed chunking + front end -> mel [128, frames] f32
//   load     --bundle B --units U --order 15,4,8,30   raw MLModel load + warm-up per function (as the runtime)
//   prepare  --bundle B --units U                     LocalParakeetService(.allBuckets).prepare(directory:)
//   encoder  --bundle B --units U --features DIR --keys FILE --warmups N --reps N   MLModel.prediction timings
//   encode   --bundle B --units U --features DIR --keys FILE --out DIR             dump encoder outputs
//   service  --bundle B --units U [--fixtures TSV --warmups N --reps N] [--asr TSV]  transcribe(fileURL:directory:)
//   decode   --bundle B --list TSV --features DIR --encoded DIR   transcripts from precomputed encoder outputs
import AVFoundation
import CoreML
import Darwin
import Foundation

// MARK: - output, clock, memory

func emit(_ object: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

struct KBError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

@inline(__always) func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
@inline(__always) func msSince(_ t0: UInt64) -> Double { Double(nowNs() &- t0) / 1e6 }

func thermal() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

/// TASK_VM_INFO of this process. `phys_footprint` is what Activity Monitor/jetsam call memory; Neural Engine
/// buffers are booked in the `neural_nofootprint` ledger, which phys_footprint excludes, so the total that
/// answers "how much memory does the model take" is phys_footprint + neural_nofootprint (+ its compressed part).
struct VMSample {
    let phys: UInt64, physPeak: Int64, resident: UInt64
    let neuralNo: Int64, neuralNoCompressed: Int64, neuralFoot: Int64, neuralNoPeak: Int64
    var total: UInt64 { phys + UInt64(max(0, neuralNo)) + UInt64(max(0, neuralNoCompressed)) }

    static func read() -> VMSample? {
        var v = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let required = count
        let kr = withUnsafeMutablePointer(to: &v) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS, count >= required else { return nil }
        return VMSample(phys: v.phys_footprint, physPeak: v.ledger_phys_footprint_peak, resident: v.resident_size,
                        neuralNo: v.ledger_tag_neural_nofootprint,
                        neuralNoCompressed: v.ledger_tag_neural_nofootprint_compressed,
                        neuralFoot: v.ledger_tag_neural_footprint, neuralNoPeak: v.ledger_tag_neural_nofootprint_peak)
    }

    var json: [String: Any] {
        ["phys_footprint": phys, "phys_footprint_lifetime_peak": physPeak, "resident": resident,
         "neural_nofootprint": neuralNo, "neural_nofootprint_compressed": neuralNoCompressed,
         "neural_footprint": neuralFoot, "neural_nofootprint_lifetime_peak": neuralNoPeak,
         "footprint_plus_neural": total]
    }
}

/// System-wide wired memory (host_statistics64): ANE/GPU driver allocations land here, but so does everything
/// else on the machine, so a delta is only indicative on a shared Mac.
func wiredBytes() -> UInt64 {
    var stats = vm_statistics64_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
    }
    var page: vm_size_t = 0
    host_page_size(mach_host_self(), &page)
    return kr == KERN_SUCCESS ? UInt64(stats.wire_count) * UInt64(page) : 0
}

/// RSS (KiB) of the system ML daemons by name, from ps (other users' processes are visible). The ANE daemon and
/// compilers are shared with every other client on the machine, so deltas are indicative only.
func daemonRSS() -> [String: Int] {
    var totals = ["aned": 0, "ANECompilerService": 0, "MTLCompilerService": 0]
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/bin/ps")
    p.arguments = ["-axo", "rss=,comm="]
    let pipe = Pipe()
    p.standardOutput = pipe
    guard (try? p.run()) != nil else { return [:] }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
        let parts = line.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true)
        guard parts.count == 2, let kib = Int(parts[0]) else { continue }
        let name = (parts[1].trimmingCharacters(in: .whitespaces) as NSString).lastPathComponent
        if totals[name] != nil { totals[name]! += kib }
    }
    return totals
}

/// Samples VMSample every `intervalMs` on its own thread; per phase: samples, first, last and max of
/// phys_footprint and footprint_plus_neural. Peaks are sampled, so sub-interval transients can be missed.
final class MemorySampler: @unchecked Sendable {
    struct Phase { var name: String; var samples = 0; var first: VMSample?; var last: VMSample?
        var maxPhys: UInt64 = 0; var maxTotal: UInt64 = 0; var wiredStart: UInt64 = 0; var wiredEnd: UInt64 = 0 }
    private let lock = NSLock()
    private var phases: [Phase] = []
    private var stopped = false
    private let intervalUs: useconds_t

    init(intervalMs: Double = 5, phase: String = "baseline") {
        intervalUs = useconds_t(intervalMs * 1000)
        phases = [Phase(name: phase, wiredStart: wiredBytes())]
        let thread = Thread { [weak self] in
            while let self, !self.isStopped { self.sample(); usleep(self.intervalUs) }
        }
        thread.qualityOfService = .utility
        thread.start()
        sample()
    }

    private var isStopped: Bool { lock.lock(); defer { lock.unlock() }; return stopped }

    func sample() {
        guard let s = VMSample.read() else { return }
        lock.lock()
        var p = phases[phases.count - 1]
        p.samples += 1
        if p.first == nil { p.first = s }
        p.last = s
        p.maxPhys = max(p.maxPhys, s.phys)
        p.maxTotal = max(p.maxTotal, s.total)
        phases[phases.count - 1] = p
        lock.unlock()
    }

    func mark(_ name: String) {
        sample()
        let wired = wiredBytes()
        lock.lock()
        phases[phases.count - 1].wiredEnd = wired
        phases.append(Phase(name: name, wiredStart: wired))
        lock.unlock()
        sample()
    }

    func stop() -> [[String: Any]] {
        sample()
        let wired = wiredBytes()
        lock.lock()
        stopped = true
        phases[phases.count - 1].wiredEnd = wired
        let out = phases.map { p -> [String: Any] in
            ["phase": p.name, "samples": p.samples, "first": p.first?.json ?? [:], "last": p.last?.json ?? [:],
             "max_phys_footprint": p.maxPhys, "max_footprint_plus_neural": p.maxTotal,
             "system_wired_start": p.wiredStart, "system_wired_end": p.wiredEnd]
        }
        lock.unlock()
        return out
    }
}

// MARK: - arguments and shared inputs

struct Args {
    let mode: String
    private var values: [String: String] = [:]
    init(_ argv: [String]) throws {
        guard let first = argv.first else { throw KBError("usage: kb1bench MODE --flag value ...") }
        mode = first
        var i = 1
        while i < argv.count {
            guard argv[i].hasPrefix("--"), i + 1 < argv.count else { throw KBError("bad argument \(argv[i])") }
            values[String(argv[i].dropFirst(2))] = argv[i + 1]
            i += 2
        }
    }
    func string(_ k: String) throws -> String { guard let v = values[k] else { throw KBError("missing --\(k)") }; return v }
    func optional(_ k: String) -> String? { values[k] }
    func url(_ k: String) throws -> URL { URL(fileURLWithPath: try string(k)) }
    func int(_ k: String, _ d: Int) throws -> Int {
        guard let v = values[k] else { return d }
        guard let i = Int(v) else { throw KBError("--\(k) must be an integer") }
        return i
    }
    func units() throws -> MLComputeUnits {
        switch try string("units") {
        case "cpuAndNeuralEngine": return .cpuAndNeuralEngine
        case "cpuAndGPU": return .cpuAndGPU
        case "cpuOnly": return .cpuOnly
        case "all": return .all
        case let u: throw KBError("unknown compute units \(u)")
        }
    }
}

/// id <TAB> path lines.
func readList(_ url: URL) throws -> [(id: String, path: URL)] {
    try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map { line in
        let parts = line.split(separator: "\t", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { throw KBError("list lines are id<TAB>path") }
        return (parts[0], URL(fileURLWithPath: parts[1]))
    }
}

func readSamples(_ url: URL) throws -> [Float] {
    var all: [Float] = []
    try ParakeetAudioReader.read(fileURL: url, check: {}) { all.append(contentsOf: $0) }
    return all
}

/// The runtime's chunk size (LocalParakeetCore.chunkSamples: the 15 s encoder function since LocalFlow 897f13e).
let chunkSamples = LocalParakeetCore.chunkSamples

/// Runs `body` on every chunk exactly as LocalParakeetRuntime.transcribe does (silence compaction, then 30 s
/// windows with 2 s overlap) and returns the assembled transcript of the words `body` returns.
func chunked(_ samples: [Float], _ body: (Int, ParakeetChunk) throws -> [ParakeetWord]) throws -> String {
    var transcription = ParakeetChunkedTranscription(size: chunkSamples, overlap: LocalParakeetCore.chunkOverlapSamples,
                                                     compactor: ParakeetSilenceCompactor())
    var k = 0
    try transcription.append(samples) { chunk in defer { k += 1 }; return try body(k, chunk) }
    return try transcription.finish { chunk in defer { k += 1 }; return try body(k, chunk) }
}

struct FeatureRow {
    let key: String, bucket: Int, frames: Int, valid: Int
}

func featureManifest(_ dir: URL) throws -> [String: FeatureRow] {
    var out: [String: FeatureRow] = [:]
    for line in try String(contentsOf: dir.appendingPathComponent("manifest.jsonl"), encoding: .utf8).split(separator: "\n") {
        guard let o = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
              let key = o["key"] as? String else { throw KBError("bad features manifest") }
        if o["silent"] as? Bool == true { continue }
        guard let b = o["bucket"] as? Int, let f = o["frames"] as? Int, let v = o["valid"] as? Int else {
            throw KBError("bad features manifest row")
        }
        out[key] = FeatureRow(key: key, bucket: b, frames: f, valid: v)
    }
    return out
}

/// Model input of one chunk: mel [1, 128, 100 b + 1] zero-padded after the stored frames, and mel_length.
func melInput(_ dir: URL, _ row: FeatureRow) throws -> MLDictionaryFeatureProvider {
    let data = try Data(contentsOf: dir.appendingPathComponent("\(row.key).mel"))
    guard data.count == 128 * row.frames * 4 else { throw KBError("feature file size of \(row.key)") }
    let size = row.bucket * 100 + 1
    let mel = try MLMultiArray(shape: [1, 128, NSNumber(value: size)], dataType: .float32)
    let dst = mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count)
    dst.initialize(repeating: 0, count: mel.count)
    data.withUnsafeBytes { raw in
        let src = raw.bindMemory(to: Float.self).baseAddress!
        for c in 0..<128 { (dst + c * size).update(from: src + c * row.frames, count: row.frames) }
    }
    let length = try MLMultiArray(shape: [1], dataType: .int32)
    length[0] = NSNumber(value: row.valid)
    return try MLDictionaryFeatureProvider(dictionary: ["mel": mel, "mel_length": length])
}

func readKeys(_ url: URL) throws -> [String] {
    try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map(String.init).filter { !$0.isEmpty }
}

// MARK: - encoder (mirrors LocalParakeetService's private ParakeetEncoder)

func loadEncoder(_ bundle: URL, bucket: Int, units: MLComputeUnits) throws -> MLModel {
    let config = MLModelConfiguration()
    config.computeUnits = units
    config.functionName = "b\(bucket)"
    return try MLModel(contentsOf: bundle.appendingPathComponent("Encoder.mlmodelc"), configuration: config)
}

func warmEncoder(_ model: MLModel, bucket: Int) throws {
    let size = bucket * 100 + 1
    let mel = try MLMultiArray(shape: [1, 128, NSNumber(value: size)], dataType: .float32)
    mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count).initialize(repeating: 0, count: mel.count)
    let length = try MLMultiArray(shape: [1], dataType: .int32)
    length[0] = NSNumber(value: size)
    _ = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["mel": mel, "mel_length": length]))
}

func loadAll(_ bundle: URL, units: MLComputeUnits, order: [Int], sampler: MemorySampler?) throws -> ([Int: MLModel], [[String: Any]], Double) {
    var models: [Int: MLModel] = [:], rows: [[String: Any]] = []
    let start = nowNs()
    for b in order {
        sampler?.mark("load_b\(b)")
        let t0 = nowNs()
        let m = try loadEncoder(bundle, bucket: b, units: units)
        let loadMs = msSince(t0)
        let t1 = nowNs()
        try warmEncoder(m, bucket: b)
        let warmMs = msSince(t1)
        models[b] = m
        rows.append(["bucket": b, "load_ms": loadMs, "first_prediction_ms": warmMs, "ready_ms": loadMs + warmMs,
                     "ready_since_start_ms": msSince(start)])
    }
    return (models, rows, msSince(start))
}

// MARK: - modes

func modeFixtures(_ a: Args) throws {
    let samples = try readSamples(try a.url("speech"))
    let out = try a.url("out")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let lengths = try a.string("lengths").split(separator: ",").map { Double($0)! }
    let offsets = try a.string("offsets").split(separator: ",").map { Double($0)! }
    let rate = 16000.0
    for length in lengths {
        for (k, offset) in offsets.enumerated() {
            let start = Int(offset * rate), count = Int(length * rate)
            guard start + count <= samples.count else { throw KBError("speech too short for \(length) s at \(offset) s") }
            let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: rate, AVNumberOfChannelsKey: 1,
                                           AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
                                           AVLinearPCMIsNonInterleaved: false]
            let id = "e2e-\(Int(length))s-\(k)"
            let url = out.appendingPathComponent("\(id).wav")
            let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
            let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(count))!
            buffer.frameLength = AVAudioFrameCount(count)
            samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress! + start, count: count) }
            try file.write(from: buffer)
            emit(["type": "fixture", "id": id, "path": url.path, "seconds": length, "offset_seconds": offset])
        }
    }
}

func modeFeatures(_ a: Args) throws {
    let bundle = try a.url("bundle"), out = try a.url("out")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let frontend = try VDSPFrontEnd(constantsDir: bundle)
    var manifest = ""
    for item in try readList(try a.url("list")) {
        let samples = try readSamples(item.path)
        _ = try chunked(samples) { k, chunk in
            let key = "\(item.id).c\(k)"
            var row: [String: Any] = ["key": key, "id": item.id, "chunk": k, "start": chunk.start, "samples": chunk.samples.count,
                                      "audio_samples": samples.count]
            // LocalParakeetRuntime.project: digital silence has no model call; < 480 samples are zero-padded.
            if chunk.samples.allSatisfy({ $0 == 0 }) {
                row["silent"] = true
            } else {
                let pcm = chunk.samples.count < 480 ? chunk.samples + [Float](repeating: 0, count: 480 - chunk.samples.count) : chunk.samples
                let bucket = try LocalParakeetCore.bucket(samples: pcm.count)
                let f = frontend.compute(pcm)
                try f.features.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: out.appendingPathComponent("\(key).mel"))
                row.merge(["silent": false, "bucket": bucket, "frames": f.frames, "valid": f.valid]) { $1 }
            }
            manifest += String(decoding: try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), as: UTF8.self) + "\n"
            return []
        }
    }
    try manifest.write(to: out.appendingPathComponent("manifest.jsonl"), atomically: true, encoding: .utf8)
    emit(["type": "features", "chunks": manifest.split(separator: "\n").count])
}

func modeLoad(_ a: Args) throws {
    let sampler = MemorySampler()
    usleep(300_000)
    let order = try a.string("order").split(separator: ",").map { Int($0)! }
    let thermalStart = thermal()
    let daemonsBefore = daemonRSS()
    let (models, rows, total) = try loadAll(try a.url("bundle"), units: try a.units(), order: order, sampler: sampler)
    sampler.mark("loaded_idle")
    usleep(1_000_000)
    let daemonsAfter = daemonRSS()
    emit(["type": "load", "functions": rows, "all_ready_ms": total, "thermal_start": thermalStart, "thermal_end": thermal(),
          "daemons_kib": ["before": daemonsBefore, "loaded": daemonsAfter], "memory": sampler.stop(), "models": models.count])
}

func modePrepare(_ a: Args) async throws {
    let sampler = MemorySampler()
    usleep(300_000)
    let bundle = try a.url("bundle")
    let service = LocalParakeetService(computeUnits: try a.units())
    let daemonsBefore = daemonRSS()
    sampler.mark("prepare")
    let t0 = nowNs()
    try await service.prepare(directory: bundle)
    let ms = msSince(t0)
    sampler.mark("prepared_idle")
    usleep(1_000_000)
    let daemonsAfter = daemonRSS()
    emit(["type": "prepare", "all_ready_ms": ms, "thermal": thermal(), "daemons_kib": ["before": daemonsBefore, "loaded": daemonsAfter],
          "memory": sampler.stop()])
    withExtendedLifetime(service) {}
}

/// The encoder functions to load (--order, default all four); a b15-only bundle passes "15".
func functionOrder(_ a: Args) throws -> [Int] {
    try (a.optional("order") ?? "15,4,8,30").split(separator: ",").map {
        guard let b = Int($0) else { throw KBError("--order must list integers") }
        return b
    }
}

func modeEncoder(_ a: Args) throws {
    let sampler = MemorySampler()
    let features = try a.url("features")
    let manifest = try featureManifest(features)
    let keys = try readKeys(try a.url("keys"))
    let warmups = try a.int("warmups", 3), reps = try a.int("reps", 10)
    let (models, loadRows, loadMs) = try loadAll(try a.url("bundle"), units: try a.units(), order: try functionOrder(a), sampler: sampler)
    emit(["type": "encoder_load", "functions": loadRows, "all_ready_ms": loadMs])
    sampler.mark("predict")
    for b in [4, 8, 15, 30] {  // every function the bundle holds (the app runs only b15 since LocalFlow 897f13e)
        let rows = keys.compactMap { manifest[$0] }.filter { $0.bucket == b }
        guard !rows.isEmpty, let model = models[b] else { continue }
        let t0 = thermal()
        for row in rows {
            let input = try melInput(features, row)
            for rep in 0..<(warmups + reps) {
                let start = nowNs()
                let out = try model.prediction(from: input)
                let ms = msSince(start)
                guard out.featureValue(for: "encoder")?.multiArrayValue != nil else { throw KBError("encoder output missing") }
                emit(["type": "encoder_call", "bucket": b, "clip": row.key, "rep": rep, "warmup": rep < warmups, "ms": ms])
            }
        }
        emit(["type": "thermal_block", "bucket": b, "thermal_start": t0, "thermal_end": thermal()])
    }
    emit(["type": "memory", "memory": sampler.stop()])
}

func modeEncode(_ a: Args) throws {
    let features = try a.url("features"), out = try a.url("out")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let manifest = try featureManifest(features)
    let (models, _, _) = try loadAll(try a.url("bundle"), units: try a.units(), order: try functionOrder(a), sampler: nil)
    var lines = ""
    for key in try readKeys(try a.url("keys")) {
        guard let row = manifest[key], let model = models[row.bucket] else { throw KBError("no features for \(key)") }
        let result = try model.prediction(from: try melInput(features, row))
        guard let enc = result.featureValue(for: "encoder")?.multiArrayValue,
              let len = result.featureValue(for: "encoder_length")?.multiArrayValue else { throw KBError("encoder output missing") }
        let frames = try EncoderFrames(enc, validLength: len[0].intValue)
        let values = try frames.timeMajor()
        try values.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: out.appendingPathComponent("\(key).f32"))
        lines += "{\"frames\":\(frames.count),\"key\":\"\(key)\"}\n"
    }
    try lines.write(to: out.appendingPathComponent("manifest.jsonl"), atomically: true, encoding: .utf8)
    emit(["type": "encode", "chunks": lines.split(separator: "\n").count])
}

final class BucketTrace: @unchecked Sendable {
    private let lock = NSLock()
    private var buckets: [Int] = []
    func record(_ b: Int) { lock.lock(); buckets.append(b); lock.unlock() }
    func take() -> [Int] { lock.lock(); defer { lock.unlock() }; let r = buckets; buckets.removeAll(); return r }
}

func modeService(_ a: Args) async throws {
    let sampler = MemorySampler()
    usleep(300_000)
    let bundle = try a.url("bundle")
    let trace = BucketTrace()
    let service = LocalParakeetService(computeUnits: try a.units(), onBucketUsed: { trace.record($0) })
    sampler.mark("prepare")
    let t0 = nowNs()
    try await service.prepare(directory: bundle)
    emit(["type": "service_prepare", "all_ready_ms": msSince(t0)])
    _ = trace.take()
    if let fixtures = a.optional("fixtures") {
        let list = try readList(URL(fileURLWithPath: fixtures))
        let warmups = try a.int("warmups", 2), reps = try a.int("reps", 5)
        sampler.mark("transcribe")
        // Interleaved: each pass transcribes every clip once, so a clip's calls spread over the process and
        // consecutive calls switch encoder functions as successive dictations would.
        var states: [String] = [thermal()]
        for rep in 0..<(warmups + reps) {
            for item in list {
                let start = nowNs()
                let text = try await service.transcribe(fileURL: item.path, directory: bundle)
                let ms = msSince(start)
                emit(["type": "e2e_call", "clip": item.id, "rep": rep, "warmup": rep < warmups, "ms": ms,
                      "buckets": trace.take(), "text": text])
            }
            states.append(thermal())
        }
        emit(["type": "thermal", "states": states])
    }
    if let asr = a.optional("asr") {
        sampler.mark("asr")
        for item in try readList(URL(fileURLWithPath: asr)) {
            let start = nowNs()
            let text = try await service.transcribe(fileURL: item.path, directory: bundle)
            emit(["type": "asr", "id": item.id, "text": text, "ms": msSince(start), "buckets": trace.take()])
        }
    }
    emit(["type": "memory", "memory": sampler.stop()])
}

/// Transcripts from precomputed encoder outputs (`encode`, or an external encoder writing the same files):
/// LocalParakeetRuntime's chunking, then per chunk the bundle's native joint projection, unbiased greedy TDT
/// decode and word timing (mirroring its private no-dictionary path), joined by ParakeetTranscriptAssembler.
func modeDecode(_ a: Args) throws {
    let bundle = try a.url("bundle"), encoded = try a.url("encoded")
    let math = try NativeMath(NativeWeights(directory: bundle))
    let vocab = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: bundle.appendingPathComponent("vocabulary.json")))
    let vocabulary = (0..<1024).map { vocab[String($0)]! }
    for item in try readList(try a.url("list")) {
        let text = try chunked(try readSamples(item.path)) { k, chunk in
            if chunk.samples.allSatisfy({ $0 == 0 }) { return [] }
            let data = try Data(contentsOf: encoded.appendingPathComponent("\(item.id).c\(k).f32"))
            guard data.count > 0, data.count % 4096 == 0 else { throw KBError("encoder output size of \(item.id).c\(k)") }
            let frames = data.count / 4096
            let buffer = FloatBuffer(count: frames * 1024)
            data.withUnsafeBytes { buffer.pointer.update(from: $0.bindMemory(to: Float.self).baseAddress!, count: frames * 1024) }
            let view = try EncoderFrames(timeMajor: buffer, frames: frames)
            let projection = FloatBuffer(count: frames * 640)
            math.project(view, length: frames, into: projection.pointer)
            var h = [Float](repeating: 0, count: 1280), c = h
            var g = [Float](repeating: 0, count: 640), z = g, logits = [Float](repeating: 0, count: 1030)
            let tokens = try LocalParakeetCore.decode(length: frames, predict: { token in
                math.predict(token, h: &h, c: &c, g: &g)
            }, joint: { frame in
                let decision = math.joint(f: projection.pointer + frame * 640, g: &g, z: &z, logits: &logits)
                guard logits.allSatisfy(\.isFinite) else { throw KBError("decoder produced invalid values") }
                return decision
            }, check: {})
            return try LocalParakeetCore.words(tokens, vocabulary: vocabulary, offset: chunk.start)
        }
        emit(["type": "asr", "id": item.id, "text": text])
    }
}

@main
struct KB1Bench {
    static func main() async {
        do {
            let a = try Args(Array(CommandLine.arguments.dropFirst()))
            switch a.mode {
            case "fixtures": try modeFixtures(a)
            case "features": try modeFeatures(a)
            case "load": try modeLoad(a)
            case "prepare": try await modePrepare(a)
            case "encoder": try modeEncoder(a)
            case "encode": try modeEncode(a)
            case "service": try await modeService(a)
            case "decode": try modeDecode(a)
            default: throw KBError("unknown mode \(a.mode)")
            }
        } catch {
            FileHandle.standardError.write("kb1bench failed: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }
}
