// b1bench: microbenchmark of 1-bit-weight x FP16-activation GEMMs (MB1 shapes) on the Apple GPU.
//
//   b1bench check   METAL_SRC OUT.json            exactness of every variant vs a CPU reference (bitwise)
//   b1bench peak    METAL_SRC OUT.json            FP32 FMA and simdgroup-MMA throughput probes (clock / ALU peak)
//   b1bench bench   METAL_SRC OUT.json [reps]     every variant x shape x M; MPS dense FP16 GEMM as a reference
//   b1bench encoder METAL_SRC OUT.json BENCH.json 240 projection GEMMs of one encoder pass per bucket (best variant
//                                                 per shape from BENCH.json), distinct weights per layer
// Synthetic data only (seeded xorshift). Times are GPU times (command buffer gpuStartTime..gpuEndTime).
import Foundation
import Metal
import MetalPerformanceShaders

struct Params {
    var M: UInt32
    var N: UInt32
    var K: UInt32
}

struct Rng {
    var s: UInt64
    mutating func next() -> UInt64 {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17
        return s
    }
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
    mutating func normal() -> Float {
        let u1 = max(unit(), 1e-7), u2 = unit()
        return sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
    }
}

struct Variant {
    let name: String  // e.g. mma_2_4_2_2
    let bm: Int
    let bn: Int
    let threads: Int
    init(_ name: String) {
        self.name = name
        let p = name.split(separator: "_").dropFirst().map { Int($0)! }
        switch name.split(separator: "_")[0] {
        case "mma": bm = 8 * p[0] * p[2]; bn = 8 * p[1] * p[3]; threads = 32 * p[2] * p[3]
        case "simt": bm = p[0] * p[2]; bn = 32 * p[1] * p[3]; threads = 32 * p[2] * p[3]
        case "lut": bm = p[0]; bn = 32 * p[1] * p[2]; threads = 32 * p[2]
        default: fatalError(name)
        }
    }
}

let VARIANTS = [
    "mma_1_4_2_2", "mma_2_4_1_2", "mma_2_4_2_2", "mma_2_4_1_4", "mma_4_4_1_2", "mma_4_4_2_2", "mma_4_2_2_2",
    "mma_2_8_2_1", "mma_4_8_1_2", "mma_8_4_1_2",
    "simt_8_4_1_2", "simt_8_4_2_2", "simt_8_8_1_1", "simt_16_4_1_2", "simt_4_8_2_2",
    "lut_8_1_4", "lut_8_2_4", "lut_16_1_4", "lut_4_2_4",
].map(Variant.init)

// MB1 projection shapes (N, K) and LocalFlow buckets (encoder frames M)
let SHAPES: [(String, Int, Int)] = [("ff_linear1", 4096, 1024), ("ff_linear2", 1024, 4096),
                                    ("attn_qkvo_pw2", 1024, 1024), ("pw1", 2048, 1024)]
let BUCKET_M: [(Int, Int)] = [(4, 51), (8, 101), (15, 188), (30, 376)]
// per layer, in order: ff1.l1, ff1.l2, q, k, v, out, pw1, pw2, ff2.l1, ff2.l2
let LAYER: [(Int, Int)] = [(4096, 1024), (1024, 4096), (1024, 1024), (1024, 1024), (1024, 1024), (1024, 1024),
                           (2048, 1024), (1024, 1024), (4096, 1024), (1024, 4096)]

// --- peak -----------------------------------------------------------------------------------------------------
let PEAK_SRC = """

kernel void peak_fma(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    float a0 = gid * 1e-9f, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    const float b = 0.999f, c = 1e-7f;
    for (uint i = 0; i < iters; ++i) {
        a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
    }
    out[gid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
kernel void peak_mma(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]], uint gid [[thread_position_in_grid]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    simdgroup_matrix<float, 8, 8> a(0.5f), b(0.999f), c0(0.0f), c1(0.0f), c2(0.0f), c3(0.0f);
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0); simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2); simdgroup_multiply_accumulate(c3, a, b, c3);
    }
    out[gid] = c0.thread_elements()[0] + c1.thread_elements()[0] + c2.thread_elements()[0] + c3.thread_elements()[0];
}
kernel void peak_mma_h(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]], uint gid [[thread_position_in_grid]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    simdgroup_matrix<half, 8, 8> a(0.5h), b(0.999h), c0(0.0h), c1(0.0h), c2(0.0h), c3(0.0h);
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(c0, a, b, c0); simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2); simdgroup_multiply_accumulate(c3, a, b, c3);
    }
    out[gid] = float(c0.thread_elements()[0] + c1.thread_elements()[0] + c2.thread_elements()[0] + c3.thread_elements()[0]);
}
"""

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let args = CommandLine.arguments
guard args.count >= 4 else {
    print("usage: b1bench check|peak|bench|encoder METAL_SRC OUT.json [...]"); exit(2)
}
let mode = args[1]
let source = try! String(contentsOfFile: args[2], encoding: .utf8)
let opts = MTLCompileOptions()
let library: MTLLibrary
do { library = try device.makeLibrary(source: source + PEAK_SRC, options: opts) } catch {
    print("compile error: \(error)"); exit(1)
}
var pipelines: [String: MTLComputePipelineState] = [:]
func pipeline(_ name: String) -> MTLComputePipelineState {
    if let p = pipelines[name] { return p }
    let f = library.makeFunction(name: name)!
    let p = try! device.makeComputePipelineState(function: f)
    pipelines[name] = p
    return p
}

func buffer<T>(_ a: [T]) -> MTLBuffer {
    a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 16), options: .storageModeShared)! }
}

struct Problem {
    let M: Int, N: Int, K: Int
    let x: MTLBuffer, w: MTLBuffer, s: MTLBuffer, y: MTLBuffer
}

func makeProblem(M: Int, N: Int, K: Int, seed: UInt64, intInputs: Bool, outBytes: Int = 2) -> (Problem, [Float16], [UInt32], [Float16]) {
    var r = Rng(s: seed | 1)
    var xs = [Float16](repeating: 0, count: M * K)
    for i in 0..<xs.count { xs[i] = intInputs ? Float16(Int(r.next() % 7) - 3) : Float16(r.normal()) }
    var ws = [UInt32](repeating: 0, count: N * K / 32)
    for i in 0..<ws.count { ws[i] = UInt32(truncatingIfNeeded: r.next() >> 11) }
    var ss = [Float16](repeating: 0, count: N)
    for i in 0..<N { ss[i] = Float16(0.005 + 0.05 * r.unit()) }
    let y = device.makeBuffer(length: max(M * N * outBytes, 16), options: .storageModeShared)!
    return (Problem(M: M, N: N, K: K, x: buffer(xs), w: buffer(ws), s: buffer(ss), y: y), xs, ws, ss)
}

func encode(_ enc: MTLComputeCommandEncoder, _ v: Variant, _ suffix: String, _ p: Problem) {
    enc.setComputePipelineState(pipeline("b1_\(v.name)_\(suffix)"))
    enc.setBuffer(p.x, offset: 0, index: 0)
    enc.setBuffer(p.w, offset: 0, index: 1)
    enc.setBuffer(p.s, offset: 0, index: 2)
    enc.setBuffer(p.y, offset: 0, index: 3)
    var prm = Params(M: UInt32(p.M), N: UInt32(p.N), K: UInt32(p.K))
    enc.setBytes(&prm, length: MemoryLayout<Params>.stride, index: 4)
    enc.dispatchThreadgroups(MTLSize(width: p.N / v.bn, height: (p.M + v.bm - 1) / v.bm, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: v.threads, height: 1, depth: 1))
}

func gpuTime(_ body: (MTLCommandBuffer) -> Void) -> Double {
    let cb = queue.makeCommandBuffer()!
    body(cb)
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error { print("command buffer error: \(e)"); exit(1) }
    return cb.gpuEndTime - cb.gpuStartTime
}

func median(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }

func writeJSON(_ obj: Any, _ path: String) {
    let d = try! JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
    try! d.write(to: URL(fileURLWithPath: path))
    print(String(data: d, encoding: .utf8)!.prefix(4000))
}

func deviceInfo() -> [String: Any] {
    ["name": device.name, "maxThreadgroupMemory": device.maxThreadgroupMemoryLength,
     "recommendedMaxWorkingSet": device.recommendedMaxWorkingSetSize]
}

// --- check ----------------------------------------------------------------------------------------------------
func check() {
    let shapes = [(7, 1024, 1024), (51, 1024, 1024), (51, 2048, 1024), (51, 1024, 4096), (101, 1024, 1024),
                  (188, 1024, 1024), (376, 1024, 1024), (9, 1024, 4096), (40, 4096, 1024)]
    var out: [[String: Any]] = []
    var allOK = true
    for (si, (M, N, K)) in shapes.enumerated() {
        let (p32, xs, ws, ss) = makeProblem(M: M, N: N, K: K, seed: UInt64(1000 + si), intInputs: true, outBytes: 4)
        let p16 = Problem(M: M, N: N, K: K, x: p32.x, w: p32.w, s: p32.s,
                          y: device.makeBuffer(length: M * N * 2, options: .storageModeShared)!)
        // exact reference: integer sums, one FP32 rounding for the scale, FP16 rounding for half output
        var ref32 = [Float](repeating: 0, count: M * N)
        let KW = K / 32
        xs.withUnsafeBufferPointer { xp in
            ws.withUnsafeBufferPointer { wp in
                for n in 0..<N {
                    for m in 0..<M {
                        var sum: Int32 = 0
                        for kw in 0..<KW {
                            let word = wp[n * KW + kw]
                            for b in 0..<32 {
                                let xv = Int32(Float(xp[m * K + kw * 32 + b]))
                                sum += ((word >> UInt32(b)) & 1) == 1 ? xv : -xv
                            }
                        }
                        ref32[m * N + n] = Float(sum) * Float(ss[n])
                    }
                }
            }
        }
        for v in VARIANTS {
            if N % v.bn != 0 { continue }
            _ = gpuTime { cb in let e = cb.makeComputeCommandEncoder()!; encode(e, v, "f", p32); encode(e, v, "h", p16); e.endEncoding() }
            let g32 = p32.y.contents().bindMemory(to: Float.self, capacity: M * N)
            let g16 = p16.y.contents().bindMemory(to: Float16.self, capacity: M * N)
            var bad32 = 0, bad16 = 0
            for i in 0..<(M * N) {
                if g32[i].bitPattern != ref32[i].bitPattern { bad32 += 1 }
                if g16[i].bitPattern != Float16(ref32[i]).bitPattern { bad16 += 1 }
            }
            if bad32 + bad16 > 0 { allOK = false }
            out.append(["M": M, "N": N, "K": K, "variant": v.name, "mismatch_f32": bad32, "mismatch_f16": bad16])
        }
        print("checked \(M)x\(N)x\(K)")
    }
    writeJSON(["mode": "check", "device": deviceInfo(), "pass": allOK, "cases": out,
               "inputs": "integer activations in [-3, 3] (sums exact in FP32), random bits, FP16 scales in [0.005, 0.055]"],
              args[3])
    exit(allOK ? 0 : 1)
}

func peak() {
    let threads = 16 * 1024 * 8
    let outb = device.makeBuffer(length: threads * 4, options: .storageModeShared)!
    var res: [String: Any] = ["mode": "peak", "device": deviceInfo()]
    for (name, flopsPerIter) in [("peak_fma", 16.0), ("peak_mma", 4.0 * 1024 / 32), ("peak_mma_h", 4.0 * 1024 / 32)] {
        // flops per thread per iteration: FMA 8 x 2; MMA 4 x (8x8x8 x 2) / 32 lanes
        let ps = pipeline(name)
        var iters: UInt32 = 4096
        var ts: [Double] = []
        for _ in 0..<5 {
            ts.append(gpuTime { cb in
                let e = cb.makeComputeCommandEncoder()!
                e.setComputePipelineState(ps); e.setBuffer(outb, offset: 0, index: 0)
                e.setBytes(&iters, length: 4, index: 1)
                e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                e.endEncoding()
            })
        }
        let t = median(ts)
        res[name + "_tflops"] = Double(threads) * Double(iters) * flopsPerIter / t / 1e12
    }
    writeJSON(res, args[3])
}

// --- bench ----------------------------------------------------------------------------------------------------
func bench() {
    let reps = args.count > 4 ? Int(args[4])! : 5
    let perCB = 20
    var rows: [[String: Any]] = []
    for (sname, N, K) in SHAPES {
        for (bucket, M) in BUCKET_M {
            let (p, xs, ws, _) = makeProblem(M: M, N: N, K: K, seed: UInt64(N * 7 + K + M), intInputs: false)
            let flops = 2.0 * Double(M) * Double(N) * Double(K)
            for v in VARIANTS where N % v.bn == 0 {
                if pipeline("b1_\(v.name)_h").maxTotalThreadsPerThreadgroup < v.threads { continue }
                _ = gpuTime { cb in let e = cb.makeComputeCommandEncoder()!; for _ in 0..<3 { encode(e, v, "h", p) }; e.endEncoding() }
                var ts: [Double] = []
                for _ in 0..<reps {
                    ts.append(gpuTime { cb in
                        let e = cb.makeComputeCommandEncoder()!
                        for _ in 0..<perCB { encode(e, v, "h", p) }
                        e.endEncoding()
                    } / Double(perCB))
                }
                let t = median(ts)
                rows.append(["shape": sname, "N": N, "K": K, "M": M, "bucket_s": bucket, "variant": v.name,
                             "us": t * 1e6, "tflops": flops / t / 1e12])
            }
            // MPS dense FP16 reference: Y = X [M,K] x Wd^T, Wd [N,K] half (codes x 1)
            var wd = [Float16](repeating: 0, count: N * K)
            let KW = K / 32
            for n in 0..<N { for k in 0..<K { wd[n * K + k] = ((ws[n * KW + k / 32] >> UInt32(k % 32)) & 1) == 1 ? 1 : -1 } }
            let wb = buffer(wd)
            let xb = buffer(xs)
            let yb = device.makeBuffer(length: M * N * 2, options: .storageModeShared)!
            let mx = MPSMatrix(buffer: xb, descriptor: MPSMatrixDescriptor(rows: M, columns: K, rowBytes: K * 2, dataType: .float16))
            let mw = MPSMatrix(buffer: wb, descriptor: MPSMatrixDescriptor(rows: N, columns: K, rowBytes: K * 2, dataType: .float16))
            let my = MPSMatrix(buffer: yb, descriptor: MPSMatrixDescriptor(rows: M, columns: N, rowBytes: N * 2, dataType: .float16))
            let mm = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true, resultRows: M,
                                             resultColumns: N, interiorColumns: K, alpha: 1, beta: 0)
            _ = gpuTime { cb in mm.encode(commandBuffer: cb, leftMatrix: mx, rightMatrix: mw, resultMatrix: my) }
            var ts: [Double] = []
            for _ in 0..<reps {
                ts.append(gpuTime { cb in for _ in 0..<perCB { mm.encode(commandBuffer: cb, leftMatrix: mx, rightMatrix: mw, resultMatrix: my) } } / Double(perCB))
            }
            let t = median(ts)
            rows.append(["shape": sname, "N": N, "K": K, "M": M, "bucket_s": bucket, "variant": "mps_dense_fp16",
                         "us": t * 1e6, "tflops": flops / t / 1e12])
            print("bench \(sname) M=\(M) done")
        }
    }
    writeJSON(["mode": "bench", "device": deviceInfo(), "reps": reps, "dispatches_per_command_buffer": perCB,
               "rows": rows], args[3])
}

// --- encoder (240 projections) --------------------------------------------------------------------------------
func encoderSeq() {
    let benchData = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[4]))) as! [String: Any]
    let rows = benchData["rows"] as! [[String: Any]]
    func best(_ N: Int, _ K: Int, _ M: Int, _ prefix: String) -> Variant {
        let c = rows.filter { ($0["N"] as! Int) == N && ($0["K"] as! Int) == K && ($0["M"] as! Int) == M &&
            ($0["variant"] as! String).hasPrefix(prefix) && !($0["variant"] as! String).hasPrefix("mps") }
        let b = c.min { ($0["us"] as! Double) < ($1["us"] as! Double) }!
        return Variant(b["variant"] as! String)
    }
    var r = Rng(s: 99)
    // distinct weights per layer (24 x 10 GEMMs, 75.5 MB of codes)
    var layerW: [[MTLBuffer]] = []
    var layerS: [[MTLBuffer]] = []
    for _ in 0..<24 {
        var wl: [MTLBuffer] = [], sl: [MTLBuffer] = []
        for (N, K) in LAYER {
            var ws = [UInt32](repeating: 0, count: N * K / 32)
            for i in 0..<ws.count { ws[i] = UInt32(truncatingIfNeeded: r.next() >> 11) }
            wl.append(buffer(ws))
            sl.append(buffer((0..<N).map { _ in Float16(0.01 + 0.04 * r.unit()) }))
        }
        layerW.append(wl); layerS.append(sl)
    }
    var out: [String: Any] = ["mode": "encoder", "device": deviceInfo()]
    var buckets: [String: Any] = [:]
    for (bucket, M) in BUCKET_M {
        let x1 = buffer((0..<(M * 4096)).map { _ in Float16(r.normal()) })
        let y = device.makeBuffer(length: M * 4096 * 2, options: .storageModeShared)!
        var entry: [String: Any] = ["M": M]
        for prefix in ["mma", "simt", "lut", ""] {
            let choice = LAYER.map { best($0.0, $0.1, M, prefix) }
            var ts: [Double] = []
            for rep in 0..<6 {
                let t = gpuTime { cb in
                    let e = cb.makeComputeCommandEncoder()!
                    for l in 0..<24 {
                        for (i, (N, K)) in LAYER.enumerated() {
                            encode(e, choice[i], "h", Problem(M: M, N: N, K: K, x: x1, w: layerW[l][i], s: layerS[l][i], y: y))
                        }
                    }
                    e.endEncoding()
                }
                if rep > 0 { ts.append(t) }
            }
            entry[prefix.isEmpty ? "best_any" : "best_" + prefix] = ["ms": median(ts) * 1e3, "variants": choice.map { $0.name }]
        }
        buckets[String(bucket)] = entry
        print("encoder M=\(M) done")
    }
    out["buckets"] = buckets
    writeJSON(out, args[3])
}

switch mode {
case "check": check()
case "peak": peak()
case "bench": bench()
case "encoder": encoderSeq()
default: print("unknown mode"); exit(2)
}
