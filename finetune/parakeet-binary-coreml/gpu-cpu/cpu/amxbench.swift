// amxbench: CPU-side speed ceiling for the MB1 encoder projections through Accelerate (cblas_sgemm, which runs on
// the M1 Pro's AMX matrix units), dense FP32 weights (codes x 1). This is optimistic for a 1-bit deployment: it
// ignores the cost of expanding 1-bit codes to FP32 (or keeping 2.4 GB of FP32 weights) and reuses one layer's
// weights for all 24 layers (memory). Synthetic data only.
//   amxbench OUT.json [reps]
import Accelerate
import Foundation

let LAYER: [(Int, Int)] = [(4096, 1024), (1024, 4096), (1024, 1024), (1024, 1024), (1024, 1024), (1024, 1024),
                           (2048, 1024), (1024, 1024), (4096, 1024), (1024, 4096)]
let BUCKET_M: [(Int, Int)] = [(4, 51), (8, 101), (15, 188), (30, 376)]
let reps = CommandLine.arguments.count > 2 ? Int(CommandLine.arguments[2])! : 5

var seed: UInt64 = 12345
func rnd() -> Float { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return Float(seed >> 40) / Float(1 << 23) - 1 }

let weights: [[Float]] = LAYER.map { (n, k) in (0..<(n * k)).map { _ in rnd() < 0 ? -1 : 1 } }
let xin = (0..<(376 * 4096)).map { _ in rnd() }
var yout = [Float](repeating: 0, count: 376 * 4096)

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) * 1e-9 }

var buckets: [String: Any] = [:]
for (b, M) in BUCKET_M {
    var perShape: [String: Double] = [:]
    func gemm(_ i: Int) {
        let (N, K) = LAYER[i]
        xin.withUnsafeBufferPointer { xp in
            weights[i].withUnsafeBufferPointer { wp in
                yout.withUnsafeMutableBufferPointer { yp in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(M), Int32(N), Int32(K), 1,
                                xp.baseAddress!, Int32(K), wp.baseAddress!, Int32(K), 0, yp.baseAddress!, Int32(N))
                }
            }
        }
    }
    for i in [0, 1, 2, 6] {
        for _ in 0..<3 { gemm(i) }
        var ts: [Double] = []
        for _ in 0..<reps { let t0 = now(); for _ in 0..<10 { gemm(i) }; ts.append((now() - t0) / 10) }
        ts.sort()
        perShape["\(LAYER[i].0)x\(LAYER[i].1)_us"] = ts[ts.count / 2] * 1e6
    }
    for i in 0..<10 { gemm(i) }
    var ts: [Double] = []
    for _ in 0..<reps {
        let t0 = now()
        for _ in 0..<24 { for i in 0..<10 { gemm(i) } }
        ts.append(now() - t0)
    }
    ts.sort()
    let ms = ts[ts.count / 2] * 1e3
    buckets[String(b)] = ["M": M, "projections_ms": ms, "tflops": 2 * 579.0e6 * Double(M) / (ms * 1e-3) / 1e12,
                          "per_shape": perShape]
    print("bucket \(b) s: \(ms) ms")
}
let out: [String: Any] = ["mode": "amx_sgemm", "reps": reps, "note": "cblas_sgemm FP32 dense, one layer's weights reused x 24",
                          "buckets": buckets]
let d = try! JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys])
try! d.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
print(String(data: d, encoding: .utf8)!)
