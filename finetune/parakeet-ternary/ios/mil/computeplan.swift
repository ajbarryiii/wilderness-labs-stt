// MLComputePlan per-op device usage of one function of a compiled model (DESIGN.md gate 6 record).
// coremltools' Python MLComputePlan cannot select a function of a multifunction model; this sets
// MLModelConfiguration.functionName. Standalone (not part of bench/), built by mil/build.py:
//   swiftc -O -parse-as-library -o <artifacts>/bin/computeplan mil/computeplan.swift
//   computeplan --model X.mlmodelc --units cpuAndNeuralEngine [--function b2]   -> JSON on stdout
// The plan alone does not establish placement (Instruments traces do).
import CoreML
import Foundation

@main
struct ComputePlanTool {
    static func label(_ d: MLComputeDevice) -> String {
        switch d {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case .neuralEngine: return "ane"
        @unknown default: return "other"
        }
    }

    static func main() async {
        var args = [String: String]()
        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let k = it.next() {
            guard k.hasPrefix("--"), let v = it.next() else { FileHandle.standardError.write("bad arguments\n".data(using: .utf8)!); exit(2) }
            args[String(k.dropFirst(2))] = v
        }
        guard let model = args["model"] else { FileHandle.standardError.write("--model required\n".data(using: .utf8)!); exit(2) }
        let config = MLModelConfiguration()
        switch args["units"] ?? "cpuAndNeuralEngine" {
        case "cpuOnly": config.computeUnits = .cpuOnly
        case "cpuAndGPU": config.computeUnits = .cpuAndGPU
        case "all": config.computeUnits = .all
        default: config.computeUnits = .cpuAndNeuralEngine
        }
        if let fn = args["function"] { config.functionName = fn }
        do {
            let t0 = Date()
            let plan = try await MLComputePlan.load(contentsOf: URL(fileURLWithPath: model), configuration: config)
            let loadS = Date().timeIntervalSince(t0)
            var byDevice = [String: Int](), cost = [String: Double](), opsByDevice = [String: [String: Int]]()
            var noUsage = [String: Int](), cpuOps = [[String: Any]]()
            func walk(_ block: MLModelStructure.Program.Block) {
                for op in block.operations {
                    let c = plan.estimatedCost(of: op)?.weight ?? 0
                    if let u = plan.deviceUsage(for: op) {
                        let d = label(u.preferred)
                        byDevice[d, default: 0] += 1
                        cost[d, default: 0] += c
                        opsByDevice[d, default: [:]][op.operatorName, default: 0] += 1
                        if d == "cpu" && cpuOps.count < 50 {
                            cpuOps.append(["op": op.operatorName, "outputs": op.outputs.prefix(2).map { $0.name }, "cost": c])
                        }
                    } else {
                        noUsage[op.operatorName, default: 0] += 1
                    }
                    for b in op.blocks { walk(b) }
                }
            }
            var functionName = args["function"] ?? "main"
            if case .program(let program) = plan.modelStructure {
                if program.functions[functionName] == nil, let only = program.functions.keys.sorted().first,
                   args["function"] == nil { functionName = only }
                guard let f = program.functions[functionName] else {
                    FileHandle.standardError.write("no function \(functionName)\n".data(using: .utf8)!); exit(1)
                }
                walk(f.block)
            }
            let total = cost.values.reduce(0, +)
            let out: [String: Any] = [
                "function": functionName, "compute_units": args["units"] ?? "cpuAndNeuralEngine", "plan_load_s": loadS,
                "ops_with_usage_by_preferred_device": byDevice,
                "estimated_cost_share_by_preferred_device": cost.mapValues { total > 0 ? $0 / total : 0 },
                "operators_by_preferred_device": opsByDevice, "ops_without_usage": noUsage, "cpu_ops": cpuOps,
            ]
            let data = try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write("\n".data(using: .utf8)!)
        } catch {
            FileHandle.standardError.write("compute plan failed: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }
}
