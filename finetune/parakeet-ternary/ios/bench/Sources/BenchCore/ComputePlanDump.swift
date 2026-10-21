import CoreML
import Foundation

/// Per-operation MLComputePlan device usage (macOS 14.4+ / iOS 17.4+) of a compiled model, as JSON-ready
/// dictionaries. DESIGN.md gate 6: the plan alone does not establish placement (Instruments traces do).
public enum ComputePlanDump {
    static func label(_ device: MLComputeDevice) -> String {
        switch device {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case .neuralEngine: return "ane"
        @unknown default: return "other"
        }
    }

    public static func dump(model url: URL, computeUnits: MLComputeUnits) async throws -> [String: Any] {
        let config = C0Models.configuration(computeUnits)
        let t0 = Clock.now()
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
        let loadMs = Clock.ms(t0, Clock.now())
        var ops: [[String: Any]] = []
        var byDevice: [String: Int] = [:], costByDevice: [String: Double] = [:]
        var opsByDevice: [String: [String: Int]] = [:]
        var noUsage: [String: Int] = [:]

        func walk(_ block: MLModelStructure.Program.Block, function: String, depth: Int) {
            for op in block.operations {
                var entry: [String: Any] = ["function": function, "op": op.operatorName,
                                            "outputs": op.outputs.map { $0.name }]
                if depth > 0 { entry["depth"] = depth }
                let cost = plan.estimatedCost(of: op)?.weight
                if let cost { entry["cost"] = cost }
                if let usage = plan.deviceUsage(for: op) {
                    let preferred = label(usage.preferred)
                    entry["preferred"] = preferred
                    entry["supported"] = usage.supported.map(label)
                    byDevice[preferred, default: 0] += 1
                    costByDevice[preferred, default: 0] += cost ?? 0
                    opsByDevice[preferred, default: [:]][op.operatorName, default: 0] += 1
                } else {
                    noUsage[op.operatorName, default: 0] += 1
                }
                ops.append(entry)
                for inner in op.blocks { walk(inner, function: function, depth: depth + 1) }
            }
        }

        var structure = "unsupported"
        switch plan.modelStructure {
        case .program(let program):
            structure = "program"
            for name in program.functions.keys.sorted() { walk(program.functions[name]!.block, function: name, depth: 0) }
        case .neuralNetwork(let network):
            structure = "neuralNetwork"
            for layer in network.layers {
                var entry: [String: Any] = ["op": layer.type, "name": layer.name]
                if let usage = plan.deviceUsage(for: layer) {
                    let preferred = label(usage.preferred)
                    entry["preferred"] = preferred
                    entry["supported"] = usage.supported.map(label)
                    byDevice[preferred, default: 0] += 1
                    opsByDevice[preferred, default: [:]][layer.type, default: 0] += 1
                }
                ops.append(entry)
            }
        case .pipeline: structure = "pipeline"
        case .unsupported: structure = "unsupported"
        @unknown default: structure = "unknown"
        }
        let totalCost = costByDevice.values.reduce(0, +)
        return [
            "model": url.lastPathComponent, "compute_units": ComputeUnitsName.name(computeUnits), "structure": structure,
            "plan_load_ms": loadMs,
            "summary": ["ops_with_usage_by_preferred_device": byDevice,
                        "estimated_cost_share_by_preferred_device": costByDevice.mapValues { totalCost > 0 ? $0 / totalCost : 0 },
                        "operators_by_preferred_device": opsByDevice,
                        "ops_without_usage (constants etc.)": noUsage,
                        "ops_total": ops.count] as [String: Any],
            "ops": ops,
        ]
    }
}
