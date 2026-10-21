import Accelerate
import CoreML
import Darwin

/// Replica of FluidAudio 0.7.8 ANEMemoryUtils.createAlignedArray / calculateOptimalStrides
/// (Sources/FluidAudio/Shared/ANEMemoryUtils.swift at tag v0.7.8): 64-byte aligned allocation, innermost
/// dimension padded to a multiple of 16 elements when it is not one (so [1, 1024, 1] gets strides
/// [16384, 16, 1]). C0's decode loop passes such arrays to Core ML, so the harness does the same.
public enum AlignedArray {
    public static let alignment = 64
    public static let tile = 16

    public static func strides(for shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 0, count: shape.count)
        var current = 1
        for i in stride(from: shape.count - 1, through: 0, by: -1) {
            strides[i] = current
            if i == shape.count - 1 && shape[i] % tile != 0 {
                current *= (shape[i] + tile - 1) / tile * tile
            } else {
                current *= shape[i]
            }
        }
        return strides
    }

    public static func make(shape: [Int], dataType: MLMultiArrayDataType, zero: Bool = false) throws -> MLMultiArray {
        let strides = strides(for: shape)
        let elementSize: Int
        switch dataType {
        case .float16: elementSize = 2
        case .float32, .int32: elementSize = 4
        case .float64: elementSize = 8
        default: elementSize = 4
        }
        let bytes = max(alignment, ((strides[0] * shape[0] * elementSize) + alignment - 1) / alignment * alignment)
        var pointer: UnsafeMutableRawPointer?
        guard posix_memalign(&pointer, alignment, bytes) == 0, let pointer else {
            throw BenchError.invalid("posix_memalign failed")
        }
        if zero { memset(pointer, 0, bytes) }
        return try MLMultiArray(dataPointer: pointer, shape: shape.map { NSNumber(value: $0) }, dataType: dataType,
                                strides: strides.map { NSNumber(value: $0) }, deallocator: { Darwin.free($0) })
    }
}

extension MLMultiArray {
    /// vDSP fill of a float32 array (FluidAudio MLMultiArray.resetData).
    func fill(_ value: Float) {
        var v = value
        dataPointer.withMemoryRebound(to: Float.self, capacity: count) { vDSP_vfill(&v, $0, 1, vDSP_Length(count)) }
    }

    var intShape: [Int] { shape.map { $0.intValue } }
    var intStrides: [Int] { strides.map { $0.intValue } }
}
