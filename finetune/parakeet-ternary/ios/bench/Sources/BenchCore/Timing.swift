import Darwin
import Foundation
import os

/// mach_absolute_time based clock (DESIGN.md "Measurements": per-stage times use os_signpost and mach_absolute_time).
public enum Clock {
    private static let timebase: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    @inline(__always) public static func now() -> UInt64 { mach_absolute_time() }

    /// Milliseconds between two mach_absolute_time readings.
    @inline(__always) public static func ms(_ start: UInt64, _ end: UInt64) -> Double {
        Double(end &- start) * Double(timebase.numer) / Double(timebase.denom) / 1e6
    }
}

/// os_signpost intervals visible in Instruments (Points of Interest and the "ParakeetBench" category).
public enum Signposts {
    public static let subsystem = "com.wildernesslabs.parakeet-bench"
    public static let poi = OSSignposter(subsystem: subsystem, category: .pointsOfInterest)
    public static let stages = OSSignposter(subsystem: subsystem, category: "stages")
}

/// Accumulates the time spent in a group of repeated calls (e.g. all decoder model calls of one utterance).
public struct Accumulator {
    public private(set) var totalMs: Double = 0
    public private(set) var count: Int = 0
    public init() {}

    @inline(__always) public mutating func measure<T>(_ body: () throws -> T) rethrows -> T {
        let t0 = Clock.now()
        let value = try body()
        totalMs += Clock.ms(t0, Clock.now())
        count += 1
        return value
    }
}

/// Current and peak phys_footprint of this process (bytes), from task_info(TASK_VM_INFO).
public func physFootprint() -> (current: UInt64, peak: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return (0, 0) }
    return (info.phys_footprint, info.ledger_phys_footprint_peak > 0 ? UInt64(info.ledger_phys_footprint_peak) : 0)
}
