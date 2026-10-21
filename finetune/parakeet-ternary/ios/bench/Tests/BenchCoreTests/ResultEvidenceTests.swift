import XCTest
@testable import BenchCore

/// Safety fixtures only: no Core ML models, inference or timing.
final class ResultEvidenceTests: XCTestCase {
    func fixture(_ body: (URL, URL, String) throws -> Void) throws {
        let area = URL(fileURLWithPath: "/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios")
        let scratch = area.appendingPathComponent("scratch/evidence-swift-" + UUID().uuidString)
        let ios = scratch.appendingPathComponent("ios")
        let root = scratch.appendingPathComponent("archive")
        try FileManager.default.createDirectory(at: ios.appendingPathComponent("results"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("fixture/gates"), withIntermediateDirectories: true)
        let config = ios.appendingPathComponent("mil/local.json")
        try FileManager.default.createDirectory(at: config.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["IOS_RESULTS_ROOT": root.path]).write(to: config)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let gate = Data("{\"pass\":true}".utf8)
        let gateName = "results/gates/test.json"
        let recordName = "results/eligibility/mp2-C4-multi-ane.json"
        let record: [String: Any] = ["design_revision": 8, "model": "mp2", "arm": "C4", "variant": "multi",
                                    "backend": "ane", "compute_units": "cpuAndNeuralEngine", "timing_allowed": true,
                                    "inputs": [gateName: sha256Hex(gate)]]
        let data = try JSONSerialization.data(withJSONObject: record)
        try gate.write(to: root.appendingPathComponent("fixture/gates/test.json"))
        try data.write(to: root.appendingPathComponent("fixture/record.json"))
        let entries: [String: [String: Any]] = [
            gateName: ["sha256": sha256Hex(gate), "bytes": gate.count, "archive": "fixture/gates/test.json"],
            recordName: ["sha256": sha256Hex(data), "bytes": data.count, "archive": "fixture/record.json"],
        ]
        try JSONSerialization.data(withJSONObject: ["version": 1, "records": entries]).write(to: ios.appendingPathComponent("results/INDEX.json"))
        try body(ios, root, recordName)
    }

    func check(_ ios: URL) throws {
        let rec = try Eligibility.check(iosDir: ios, model: "mp2", arm: "C4", variant: .multifunction,
                                       computeUnits: "cpuAndNeuralEngine",
                                       encoderPath: ios.appendingPathComponent("models/mp2/C4/multi.mlmodelc"))
        XCTAssertEqual(rec["timing_allowed"] as? Bool, true)
    }

    func testArchivedEligibilityAndMissingRecord() throws {
        try fixture { ios, root, _ in
            try check(ios)
            try FileManager.default.removeItem(at: root.appendingPathComponent("fixture/record.json"))
            XCTAssertThrowsError(try check(ios))
        }
    }

    func testTamperedRecordAndInput() throws {
        try fixture { ios, root, name in
            let original = try ResultEvidence.read(iosDir: ios, relative: name)
            try Data("tampered".utf8).write(to: root.appendingPathComponent("fixture/record.json"))
            XCTAssertThrowsError(try check(ios))
            try original.write(to: root.appendingPathComponent("fixture/record.json"))
            try Data("tampered".utf8).write(to: root.appendingPathComponent("fixture/gates/test.json"))
            XCTAssertThrowsError(try check(ios))
        }
    }

    func testDisplayAndQuarantineCannotAuthorize() throws {
        try fixture { ios, root, name in
            let display = ios.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: display.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("{\"timing_allowed\":true}".utf8).write(to: display)
            try FileManager.default.removeItem(at: root.appendingPathComponent("fixture/record.json"))
            XCTAssertThrowsError(try check(ios))
            XCTAssertThrowsError(try ResultEvidence.read(iosDir: ios, relative: "results/eligibility/pipelines/quarantine/fake.json"))
            XCTAssertThrowsError(try ResultEvidence.read(iosDir: ios, relative: "results/../outside.json"))
        }
    }
}
