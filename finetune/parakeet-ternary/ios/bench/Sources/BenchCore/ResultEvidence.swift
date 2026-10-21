import Foundation

/// Full gate records are immutable external artifacts. INDEX.json commits the current
/// SHA-256, byte count and relative archive path. Display summaries never authorize timing.
public enum ResultEvidence {
    public static func root(iosDir: URL) throws -> URL {
        var setting = ProcessInfo.processInfo.environment["IOS_RESULTS_ROOT"]
        let config = iosDir.appendingPathComponent("mil/local.json")
        if setting == nil, let data = try? Data(contentsOf: config),
           let doc = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
            setting = doc["IOS_RESULTS_ROOT"] as? String
        }
        #if os(macOS)
        let area = URL(fileURLWithPath: "/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios")
        #else
        let area = URL(fileURLWithPath: "/mnt/hd/wilderness-labs-stt/parakeet-ios")
        #endif
        let root = (setting.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
                    ?? area.appendingPathComponent("results-archive")).standardizedFileURL.resolvingSymlinksInPath()
        let allowed = area.standardizedFileURL.resolvingSymlinksInPath().path + "/"
        guard root.path.hasPrefix(allowed),
              !root.path.hasPrefix(iosDir.standardizedFileURL.resolvingSymlinksInPath().path + "/") else {
            throw BenchError.invalid("results archive must be in the artifact area, outside Git")
        }
        return root
    }

    public static func read(iosDir: URL, relative: String) throws -> Data {
        let parts = relative.split(separator: "/", omittingEmptySubsequences: false)
        guard !relative.hasPrefix("/"), !parts.contains(".."), !parts.contains("quarantine") else {
            throw BenchError.invalid("unsafe or quarantined result path: \(relative)")
        }
        if !relative.hasPrefix("results/") {
            return try Data(contentsOf: iosDir.appendingPathComponent(relative))
        }
        let indexURL = iosDir.appendingPathComponent("results/INDEX.json")
        guard let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any],
              index["version"] as? Int == 1, let records = index["records"] as? [String: [String: Any]],
              let entry = records[relative], let name = entry["archive"] as? String,
              let digest = entry["sha256"] as? String, let bytes = entry["bytes"] as? Int,
              !name.hasPrefix("/"), !name.split(separator: "/").contains("..") else {
            throw BenchError.invalid("missing or malformed indexed record: \(relative)")
        }
        let root = try root(iosDir: iosDir)
        let path = root.appendingPathComponent(name).standardizedFileURL.resolvingSymlinksInPath()
        guard path.path.hasPrefix(root.path + "/") else {
            throw BenchError.invalid("archive path escapes root")
        }
        let data = try Data(contentsOf: path)
        guard data.count == bytes, sha256Hex(data) == digest else {
            throw BenchError.invalid("result SHA-256/size mismatch: \(relative)")
        }
        return data
    }
}
