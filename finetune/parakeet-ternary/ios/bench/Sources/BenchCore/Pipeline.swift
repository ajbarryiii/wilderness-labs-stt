import CryptoKit
import Foundation

/// SHA-256 identities of loaded components (WP7, review finding 2).
public enum ComponentHash {
    /// Streaming SHA-256 of one file. Each chunk is read inside its own autorelease pool: FileHandle returns
    /// bridged, autoreleased data, and the CLI's pool never drains before timing, so without it hashing a model
    /// package left its whole size in phys_footprint (WP7 sweep: 390 MB before any model was loaded).
    public static func file(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while try autoreleasepool(invoking: { () throws -> Bool in
            guard let chunk = try handle.read(upToCount: 1 << 22), !chunk.isEmpty else { return false }
            hasher.update(data: chunk)
            return true
        }) {}
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// SHA-256 of a directory (e.g. a compiled .mlmodelc): over the lines "<relative path>\t<file SHA-256>\n" of
    /// every regular file, sorted by relative path. Symbolic links are refused.
    public static func directory(_ url: URL) throws -> String {
        let root = url.standardizedFileURL.resolvingSymlinksInPath()
        guard let e = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]) else {
            throw BenchError.invalid("cannot enumerate \(root.path)")
        }
        var lines: [String] = []
        for case let f as URL in e {
            let v = try f.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            if v.isSymbolicLink == true { throw BenchError.invalid("symbolic link in \(root.path): \(f.path)") }
            guard v.isRegularFile == true else { continue }
            let rel = String(f.standardizedFileURL.resolvingSymlinksInPath().path.dropFirst(root.path.count + 1))
            lines.append("\(rel)\t\(try file(f))\n")
        }
        guard !lines.isEmpty else { throw BenchError.invalid("\(root.path) has no files") }
        return sha256Hex(lines.sorted().joined().data(using: .utf8)!)
    }

    /// A native.py blob's manifest (<stem>.json), after streaming the blob file it names against the manifest's
    /// SHA-256 (no tensors are materialized, so nothing stays in the process's footprint).
    public static func blobManifest(_ dir: URL, stem: String) throws -> [String: Any] {
        guard let m = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("\(stem).json"))) as? [String: Any],
              let file = m["file"] as? String, let digest = m["sha256"] as? String else {
            throw BenchError.invalid("\(dir.path)/\(stem).json: no file / sha256")
        }
        guard try Self.file(dir.appendingPathComponent(file)) == digest else {
            throw BenchError.invalid("\(dir.path)/\(file): SHA-256 differs from its manifest")
        }
        return m
    }
}

/// The executing implementation (review WP7 r1 finding 2): the SHA-256 of this process's executable, so a record made
/// by one build of parakeet-bench does not admit another build's loop code or loaders.
public enum BuildIdentity {
    public static let executableSHA256: String = {
        guard let url = Bundle.main.executableURL, let digest = try? ComponentHash.file(url) else { return "unknown" }
        return digest
    }()
    /// Settings every pipeline model is loaded with (C0Models.configuration) and the decode-loop constants.
    public static func configuration(computeUnits: String) -> [String: Any] {
        let loop = LabelLoop()
        return ["compute_units": computeUnits, "allow_low_precision_accumulation_on_gpu": true,
                "label_loop": ["blank": loop.blank, "durations": loop.durations, "max_symbols": loop.maxSymbols]]
    }
}

/// C0's identity (review WP7 r1 finding 3): every file of the pinned published export (ios/c0.json: repo, revision,
/// per-file size and SHA-256) must be present and equal, and each model directory may hold no other file.
public enum C0Identity {
    public static func verify(directory: URL, pinned: URL) throws -> [String: Any] {
        let raw = try Data(contentsOf: pinned)
        guard let doc = try JSONSerialization.jsonObject(with: raw) as? [String: Any],
              let files = doc["files"] as? [[String: Any]], !files.isEmpty else {
            throw BenchError.invalid("\(pinned.path): no pinned files")
        }
        var problems: [String] = [], expected = Set<String>()
        for f in files {
            guard let path = f["path"] as? String, let sha = f["sha256"] as? String, let bytes = f["bytes"] as? Int else {
                throw BenchError.invalid("\(pinned.path): malformed file entry")
            }
            expected.insert(path)
            let url = directory.appendingPathComponent(path)
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? -1
            if size != bytes { problems.append("\(path): \(size) bytes, pinned \(bytes)"); continue }
            if (try? ComponentHash.file(url)) != sha { problems.append("\(path): SHA-256 differs from c0.json") }
        }
        for dir in Set(expected.compactMap { $0.contains("/") ? String($0.split(separator: "/")[0]) : nil }) {
            let root = directory.appendingPathComponent(dir).standardizedFileURL.resolvingSymlinksInPath()
            guard let e = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey]) else { continue }
            for case let u as URL in e where (try? u.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
                let rel = dir + "/" + String(u.standardizedFileURL.resolvingSymlinksInPath().path.dropFirst(root.path.count + 1))
                if !expected.contains(rel) { problems.append("\(rel): not in c0.json") }
            }
        }
        guard problems.isEmpty else {
            throw BenchError.invalid("refusing: \(directory.path) is not the pinned C0 export: \(problems.prefix(5))")
        }
        return ["verified": true, "repo": doc["repo"] ?? "", "revision": doc["revision"] ?? "", "files": files.count,
                "c0_json_sha256": sha256Hex(raw)]
    }
}

/// One deployed pipeline: front end + encoder arm (model, arm, variant, backend) + decode loop (+ its components).
public struct PipelineSpec {
    public let model: String
    public let arm: String
    public let variant: LengthVariant
    public let computeUnits: String
    public let frontEnd: String          // "vdsp" or "c0pre"
    public let decode: String            // "f0", "f1", "f2"
    public let encoderPath: URL
    public let frontendConstants: URL?   // vdsp
    public let c0Dir: URL?               // c0pre (C0's Preprocessor)
    public let nativeWeights: URL?       // f2
    public let decoderModels: URL?       // f0 / f1

    public init(model: String, arm: String, variant: LengthVariant, computeUnits: String, frontEnd: String, decode: String,
                encoderPath: URL, frontendConstants: URL?, c0Dir: URL?, nativeWeights: URL?, decoderModels: URL?) {
        self.model = model; self.arm = arm; self.variant = variant; self.computeUnits = computeUnits
        self.frontEnd = frontEnd; self.decode = decode; self.encoderPath = encoderPath
        self.frontendConstants = frontendConstants; self.c0Dir = c0Dir; self.nativeWeights = nativeWeights
        self.decoderModels = decoderModels
    }

    public var backend: String { Eligibility.backend(computeUnits) ?? computeUnits }
    public var variantName: String { Eligibility.variants[variant] ?? variant.rawValue }
    public var recordName: String { "\(model)-\(arm)-\(variantName)-\(backend)-\(frontEnd)-\(decode).json" }

    /// Identities and configuration of every component this pipeline loads.
    public func components() throws -> [String: Any] {
        var c: [String: Any] = [
            "compute_units": computeUnits,
            "configuration": BuildIdentity.configuration(computeUnits: computeUnits),
            "executable_sha256": BuildIdentity.executableSHA256,
            "encoder": ["model": model, "arm": arm, "variant": variantName,
                        "package": "\(model)/\(arm)/\(variantName).mlmodelc", "sha256": try ComponentHash.directory(encoderPath)],
        ]
        switch frontEnd {
        case "vdsp":
            guard let dir = frontendConstants else { throw BenchError.invalid("vdsp needs front-end constants") }
            let m = try ComponentHash.blobManifest(dir, stem: "frontend")  // verifies the blob against its manifest
            // the manifest is bound too: it maps names to offsets and shapes (review WP7 r1 finding 2)
            c["front_end"] = ["kind": "vdsp", "constants_sha256": m["sha256"] ?? "", "provenance": m["provenance"] ?? [:],
                              "manifest_sha256": try ComponentHash.file(dir.appendingPathComponent("frontend.json"))]
        case "c0pre":
            guard let dir = c0Dir else { throw BenchError.invalid("c0pre needs the C0 directory") }
            c["front_end"] = ["kind": "c0pre", "preprocessor_sha256": try ComponentHash.directory(dir.appendingPathComponent("Preprocessor.mlmodelc"))]
        default: throw BenchError.invalid("unknown front end \(frontEnd)")
        }
        switch decode {
        case "f2":
            guard let dir = nativeWeights else { throw BenchError.invalid("f2 needs native weights") }
            let m = try ComponentHash.blobManifest(dir, stem: "decoder_joint")
            c["decode"] = ["kind": "f2", "precision": "fp32", "native_weights_sha256": m["sha256"] ?? "",
                           "provenance": m["provenance"] ?? [:],
                           "manifest_sha256": try ComponentHash.file(dir.appendingPathComponent("decoder_joint.json"))]
        case "f0", "f1":
            guard let dir = decoderModels else { throw BenchError.invalid("\(decode) needs decoder models") }
            let names = decode == "f0" ? ["Decoder", "JointDecision"] : ["DecoderJoint"]
            var models: [String: String] = [:]
            for n in names { models[n] = try ComponentHash.directory(dir.appendingPathComponent("\(n).mlmodelc")) }
            var precision = "unknown"
            let manifestURL = dir.appendingPathComponent("manifest.json")
            if let data = try? Data(contentsOf: manifestURL),
               let m = try JSONSerialization.jsonObject(with: data) as? [String: Any], let p = m["precision"] as? String {
                precision = p
            }
            c["decode"] = ["kind": decode, "precision": precision, "models_sha256": models,
                           "manifest_sha256": (try? ComponentHash.file(manifestURL)) ?? "missing"]
        default: throw BenchError.invalid("unknown decode \(decode)")
        }
        return c
    }
}

/// Timing gate for deployed pipelines (WP7): the exact combination needs a passing pipeline record
/// results/eligibility/pipelines/<model>-<arm>-<variant>-<backend>-<frontend>-<decode>.json of the required design
/// revision whose components (SHA-256 + configuration) equal what this process is about to load, whose input files are
/// unchanged, and whose encoder arm still has its passing WP3 record.
public enum PipelineEligibility {
    /// Pipeline records follow DESIGN.md revision 10 (gate 4: "Deployed-pipeline references", "Decisions-only
    /// decoder models"); the encoder arms' WP3 records stay at Eligibility.requiredRevision (8).
    public static let requiredRevision = 10

    public static func check(iosDir: URL, spec: PipelineSpec) throws -> [String: Any] {
        guard let data = try? ResultEvidence.read(iosDir: iosDir, relative: "results/eligibility/pipelines/" + spec.recordName),
              let rec = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw BenchError.invalid("refusing to time: no pipeline record \(spec.recordName)")
        }
        guard (rec["design_revision"] as? Int) == requiredRevision else {
            throw BenchError.invalid("refusing to time: \(spec.recordName) is not a revision-\(requiredRevision) record")
        }
        guard rec["timing_allowed"] as? Bool == true else {
            let reasons = (rec["reasons"] as? [String])?.joined(separator: "; ") ?? "?"
            throw BenchError.invalid("refusing to time: pipeline \(spec.recordName) not eligible: \(reasons)")
        }
        let encoderRecord = try Eligibility.check(iosDir: iosDir, model: spec.model, arm: spec.arm, variant: spec.variant,
                                                  computeUnits: spec.computeUnits, encoderPath: spec.encoderPath)
        let want = try spec.components()
        guard let have = rec["components"] as? [String: Any],
              canonical(have) == canonical(want) else {
            throw BenchError.invalid("refusing to time: loaded components differ from \(spec.recordName) "
                + "(record \(canonical(rec["components"] ?? [:]).prefix(300)) vs loaded \(canonical(want).prefix(300)))")
        }
        var stale: [String] = []
        for (file, digest) in (rec["inputs"] as? [String: String]) ?? [:] {
            let d = try? ResultEvidence.read(iosDir: iosDir, relative: file)
            if d == nil || sha256Hex(d!) != digest { stale.append(file) }
        }
        guard stale.isEmpty else { throw BenchError.invalid("refusing to time: \(spec.recordName) inputs changed: \(stale.prefix(5))") }
        return ["pipeline_record": spec.recordName, "encoder_record": encoderRecord, "components": want, "built": rec["built"] ?? NSNull()]
    }

    /// Sorted-key JSON text of the identity-bearing fields (provenance blocks excluded).
    static func canonical(_ x: Any) -> String {
        func strip(_ v: Any) -> Any {
            if let d = v as? [String: Any] { return d.filter { $0.key != "provenance" }.mapValues(strip) }
            if let a = v as? [Any] { return a.map(strip) }
            return v
        }
        guard let data = try? JSONSerialization.data(withJSONObject: strip(x), options: [.sortedKeys]) else { return "" }
        return String(data: data, encoding: .utf8) ?? ""
    }
}
