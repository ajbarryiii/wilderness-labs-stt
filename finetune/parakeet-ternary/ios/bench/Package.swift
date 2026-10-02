// swift-tools-version: 6.0
// ParakeetBench: the Parakeet iOS benchmark harness (DESIGN.md "Harness"). BenchCore is shared by the macOS
// command-line target and (later) the iOS app; no third-party dependencies.
import PackageDescription

let package = Package(
    name: "ParakeetBench",
    platforms: [.macOS(.v15), .iOS("26.0")],
    products: [
        .library(name: "BenchCore", targets: ["BenchCore"]),
        .executable(name: "parakeet-bench", targets: ["parakeet-bench"]),
    ],
    targets: [
        .target(name: "BenchCore", path: "Sources/BenchCore"),
        .executableTarget(name: "parakeet-bench", dependencies: ["BenchCore"], path: "Sources/parakeet-bench"),
    ],
    swiftLanguageModes: [.v5]
)
