//! Toolchain and codegen pin (plans/cpu-benchmark-mac.md "Code"): on aarch64-apple-darwin the build
//! fails unless rustc is exactly PINNED_RUSTC and the encoded rustflags carry -C target-cpu=apple-m1.
//! Other targets (NixOS x86_64, used only for portable unit tests and golden-vector tooling) build
//! with whatever rustc they have; the version is recorded in CPU1BIT_RUSTC either way.
use std::process::Command;

const PINNED_RUSTC: &str = "rustc 1.99.1 (58cf8f9c3 2026-10-13)";

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let out = Command::new(&rustc).arg("-V").output().expect("run rustc -V");
    let version = String::from_utf8(out.stdout).unwrap().trim().to_string();
    let target = std::env::var("TARGET").unwrap();
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let flags: Vec<&str> = flags.split('\x1f').collect();
    if target == "aarch64-apple-darwin" {
        assert_eq!(version, PINNED_RUSTC, "cpu1bit: Mac builds must use the pinned stable toolchain");
        let cpu = flags.windows(2).any(|w| w == ["-C", "target-cpu=apple-m1"])
            || flags.contains(&"-Ctarget-cpu=apple-m1");
        assert!(cpu, "cpu1bit: -C target-cpu=apple-m1 missing from rustflags {flags:?}");
    }
    println!("cargo:rustc-env=CPU1BIT_RUSTC={version}");
    println!("cargo:rustc-env=CPU1BIT_RUSTFLAGS={}", flags.join(" "));
    println!("cargo:rustc-env=CPU1BIT_TARGET={target}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
}
