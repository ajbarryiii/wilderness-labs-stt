fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let v = std::process::Command::new(rustc).arg("-V").output().unwrap();
    println!("cargo:rustc-env=GOLDEN_RUSTC={}", String::from_utf8(v.stdout).unwrap().trim());
}
