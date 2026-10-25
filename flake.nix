{
  description = "wilderness-labs-stt development shells";

  # bend2 is only in nixpkgs-unstable (not in the 26.05 system channel); flake.lock pins the revision.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
      # `bend --verdict` builds its proof kernel only with exactly Lean 4.34.0 (bend2/safe.ts LEAN_VERSION).
      # nixpkgs went 4.30.0 -> 4.34.1 -> 4.35.0, so this is the official release binary (sha256 from the
      # GitHub release's asset digest).
      leanRelease = {
        x86_64-linux = { asset = "linux"; sha256 = "caaa98356098c85dc0fcbbd28e1ec66f39eb6551829972b752ff20e1286b646b"; };
        aarch64-darwin = { asset = "darwin_aarch64"; sha256 = "69f263fa6e21bbc2466bbfb1affcd92479ee2714c883a07de548e099a5922932"; };
      };
      lean4_34 = pkgs:
        let rel = leanRelease.${pkgs.stdenv.hostPlatform.system}; in
        pkgs.stdenv.mkDerivation {
          pname = "lean4-bin";
          version = "4.34.0";
          src = pkgs.fetchurl {
            url = "https://github.com/leanprover/lean4/releases/download/v4.34.0/lean-4.34.0-${rel.asset}.tar.zst";
            inherit (rel) sha256;
          };
          nativeBuildInputs = [ pkgs.zstd ] ++ pkgs.lib.optional pkgs.stdenv.hostPlatform.isLinux pkgs.autoPatchelfHook;
          buildInputs = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
          dontConfigure = true;
          dontBuild = true;
          dontStrip = true;
          installPhase = ''
            mkdir -p $out
            cp -r ./* $out/
          '';
        };
      # The BendTT kernel, from the bendtt.lean of the bend2 in use, built the way bend2/safe.ts builds it.
      # leanc links for generic Linux, so on NixOS the binary is patched; the devshell points $BENDTT at it,
      # which bend uses instead of building its own copy under ~/.bend.
      bendtt = pkgs: pkgs.stdenv.mkDerivation {
        pname = "bendtt";
        version = pkgs.bend2.version;
        src = "${pkgs.bend2}/share/bend2/bend2/bendtt.lean";
        dontUnpack = true;
        nativeBuildInputs = [ (lean4_34 pkgs) ] ++ pkgs.lib.optional pkgs.stdenv.hostPlatform.isLinux pkgs.autoPatchelfHook;
        buildPhase = ''
          cp $src bendtt.lean
          lean -c bendtt.c bendtt.lean
          leanc -O3 -DNDEBUG bendtt.c -o bendtt
        '';
        installPhase = ''
          install -Dm755 bendtt $out/bin/bendtt
        '';
      };
    in
    {
      packages = forAll (pkgs: { lean4_34 = lean4_34 pkgs; bendtt = bendtt pkgs; });

      devShells = forAll (pkgs: {
        # Bend 2 experiments (the batch planner). bend2 wraps its own clang for native builds;
        # `bend update` cannot write the Nix store, so upgrade by moving the lock. NoCC: bend reads
        # $CC and refuses one that is not clang 14+ (2.0.39), so the shell must not set CC=gcc.
        # `bend --verdict` uses $BENDTT, the kernel built above with Lean 4.34.0.
        default = pkgs.mkShellNoCC {
          packages = [ pkgs.bend2 (lean4_34 pkgs) ];
          BENDTT = "${bendtt pkgs}/bin/bendtt";
        };
      });
    };
}
