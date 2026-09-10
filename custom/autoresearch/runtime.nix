# Exact nixpkgs source is GC-rooted at .runtime-nixpkgs by runtime_setup.py.
let
  pkgs = import /nix/store/agyg2m8bkbcr2d5p57my3bla8794arlp-source { config.allowUnfree = true; };
  # Use NVIDIA's published binary NCCL. The default Nix expression compiles
  # every GPU architecture; that is unnecessary for this single-GPU pilot.
  ncclBinary = pkgs.stdenv.mkDerivation {
    pname = "nccl-binary";
    version = "2.28.9";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/4a/4e/44dbb46b3d1b0ec61afda8e84837870f2f9ace33c564317d59b70bc19d3e/nvidia_nccl_cu12-2.28.9-py3-none-manylinux_2_18_x86_64.whl";
      sha256 = "485776daa8447da5da39681af455aa3b2c2586ddcf4af8772495e7c532c7e5ab";
    };
    nativeBuildInputs = [ pkgs.unzip pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib pkgs.cudaPackages.cuda_cudart ];
    dontUnpack = true;
    installPhase = ''
      mkdir -p "$out"
      unzip -q "$src" -d wheel
      cp -r wheel/nvidia/nccl/lib "$out/"
      cp -r wheel/nvidia/nccl/include "$out/"
    '';
  };
  nvshmemBinary = pkgs.stdenv.mkDerivation {
    pname = "nvshmem-binary";
    version = "3.6.5";
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/9e/da/36fa8307cc40889307fed415d70b67d35ec330ffce889a9c03cf8f616cfa/nvidia_nvshmem_cu12-3.6.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl";
      sha256 = "f86db35f1ced21a790fa255dcae7db8998bf8655a95e76c033a6574190b398e4";
    };
    nativeBuildInputs = [ pkgs.unzip pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib pkgs.cudaPackages.cuda_cudart ncclBinary pkgs.rdma-core pkgs.libfabric pkgs.ucx pkgs.openmpi pkgs.pmix ];
    autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" "libnvidia-ml.so.1" ];
    dontUnpack = true;
    installPhase = ''
      mkdir -p "$out/lib"
      unzip -q "$src" -d wheel
      cp -r wheel/nvidia/nvshmem/lib/* "$out/lib/"
    '';
  };
  torchBinary = pkgs.python3Packages.torch-bin.override {
    cudaPackages = pkgs.cudaPackages // { nccl = ncclBinary; libnvshmem = nvshmemBinary; };
  };
in pkgs.python3.withPackages (p: [ torchBinary p.numpy p.scipy p.soundfile ])
