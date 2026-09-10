"""Build and verify the pinned Nix CUDA runtime before any research timer."""
import json, pathlib, subprocess, sys, datetime
ROOT=pathlib.Path(__file__).resolve().parent
NIXPKGS="/nix/store/agyg2m8bkbcr2d5p57my3bla8794arlp-source"
def main():
 subprocess.run(["nix-store","--add-root",str(ROOT/".runtime-nixpkgs"),"--indirect","--realise",NIXPKGS],check=True,stdout=subprocess.DEVNULL)
 subprocess.run(["nix","build","--impure","--file",str(ROOT/"runtime.nix"),"--out-link",str(ROOT/".training-runtime")],check=True)
 runtime=str((ROOT/".training-runtime").resolve())
 evidence=json.loads(subprocess.check_output([str(ROOT/"runtime-python"),str(ROOT/"runtime_verify.py")],text=True))
 nix_info=json.loads(subprocess.check_output(["nix","path-info","--json","--json-format","2",NIXPKGS,runtime],text=True))
 lock={"schema_version":1,"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"nixpkgs_store_path":NIXPKGS,"runtime_store_path":runtime,"nix_paths":nix_info,"versions":evidence,"recipe":"runtime.nix; all package sources and transitive versions fixed by Nix source/store content","binary_overrides":{"nccl":"NVIDIA nvidia-nccl-cu12 2.28.9","nvshmem":"NVIDIA nvidia-nvshmem-cu12 3.6.5"}}
 (ROOT/"runtime_lock.json").write_text(json.dumps(lock,indent=2)+"\n")
 (ROOT/"runtime_evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
 print(json.dumps(evidence,indent=2))
if __name__=="__main__":main()
