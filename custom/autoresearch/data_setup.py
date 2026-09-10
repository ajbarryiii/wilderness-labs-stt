"""Download hash-pinned public audio outside the research timer, then prepare."""
import hashlib, pathlib, subprocess, sys, tarfile, urllib.request, zipfile
ROOT=pathlib.Path(__file__).resolve().parent
SOURCES=[
 ("train-clean-5.tar.gz","https://www.openslr.org/resources/31/train-clean-5.tar.gz","47805806c8b15f7549f3c51bd6b7da72ec0128a34d32830c8014a88658c0292d",332954390),
 ("rirs_noises.zip","https://www.openslr.org/resources/28/rirs_noises.zip","3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7",1311166223),
]
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  while b:=f.read(1024*1024): h.update(b)
 return h.hexdigest()
def main():
 import shutil
 assert shutil.disk_usage(ROOT).free>150*1024**3,"Keep at least150GiB free"
 downloads=ROOT/".data/downloads"; audio=ROOT/".data/audio"
 downloads.mkdir(parents=True,exist_ok=True);audio.mkdir(parents=True,exist_ok=True)
 for name,url,expected,size in SOURCES:
  p=downloads/name
  if not p.exists() or p.stat().st_size!=size or sha(p)!=expected:
   tmp=p.with_suffix(p.suffix+".partial")
   with urllib.request.urlopen(url,timeout=60) as response,tmp.open("wb") as f:
    total=0
    while b:=response.read(1024*1024):
     total+=len(b);assert total<=size,"Unexpected download size";f.write(b)
   assert tmp.stat().st_size==size and sha(tmp)==expected,"Download checksum mismatch"
   tmp.replace(p)
  print("verified",name,flush=True)
  if name.endswith(".tar.gz"):
   with tarfile.open(p) as archive: archive.extractall(audio,filter="data")
  else:
   with zipfile.ZipFile(p) as archive:
    members=[n for n in archive.namelist() if n.startswith("RIRS_NOISES/pointsource_noises/")]
    assert all(".." not in pathlib.PurePosixPath(n).parts for n in members)
    archive.extractall(audio,members=members)
 subprocess.run([sys.executable,str(ROOT/"data_prepare.py")],check=True)
if __name__=="__main__":main()
