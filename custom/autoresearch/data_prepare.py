"""Build a fixed, speaker/source separated real-audio speech-presence proxy.

Only full transcript-bearing LibriSpeech utterances are speech positives. These
labels do not label individual speech frames. MUSAN noise labels are inherited
from its corpus designation, not newly human-audited by this project.
"""
import collections, hashlib, json, pathlib, random, struct, wave
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / ".data"
SEED = 20260909

def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1024*1024): h.update(chunk)
    return h.hexdigest()

def info(p):
    if p.suffix == ".wav":
        with wave.open(str(p)) as f:
            assert f.getnchannels() == 1 and f.getframerate() == 16000
            return f.getnframes()
    with p.open("rb") as f:
        assert f.read(4) == b"fLaC"
        header = f.read(4)
        assert header[0] & 127 == 0 and int.from_bytes(header[1:], "big") == 34
        s = f.read(34); packed = int.from_bytes(s[10:18], "big")
        assert packed >> 44 == 16000 and ((packed >> 41) & 7) == 0
        return packed & ((1<<36)-1)

def groupsplit(groups, ntrain=None, ncal=None):
    ordered = sorted(set(groups), key=lambda s: hashlib.sha256((str(SEED)+s).encode()).digest())
    ntrain = ntrain or int(len(ordered)*.7)
    ncal = ncal or int(len(ordered)*.15)
    return {g: ("train" if i<ntrain else "calibration" if i<ntrain+ncal else "development") for i,g in enumerate(ordered)}

def main():
    rng = random.Random(SEED); clips = []; excluded = collections.Counter()
    speech = sorted((DATA/"audio/LibriSpeech/train-clean-5").glob("*/*/*.flac"))
    assert len(speech) > 1000
    speakers = groupsplit([p.parent.parent.name for p in speech],18,5)
    by_split = collections.defaultdict(list)
    for p in speech:
        n = info(p)
        if not 3*16000 <= n <= 20*16000:
            excluded["speech_duration_outside_3_to_20sec"] += 1; continue
        speaker = p.parent.parent.name
        row={"path":str(p), "sha256":digest(p), "label":1,"split":speakers[speaker],"offset_samples":0,"num_samples":n,"speaker_id":speaker,"source_id":"librispeech-speaker-"+speaker,"recording_id":p.stem,"condition":"clean_read_english","label_basis":"complete_transcript_bearing_utterance"}
        clips.append(row); by_split[row["split"]].append(n)
    noise_dir = DATA/"audio/RIRS_NOISES/pointsource_noises"
    kinds={}
    for line in (noise_dir/"noise_list").read_text().splitlines():
        fields=line.split(); kinds[pathlib.Path(fields[-1]).name]=fields[fields.index("--bg-fg-type")+1]
    noises=[]; seen=set()
    for p in sorted(noise_dir.glob("*.wav")):
        n=info(p)
        if n < 3*16000:
            excluded["noise_shorter_than_3sec"]+=1; continue
        h=digest(p)
        if h in seen:
            excluded["exact_duplicate_noise_recording"]+=1; continue
        seen.add(h); noises.append((p,n,h))
    assignments=groupsplit([h for p,n,h in noises])
    for p,n,h in noises:
        split=assignments[h]
        # Up to four disjoint real-audio windows from the same recording,
        # with lengths sampled from the same split's full speech utterances.
        offset=0
        for i in range(4):
            if n-offset < 3*16000: break
            target=min(rng.choice(by_split[split]), n-offset)
            row={"path":str(p),"sha256":h,"label":0,"split":split,"offset_samples":offset,"num_samples":target,"speaker_id":None,"source_id":"musan-noise-"+h,"recording_id":p.stem,"condition":"real_"+kinds.get(p.name,"unknown")+"_noise","label_basis":"MUSAN_noise_corpus_designation"}
            clips.append(row); offset+=target
    counts=collections.Counter((c["split"],c["label"]) for c in clips)
    for split in ("train","calibration","development"):
        assert counts[split,0]>=100 and counts[split,1]>=100, counts
    hashes=collections.defaultdict(set); sources=collections.defaultdict(set)
    for c in clips: hashes[c["sha256"]].add(c["split"]); sources[c["source_id"]].add(c["split"])
    assert all(len(v)==1 for v in hashes.values()), "duplicate recording across split"
    assert all(len(v)==1 for v in sources.values()), "source leakage"
    archives=[]
    for name,url,license in [("train-clean-5.tar.gz","https://www.openslr.org/resources/31/train-clean-5.tar.gz","CC-BY-4.0"),("rirs_noises.zip","https://www.openslr.org/resources/28/rirs_noises.zip","Apache-2.0 archive; selected pointsource recordings Public Domain per included LICENSE")]:
        p=DATA/"downloads"/name
        archives.append({"filename":name,"url":url,"bytes":p.stat().st_size,"sha256":digest(p),"license":license})
    manifest={"schema_version":1,"data_kind":"real_audio","label_granularity":"clip","sample_rate":16000,"max_clip_samples":320000,"seed":SEED,"sources":archives,"dataset_roles":{"speech":"MiniLibriSpeech train-clean-5, full 3-20s utterances; original official dev/test not downloaded or used","nonspeech":"RIRS_NOISES pointsource_noises: 843 FreeSound MUSAN recordings, selected >=3s unique recordings; real waveform windows only"},"split_policy":"18 train / 5 calibration / 5 development LibriSpeech speakers sorted by SHA256(seed+speaker); 70/15/15 percent unique noise recording hashes; all windows from a recording remain together","label_limitations":["Speech-presence whole-clip proxy only: no frame speech/no-speech ground truth or onset/offset claim.","Noise labels inherited from MUSAN corpus designation; residual unannotated speech contamination has not been ruled out by fresh human review.","Source recording and exact file hashes separated; contributor/session relations between different Freesound IDs are not provided in this archive.","Full utterance positives and nonspeech windows have approximate duration matching; this benchmark cannot establish streaming state behavior, domain robustness, or device energy."],"counts":{s:{"speech":counts[s,1],"nonspeech":counts[s,0]} for s in ("train","calibration","development")},"excluded":dict(excluded),"files":[],"clips":clips}
    unique={c["path"]:c["sha256"] for c in clips}
    manifest["files"]=[{"path":p,"sha256":h} for p,h in sorted(unique.items())]
    output=ROOT/"data_manifest.json"; output.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"manifest":str(output),"sha256":digest(output),"counts":manifest["counts"],"num_files":len(unique),"num_clips":len(clips),"download_bytes":sum(a["bytes"] for a in archives)}))
if __name__=="__main__": main()
