#!/usr/bin/env python3
"""Fixed real-audio clip-presence benchmark. This is NOT a frame VAD evaluator."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
FEATURES = 16
MAX_BINS = 500
SCHEMA = 2

def read_json(path):
    return json.loads(Path(path).read_text())

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def logmel(audio):
    """Fixed 20ms trailing windows/10ms hop,16mel bands; average4frames=40ms."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < 320:
        audio = np.pad(audio, (0, 320-len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, 320)[::160]
    window = np.hanning(320).astype(np.float32)
    power = np.abs(np.fft.rfft(frames*window, n=512, axis=-1)) ** 2 / np.sum(window**2)
    mel = np.linspace(2595*np.log10(1+60/700), 2595*np.log10(1+7800/700), FEATURES+2)
    hz = 700*(10**(mel/2595)-1)
    frequencies = np.fft.rfftfreq(512, 1/SAMPLE_RATE)
    bank = np.stack([np.maximum(0,np.minimum((frequencies-hz[i])/(hz[i+1]-hz[i]),(hz[i+2]-frequencies)/(hz[i+2]-hz[i+1]))) for i in range(FEATURES)])
    banks = np.log(np.maximum(power@bank.T, 1e-8)).clip(-18,5)/6+1
    # Last trailing group has only its real frames; no artificial silence labels.
    values = np.stack([banks[i:i+4].mean(axis=0) for i in range(0,len(banks),4)],axis=-1)
    if values.shape[-1] > MAX_BINS:
        raise ValueError("Clip exceeds20s proxy limit; do not truncate a speech-positive utterance")
    return values.astype(np.float32)

def load_audio(clip):
    import soundfile as sf
    path = Path(clip["path"])
    offset = int(clip.get("offset_samples",0))
    samples = int(clip["num_samples"])
    audio,rate = sf.read(path, start=offset, frames=samples, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or len(audio) != samples:
        raise ValueError(f"Audio rate/length mismatch: {path}")
    return audio.mean(axis=1)

def manifest_clips(manifest):
    if manifest.get("data_kind") != "real_audio" or manifest.get("label_granularity") != "clip":
        raise ValueError("Scored benchmark requires real audio with explicit clip labels")
    clips = manifest["clips"]
    seen = {}
    group_splits = {}
    for clip in clips:
        split = clip["split"]
        if split not in ("train","calibration","development") or clip["label"] not in (0,1):
            raise ValueError("Invalid split/label")
        if not 320 <= int(clip["num_samples"]) <= 320000:
            raise ValueError("Invalid clip duration")
        identity = (clip["path"],int(clip.get("offset_samples",0)),int(clip["num_samples"]))
        if identity in seen:
            raise ValueError("Duplicate clip")
        seen[identity] = split
        group = ("speech",clip["speaker_id"]) if clip["label"] else ("noise",clip["source_id"])
        if not group[1]:
            raise ValueError("Missing split isolation group")
        if group in group_splits and group_splits[group] != split:
            raise ValueError("Speaker/noise source leakage across splits")
        group_splits[group] = split
    return clips

def cache_path(benchmark):
    return Path(benchmark["cache_path"])

def build_cache(output):
    manifest_path = HERE/"data_manifest.json"
    manifest = read_json(manifest_path)
    clips = manifest_clips(manifest)
    x = np.zeros((len(clips),FEATURES,MAX_BINS),dtype=np.float32)
    lengths = np.zeros(len(clips),dtype=np.int64)
    for i,clip in enumerate(clips):
        features = logmel(load_audio(clip))
        x[i,:,:features.shape[-1]] = features
        lengths[i] = features.shape[-1]
        if (i+1)%250 == 0:
            print(f"Prepared {i+1}/{len(clips)} real clips",flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(output, x=x, lengths=lengths, y=np.asarray([c["label"] for c in clips],dtype=np.float32),
             splits=np.asarray([c["split"] for c in clips]), conditions=np.asarray([c["condition"] for c in clips]),
             manifest_sha256=np.asarray(sha256(manifest_path)))
    print(json.dumps({"cache_path":str(output),"sha256":sha256(output),"clips":len(clips),"bytes":output.stat().st_size}))

def load_cache(benchmark, verify_hash=True):
    path = cache_path(benchmark)
    if verify_hash and sha256(path) != benchmark["cache_sha256"]:
        raise ValueError("Frozen feature cache checksum mismatch")
    data = np.load(path,allow_pickle=False)
    if str(data["manifest_sha256"]) != sha256(HERE/"data_manifest.json"):
        raise ValueError("Manifest/cache mismatch")
    for split in ("train","calibration","development"):
        for label in (0,1):
            count = np.sum((data["splits"] == split)&(data["y"] == label))
            if count < int(benchmark["minimum_samples_per_class"][split]):
                raise ValueError(f"Insufficient samples: {split}/{label}={count}")
    return data

def validate_model(candidate):
    """Load safe arrays/data, never execute candidate Python or pickle."""
    config = read_json(Path(candidate)/"model.json")
    if config["schema_version"] != SCHEMA or config["model_type"] != "causal_ternary_clip_cnn":
        raise ValueError("Unsupported inference graph")
    if config["activation_precision"] != "fp32_reference" or config["learned_bias"] is not False:
        raise ValueError("Precision contract changed")
    layers = config["layers"]
    if not 1 <= len(layers) <= 8:
        raise ValueError("Invalid layer count")
    arrays = np.load(Path(candidate)/"weights.npz",allow_pickle=False)
    parameters = 0
    tensor_specs = []
    channels = FEATURES
    for i,layer in enumerate(layers):
        width,kernel = int(layer["channels"]),int(layer["kernel"])
        if not 1 <= width <= 256 or not 1 <= kernel <= 15:
            raise ValueError("Unsupported layer dimensions")
        tensor_specs.append((f"conv{i}",(width,channels,kernel)))
        channels=width
    tensor_specs.append(("head",(1,channels*2)))
    if set(arrays.files) != {name+suffix for name,_ in tensor_specs for suffix in ("_codes","_scale")}:
        raise ValueError("Unexpected or missing learned tensors")
    tensors = []
    packed = []
    for name,shape in tensor_specs:
        codes,scale = arrays[name+"_codes"],arrays[name+"_scale"]
        if codes.dtype != np.int8 or codes.shape != shape or not np.isin(codes,[-1,0,1]).all():
            raise ValueError("Invalid ternary code tensor")
        if scale.shape != () or not np.isfinite(scale) or float(scale)<=0:
            raise ValueError("Invalid disclosed per-tensor scale")
        parameters += codes.size
        tensors.append((codes.astype(np.float32)*float(scale)).copy())
        unsigned=(codes.reshape(-1)+1).astype(np.uint8)
        unsigned=np.pad(unsigned,(0,(-len(unsigned))%4),constant_values=1)
        packed.append((unsigned[0::4]|unsigned[1::4]<<2|unsigned[2::4]<<4|unsigned[3::4]<<6).tobytes())
    if not 0 < parameters <= 50000:
        raise ValueError("Parameter budget exceeded")
    if (Path(candidate)/"weights.2bit").read_bytes() != b"".join(packed):
        raise ValueError("Packed export differs from audited ternary codes")
    return config,tensors,parameters

def forward_reference(x,lengths,tensors,config):
    """Independent fixed graph, shared only as immutable training support."""
    import torch
    import torch.nn.functional as F
    for weight,layer in zip(tensors[:-1],config["layers"]):
        x = F.conv1d(F.pad(x,(int(layer["kernel"])-1,0)),weight)
        x = F.relu(x * torch.rsqrt(x.square().mean(dim=1,keepdim=True)+1e-5))
    mask = torch.arange(x.shape[-1],device=x.device)[None,:] < lengths[:,None]
    mean = (x*mask[:,None,:]).sum(dim=-1)/lengths[:,None]
    maximum = x.masked_fill(~mask[:,None,:],-1e9).amax(dim=-1)
    return F.linear(torch.cat([mean,maximum],dim=-1),tensors[-1]).squeeze(-1)

def choose_threshold(scores,labels,target_fnr=0.01):
    positives = np.sort(np.asarray(scores)[np.asarray(labels)==1])
    if not len(positives):
        raise ValueError("No calibration positives")
    return float(positives[min(len(positives)-1,math.floor(target_fnr*len(positives)))])

def binary_metrics(scores,labels,threshold):
    scores,labels = np.asarray(scores),np.asarray(labels)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite scores")
    speech,noise = labels==1,labels==0
    if not speech.any() or not noise.any():
        raise ValueError("Empty metric denominator")
    positive = scores>=threshold
    return {"fnr":float(np.mean(~positive[speech])),"fpr":float(np.mean(positive[noise])),
            "speech_samples":int(speech.sum()),"nonspeech_samples":int(noise.sum())}

def auroc(scores,labels):
    scores,labels = np.asarray(scores),np.asarray(labels)
    order = np.argsort(scores,kind="stable")
    ordered=scores[order]
    ranks=np.empty(len(scores),dtype=float)
    starts=np.r_[0,np.flatnonzero(np.diff(ordered)!=0)+1]
    ends=np.r_[starts[1:],len(scores)]
    for start,end in zip(starts,ends):
        ranks[order[start:end]]=(start+1+end)/2
    positives=labels==1
    p,n=int(positives.sum()),int((~positives).sum())
    if not p or not n:
        raise ValueError("AUROC requires both labels")
    return float((ranks[positives].sum()-p*(p+1)/2)/(p*n))

def discrimination_gate(scores,labels,metric):
    score_range=float(np.ptp(scores))
    area=auroc(scores,labels)
    if score_range < 1e-5 or area <= 0.51 or metric["fnr"]>=0.95 or metric["fpr"]>=0.95 or (metric["fnr"]+metric["fpr"])/2>=0.49:
        raise ValueError(f"Degenerate/non-discriminative classifier: range={score_range},AUROC={area},metrics={metric}")
    return score_range,area

def evaluate(candidate,output):
    import torch
    torch.set_num_threads(1)
    benchmark=read_json(HERE/"benchmark.json")
    data=load_cache(benchmark)
    config,weights,parameters=validate_model(candidate)
    tensors=[torch.from_numpy(w) for w in weights]
    scores={}
    with torch.inference_mode():
        for split in ("calibration","development"):
            idx=np.flatnonzero(data["splits"]==split)
            values=[]
            for start in range(0,len(idx),64):
                batch=idx[start:start+64]
                x=torch.from_numpy(data["x"][batch])
                length=torch.from_numpy(data["lengths"][batch])
                values.append(forward_reference(x,length,tensors,config).numpy())
            scores[split]=np.concatenate(values)
    calibration_labels=data["y"][data["splits"]=="calibration"]
    threshold=choose_threshold(scores["calibration"],calibration_labels,benchmark["target_fnr"])
    devmask=data["splits"]=="development"
    labels=data["y"][devmask]
    metric=binary_metrics(scores["development"],labels,threshold)
    # Hard validity checks apply before all rankings, including infeasible candidates.
    score_range,area=discrimination_gate(scores["development"],labels,metric)
    conditions=data["conditions"][devmask]
    breakdown={}
    for condition in np.unique(conditions):
        sel=conditions==condition
        predictions=scores["development"][sel]>=threshold
        truth=labels[sel]
        breakdown[str(condition)]={"count":int(sel.sum()),"fnr":float(np.mean(~predictions[truth==1])) if np.any(truth==1) else None,
                                   "fpr":float(np.mean(predictions[truth==0])) if np.any(truth==0) else None}
    # Complete fixed one-second mono PCM->DSP->classifier workload, remote x86 FP32.
    devclip=next(c for c in read_json(HERE/"data_manifest.json")["clips"] if c["split"]=="development")
    audio=load_audio(devclip)[:SAMPLE_RATE]
    if len(audio)<SAMPLE_RATE:
        audio=np.pad(audio,(0,SAMPLE_RATE-len(audio)))
    timings=[]
    with torch.inference_mode():
        for iteration in range(120):
            before=time.perf_counter_ns()
            feature=logmel(audio)
            feature_tensor=torch.from_numpy(feature[None])
            length=torch.tensor([feature.shape[-1]])
            forward_reference(feature_tensor,length,tensors,config)
            elapsed=(time.perf_counter_ns()-before)/1e6
            if iteration>=20: timings.append(elapsed)
    submitted_training=config["training"]
    # Never merge arbitrary candidate metadata over independently calculated metrics.
    training_keys=("training_cuda_verified","training_device","gpu_name","training_steps","training_seconds","process_seconds","peak_vram_mb","initial_train_loss","final_train_loss","master_weight_l1_change","seed","batch_size","optimizer","learning_rate","training_activation_precision","torch_version","cuda_version","train_samples")
    training={key:submitted_training[key] for key in training_keys}
    for key in ("training_seconds","process_seconds","peak_vram_mb","initial_train_loss","final_train_loss","master_weight_l1_change","learning_rate"):
        value=training[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(f"Invalid training metric: {key}")
    if training["training_seconds"]<=0 or training["master_weight_l1_change"]<=0 or type(training["training_steps"]) is not int:
        raise ValueError("Training lacked positive elapsed update time/changed weights/integer steps")
    if not str(training["training_device"]).startswith("cuda") or int(training["training_steps"]) <= 0 or training.get("training_cuda_verified") is not True:
        raise ValueError("Candidate did not train on CUDA")
    metric.update({"schema_version":SCHEMA,"data_kind":"real_audio","label_granularity":"clip",
        "cpu_p95_ms":float(np.quantile(timings,.95)),"cpu_timing_workload":"one second mono16k PCM, fixed DSP+FP32 independent model; one x86 CPU thread",
        "num_parameters":parameters,"precision_ok":True,"ternary_code_bits":2,
        "packed_weight_bytes":sum((w.size+3)//4 for w in weights),"scale_bytes":len(tensors)*4,
        "activation_precision":"fp32_reference","arithmetic":"expanded scaled ternary weights, ordinary FP32 convolutions; not packed execution",
        "threshold":threshold,"auroc":area,"score_range":score_range,
        "balanced_error_rate":(metric["fnr"]+metric["fpr"])/2,
        "feasible":metric["fnr"]<=benchmark["target_fnr"] and metric["fpr"]<=benchmark["max_fpr"],
        "calibration":binary_metrics(scores["calibration"],calibration_labels,threshold),
        "conditions":breakdown,"training":training,
        "training_cuda_verified":training["training_cuda_verified"],"training_device":training["training_device"],
        "training_steps":training["training_steps"],"training_seconds":training["training_seconds"],
        "gpu_name":training["gpu_name"],"peak_vram_mb":training["peak_vram_mb"],
        "cache_sha256":benchmark["cache_sha256"],"manifest_sha256":sha256(HERE/"data_manifest.json"),
        "iphone_joules":None,"frame_vad_metrics":None,
        "limitations":["Whole-utterance presence proxy; no frame activity labels","Public corpus/source differences may inflate discrimination","No held-out final test or iPhone energy evaluation"]})
    Path(output).write_text(json.dumps(metric,indent=2,allow_nan=False)+"\n")
    print(json.dumps(metric,allow_nan=False))

def validate_data():
    manifest=read_json(HERE/"data_manifest.json")
    clips=manifest_clips(manifest)
    files={f["path"]:f["sha256"] for f in manifest["files"]}
    for clip in clips:
        if clip["path"] not in files or clip["sha256"] != files[clip["path"]]:
            raise ValueError("Clip provenance missing from frozen files")
    for path,digest in files.items():
        if sha256(path) != digest:
            raise ValueError(f"Raw audio checksum mismatch: {path}")
    benchmark=read_json(HERE/"benchmark.json")
    data=load_cache(benchmark)
    print(json.dumps({"data_kind":"real_audio","label_granularity":"clip","audio_files_verified":len(files),"clips":len(clips),"cache_sha256":benchmark["cache_sha256"]}))

def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("validate-data")
    cache=commands.add_parser("cache");cache.add_argument("--output",required=True)
    evaluator=commands.add_parser("evaluate");evaluator.add_argument("--candidate",required=True);evaluator.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="cache":build_cache(args.output)
    elif args.command=="validate-data":validate_data()
    else:evaluate(args.candidate,args.output)

if __name__=="__main__":main()
