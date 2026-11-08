"""Evaluate cases independently; never join boundaries between surgeries."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from ml.datasets.cholec80 import validate_split
from ml.datasets.preprocessing import device
from ml.training.checkpoint import load_checkpoint
from ml.evaluation.metrics import evaluate
p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--alpha",type=float,default=1.);p.add_argument("--frames",action="store_true");a=p.parse_args()
if not 0<a.alpha<=1:p.error("alpha must be in (0,1]")
dev=device();net,ck=load_checkpoint(a.checkpoint,dev);split=json.loads((Path(a.data)/"split.json").read_text());validate_split(split)
if set(split["test"]) & set(ck["provenance"]["split"]["train"]+ck["provenance"]["split"]["val"]):raise ValueError("Held-out case leakage")
results={}
if not split["test"]:raise ValueError("No held-out cases")
if ck["config"].get("fine_tune","frozen")!="frozen" and not a.frames:raise ValueError("Fine-tuned CNN evaluation requires --frames")
with torch.inference_mode():
    for case in split["test"]:
        d=Path(a.data)/case;y=np.load(d/"labels.npy");pred=[];lat=[];state=None;prob=None
        if a.frames:
            from ml.datasets.preprocessing import preprocess
            source=np.load(d/"frames.npy",mmap_mode="r")
            # Causal path is bounded to one preprocessed frame. Offline bidirectional embeds in batches.
            x=(preprocess(f).to(dev) for f in source)
            if net.bidirectional:
                features=[]
                for start in range(0,len(source),16):features.append(net.cnn(torch.stack([preprocess(f) for f in source[start:start+16]]).to(dev)))
                x=torch.cat(features)
        else:x=torch.from_numpy(np.load(d/"features.npy")).float().to(dev)
        if net.bidirectional:
            logits,_=net(x.unsqueeze(0));pred=logits[0].argmax(-1).cpu().tolist()
        else:
            for frame in x:
                start=time.perf_counter();logits,state=net(frame[None,None],state=state);raw=logits[0,0].softmax(-1);prob=raw if prob is None else a.alpha*raw+(1-a.alpha)*prob;pred.append(int(prob.argmax()));lat.append((time.perf_counter()-start)*1000)
        results[case]=evaluate(y,pred,json.loads((d/"metadata.json").read_text())["sample_fps"],lat)
report={"synthetic_engineering_only":ck["provenance"].get("synthetic",False),"run_id":ck["provenance"]["run_id"],"checkpoint":Path(a.checkpoint).name,"mode":"offline_bidirectional" if net.bidirectional else "causal","smoothing_alpha":a.alpha,"cases":results,"macro_case_accuracy":float(np.mean([r["accuracy"] for r in results.values()])),"macro_case_f1":float(np.mean([r["macro_f1"] for r in results.values()])),"timing_scope":"CNN + temporal model; excludes preprocessing and decoding" if a.frames else "temporal model only; excludes CNN and decoding"}
Path(a.output).write_text(json.dumps(report,indent=2))
