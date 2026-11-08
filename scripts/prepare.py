"""Extract authorized case data without keeping all decoded frames in memory."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from ml.datasets.cholec80 import parse_annotations,align_labels,standard_split,validate_split
from ml.datasets.preprocessing import OpenCVDecoder
p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",required=True);p.add_argument("--fps",type=float,default=1);p.add_argument("--authorized",action="store_true");a=p.parse_args()
if not a.authorized:p.error("Confirm lawful research use with --authorized after reviewing the dataset license")
root=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
split=standard_split();validate_split(split);(out/"split.json").write_text(json.dumps(split,indent=2))
for video in sorted((root/"videos").glob("video*.mp4")):
    import cv2
    annpath=root/"phase_annotations"/(video.stem+"-phase.txt");ann=parse_annotations(annpath)
    decoder=OpenCVDecoder(video,a.fps);target=out/video.stem;target.mkdir(exist_ok=True)
    n=int(np.floor((decoder.count-1)/decoder.fps*a.fps+1e-7))+1;frames=np.lib.format.open_memmap(target/"frames.npy",mode="w+",dtype=np.uint8,shape=(n,224,224,3))
    indices=[];times=[]
    for i,(idx,t,rgb) in enumerate(decoder):frames[i]=cv2.resize(rgb,(224,224));indices.append(idx);times.append(t)
    frames.flush();del frames
    if len(times)!=n:raise ValueError("Extraction count mismatch")
    np.save(target/"labels.npy",align_labels(indices,ann))
    meta={"case_id":video.stem,"timestamps":times,"frame_indices":indices,"source_fps":decoder.fps,"sample_fps":a.fps,"annotation_sha256":hashlib.sha256(annpath.read_bytes()).hexdigest()}
    (target/"metadata.json").write_text(json.dumps(meta));print(video.stem,len(times))
