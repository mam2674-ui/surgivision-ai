"""Cholec80 Frame/Phase TSV parsing. Canonical ordering; never infer phase progression."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
PHASE_KEYS = ["Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection", "GallbladderRetraction", "CleaningCoagulation", "GallbladderPackaging"]
PHASE_NAMES = ["Preparation", "Calot’s Triangle Dissection", "Clipping and Cutting", "Gallbladder Dissection", "Gallbladder Retraction", "Cleaning and Coagulation", "Gallbladder Packaging"]
def parse_annotations(path, labels=None):
    labels = PHASE_KEYS if labels is None else labels
    rows=[]
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            frame=int(row["Frame"]); phase=row["Phase"]
            if frame<0 or (rows and frame<=rows[-1][0]): raise ValueError("Frames must be nonnegative and strictly increasing")
            if phase not in labels: raise ValueError("Unknown phase: "+phase)
            rows.append((frame,labels.index(phase)))
    if not rows or rows[0][0]!=0: raise ValueError("Annotation must start at frame 0")
    return np.asarray(rows, dtype=np.int64)
def align_labels(indices, annotations):
    pos=np.searchsorted(annotations[:,0],indices,side="right")-1
    if np.any(pos<0) or np.any(np.asarray(indices)>annotations[-1,0]): raise ValueError("Video sample outside annotated coverage")
    return annotations[pos,1]
def standard_split():
    # Standard first-40 development / last-40 test; reserve cases 33–40 for validation.
    return {"train":[f"video{i:02}" for i in range(1,33)],"val":[f"video{i:02}" for i in range(33,41)],"test":[f"video{i:02}" for i in range(41,81)]}
def validate_split(split):
    sets=[set(split[k]) for k in ("train","val","test")]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)): raise ValueError("Case leakage")
    if any(len(split[k])!=len(set(split[k])) for k in split): raise ValueError("Duplicate case")
class SequenceDataset(Dataset):
    def __init__(self, root, cases, length=32, features=True):
        self.root=Path(root);self.length=length;self.features=features;self.windows=[]
        for case in cases:
            meta=json.loads((self.root/case/"metadata.json").read_text())
            for start in range(0,len(meta["timestamps"]),length):self.windows.append((case,start))
    def __len__(self):return len(self.windows)
    def __getitem__(self,i):
        case,start=self.windows[i]; d=self.root/case
        x=np.load(d/("features.npy" if self.features else "frames.npy"),mmap_mode="r")[start:start+self.length].copy()
        y=np.load(d/"labels.npy")[start:start+self.length].copy()
        if not self.features:
            from ml.datasets.preprocessing import preprocess
            x=torch.stack([preprocess(frame) for frame in x])
        else:x=torch.from_numpy(x).float()
        return x,torch.from_numpy(y).long(),case
def collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    x,y,c=zip(*batch)
    return pad_sequence(x,batch_first=True),pad_sequence(y,batch_first=True,padding_value=-100),torch.tensor([len(v) for v in y]),c
