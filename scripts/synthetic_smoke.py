"""Engineering-only synthetic embeddings: not surgical training or benchmark evidence."""
import argparse,json
from pathlib import Path
import numpy as np,yaml
from ml.datasets.cholec80 import PHASE_KEYS
p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(42)
split={"train":["synthetic_train"],"val":["synthetic_val"],"test":["synthetic_test"]}
for cases in split.values():
    for case in cases:
        d=out/case;d.mkdir(exist_ok=True);labels=np.tile(np.arange(7),2);features=rng.normal(0,.2,(14,512)).astype(np.float32)
        for i,label in enumerate(labels):features[i,label]+=1
        np.save(d/"features.npy",features);np.save(d/"labels.npy",labels)
        (d/"metadata.json").write_text(json.dumps({"case_id":case,"timestamps":list(range(14)),"sample_fps":1.,"synthetic":True}))
        (d/"feature_provenance.json").write_text(json.dumps({"backbone":"resnet18","dim":512,"synthetic":True}))
(out/"split.json").write_text(json.dumps(split))
cfg=yaml.safe_load(Path("ml/configs/default.yaml").read_text());cfg.update(backbone="resnet18",pretrained=False,hidden=8,layers=1,dropout=0,epochs=1,batch_size=1,sequence_length=8)
(out/"smoke.yaml").write_text(yaml.safe_dump(cfg))
print("Synthetic engineering fixture written. Do not install its checkpoint in the research API.")
