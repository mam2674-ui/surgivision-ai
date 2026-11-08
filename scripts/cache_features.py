import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch,yaml
from ml.models.cnn_lstm import CNNBackbone
from ml.datasets.preprocessing import preprocess,device
p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--config",default="ml/configs/default.yaml");a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());dev=device();net=CNNBackbone(cfg["backbone"],cfg["pretrained"]).to(dev).eval()
if not cfg["pretrained"]:raise ValueError("Research feature caches require pretrained weights")
with torch.inference_mode():
    for meta in Path(a.data).glob("*/metadata.json"):
        frames=np.load(meta.parent/"frames.npy",mmap_mode="r");out=np.lib.format.open_memmap(meta.parent/"features.npy",mode="w+",dtype=np.float32,shape=(len(frames),net.dim))
        for i in range(0,len(frames),16):out[i:i+16]=net(torch.stack([preprocess(f) for f in frames[i:i+16]]).to(dev)).cpu().numpy()
        out.flush();(meta.parent/"feature_provenance.json").write_text(json.dumps({"backbone":cfg["backbone"],"weights":"torchvision.DEFAULT","dim":net.dim,"preprocessing":"ImageNet resize224","metadata_sha256":hashlib.sha256(meta.read_bytes()).hexdigest()}))
