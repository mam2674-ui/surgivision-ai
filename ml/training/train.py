import argparse,json,random,time,uuid
from pathlib import Path
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score,accuracy_score
from ml.models.cnn_lstm import PhaseModel
from ml.datasets.cholec80 import SequenceDataset,collate,validate_split
from ml.datasets.preprocessing import device
from ml.training.checkpoint import save_checkpoint
from ml.training.losses import FocalLoss

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--config",default="ml/configs/default.yaml");p.add_argument("--output",default="runs");p.add_argument("--frames",action="store_true");a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text());seed=cfg["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    split=json.loads((Path(a.data)/"split.json").read_text());validate_split(split)
    if not a.frames and cfg["fine_tune"]!="frozen":raise ValueError("Fine-tuning requires --frames")
    if not a.frames:
        for case in split["train"]+split["val"]:
            signature=json.loads((Path(a.data)/case/"feature_provenance.json").read_text())
            if signature["backbone"]!=cfg["backbone"]:raise ValueError("Feature cache backbone mismatch")
            if signature.get("synthetic") and cfg.get("pretrained"):raise ValueError("Synthetic cache cannot be used for surgical training")
    loaders={k:DataLoader(SequenceDataset(a.data,split[k],cfg["sequence_length"],not a.frames),batch_size=cfg["batch_size"],shuffle=k=="train",collate_fn=collate) for k in ("train","val")}
    if any(len(v.dataset)==0 for v in loaders.values()):raise ValueError("Empty development split")
    dev=device();net=PhaseModel(**cfg).to(dev);opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,net.parameters()),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=2);run=Path(a.output)/uuid.uuid4().hex[:12];run.mkdir(parents=True);writer=SummaryWriter(str(run));best=float("inf");stale=0;started=time.time()
    weights=None
    if cfg["class_weighting"]:
        counts=np.zeros(7)
        for case in split["train"]:counts+=np.bincount(np.load(Path(a.data)/case/"labels.npy"),minlength=7)
        weights=torch.tensor(counts.sum()/(7*np.maximum(counts,1)),dtype=torch.float32,device=dev)
    loss_fn=FocalLoss(weights) if cfg.get("loss")=="focal" else torch.nn.CrossEntropyLoss(weight=weights,ignore_index=-100)
    scaler=torch.amp.GradScaler("cuda",enabled=dev.type=="cuda")
    for epoch in range(cfg["epochs"]):
        for stage,loader in loaders.items():
            net.train(stage=="train");total=0.;count=0;truth=[];pred=[]
            for x,y,lengths,_ in loader:
                x=x.to(dev);y=y.to(dev);opt.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(stage=="train"),torch.autocast(device_type=dev.type,enabled=dev.type=="cuda"):
                    logits,_=net(x,lengths);loss=loss_fn(logits.reshape(-1,7),y.reshape(-1))
                if stage=="train":
                    scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(net.parameters(),cfg["grad_clip"]);scaler.step(opt);scaler.update()
                mask=y!=-100;n=int(mask.sum());total+=loss.item()*n;count+=n;truth.extend(y[mask].cpu().tolist());pred.extend(logits.argmax(-1)[mask].cpu().tolist())
            score=total/count;writer.add_scalar(stage+"/loss",score,epoch);writer.add_scalar(stage+"/accuracy",accuracy_score(truth,pred),epoch);writer.add_scalar(stage+"/macro_f1",f1_score(truth,pred,labels=list(range(7)),average="macro",zero_division=0),epoch)
            for i,f in enumerate(f1_score(truth,pred,labels=list(range(7)),average=None,zero_division=0)):writer.add_scalar(f"{stage}/phase_{i}_f1",f,epoch)
            if stage=="val":
                scheduler.step(score)
                if score<best:
                    best=score;stale=0;save_checkpoint(run/"best.pt",net,cfg,{"trained":True,"run_id":run.name,"split":split,"seed":seed,"epoch":epoch,"duration_seconds":time.time()-started,"data_root_name":Path(a.data).name,"synthetic":any(c.startswith("synthetic_") for c in split["train"])})
                else:stale+=1
        writer.add_scalar("learning_rate",opt.param_groups[0]["lr"],epoch);print(epoch,best)
        if stale>=cfg["patience"]:break
    (run/"config.json").write_text(json.dumps(cfg,indent=2));writer.close()
if __name__=="__main__":main()
