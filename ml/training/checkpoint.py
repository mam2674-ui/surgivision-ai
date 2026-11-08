import torch
from ml.models.cnn_lstm import PhaseModel
from ml.datasets.cholec80 import PHASE_KEYS
def save_checkpoint(path,model,config,provenance):
    torch.save({"format":1,"state_dict":model.state_dict(),"config":config,"labels":PHASE_KEYS,"provenance":provenance},path)
def load_checkpoint(path,device="cpu"):
    ck=torch.load(path,map_location=device,weights_only=True)
    if ck.get("format")!=1 or ck.get("labels")!=PHASE_KEYS or not ck.get("provenance",{}).get("trained"):raise ValueError("Checkpoint lacks compatible trained provenance")
    config={**ck["config"],"pretrained":False};model=PhaseModel(**config).to(device);model.load_state_dict(ck["state_dict"],strict=True);model.eval()
    return model,ck
