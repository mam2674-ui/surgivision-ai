"""Representation attribution for current frame, given a fixed prior LSTM state."""
import torch
import torch.nn.functional as F
from ml.datasets.preprocessing import preprocess

def gradcam(model,rgb,phase_id,state=None):
    # MPS single-step LSTM backward can abort natively. Use a CPU attribution copy.
    import copy
    if next(model.parameters()).device.type=="mps":
        model=copy.deepcopy(model).cpu()
        state=None if state is None else tuple(v.detach().cpu() for v in state)
    model.eval();device=next(model.parameters()).device;captured={}
    def hook(_m,_i,out):captured["activation"]=out;out.retain_grad()
    handle=model.cnn.net.layer4.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        x=preprocess(rgb).to(device)[None,None].requires_grad_(True)
        prior=None if state is None else tuple(v.detach() for v in state)
        with torch.enable_grad():
            logits,_=model(x,state=prior);logits[0,-1,phase_id].backward();act=captured["activation"];weights=act.grad.mean((2,3),keepdim=True);cam=(weights*act).sum(1,keepdim=True).relu();cam=F.interpolate(cam,size=rgb.shape[:2],mode="bilinear",align_corners=False)[0,0];cam=cam/(cam.max()+1e-8)
        return cam.detach().cpu().numpy()
    finally:handle.remove();model.zero_grad(set_to_none=True)
