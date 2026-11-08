import time
import torch
from ml.training.checkpoint import load_checkpoint
from ml.datasets.preprocessing import device,preprocess
from ml.datasets.cholec80 import PHASE_NAMES
class Predictor:
    def __init__(self,checkpoint,streaming=True,alpha=None,threshold=None):
        self.device=device();self.model,self.checkpoint=load_checkpoint(checkpoint,self.device)
        if self.checkpoint["provenance"].get("synthetic"):raise ValueError("Synthetic engineering checkpoints cannot generate surgical predictions")
        if streaming and self.model.bidirectional:raise ValueError("Bidirectional checkpoints cannot stream")
        config=self.checkpoint["config"]
        self.sample_fps=float(config.get("sample_fps",1.));self.alpha=float(config.get("smoothing_alpha",1.) if alpha is None else alpha);self.threshold=float(config.get("uncertainty_threshold",.65) if threshold is None else threshold)
        if not 0<self.alpha<=1 or not 0<=self.threshold<=1 or not 0<self.sample_fps:raise ValueError("Invalid inference configuration")
        self.streaming=streaming;self.reset()
    def reset(self):self.state=None;self.probs=None;self.last=-1
    @torch.inference_mode()
    def predict(self,rgb,t):
        if self.model.bidirectional:raise ValueError("Use full-sequence evaluator for bidirectional checkpoints")
        if t<=self.last:raise ValueError("Reset temporal state before seeking")
        start=time.perf_counter();x=preprocess(rgb).unsqueeze(0).unsqueeze(0).to(self.device)
        logits,self.state=self.model(x,state=self.state);p=logits[0,-1].softmax(-1)
        self.probs=p if self.probs is None else self.alpha*p+(1-self.alpha)*self.probs
        probs=self.probs.cpu().tolist();idx=int(self.probs.argmax());self.last=t
        return {"video_time_seconds":t,"phase_id":idx,"predicted_phase":PHASE_NAMES[idx],"confidence":probs[idx],"sample_fps":self.sample_fps,"probabilities":probs,"uncertain":probs[idx]<self.threshold,"inference_mode":"causal_streaming" if self.streaming else "causal_offline","model_version":self.checkpoint["provenance"].get("run_id","unknown"),"latency_ms":(time.perf_counter()-start)*1000}
