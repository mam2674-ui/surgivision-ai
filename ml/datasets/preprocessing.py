import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
_transform=Compose([Resize((224,224)),ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])])
def preprocess(rgb):return _transform(Image.fromarray(rgb))
class OpenCVDecoder:
    def __init__(self,path,sample_fps=1.0):
        if sample_fps<=0:raise ValueError("sample_fps must be positive")
        self.cap=cv2.VideoCapture(str(path));self.sample_fps=sample_fps
        self.fps=self.cap.get(cv2.CAP_PROP_FPS);self.count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.cap.isOpened() or not np.isfinite(self.fps) or self.fps<=0 or self.count<=0:
            self.cap.release();raise ValueError("Invalid or unsupported video")
        if sample_fps>self.fps:self.cap.release();raise ValueError("Sample FPS exceeds source FPS")
        self.duration=self.count/self.fps
    def __iter__(self):
        # Cholec80 is constant-rate 25 FPS. Frame index is the source timestamp.
        # Variable frame-rate sources require a timestamp-aware PyAV decoder before use.
        next_time=0.;index=0
        try:
            while True:
                ok,frame=self.cap.read()
                if not ok:break
                t=index/self.fps
                if t+1e-7>=next_time:
                    yield index,t,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    next_time+=1/self.sample_fps
                index+=1
            if index < self.count-1:raise ValueError("Video ended before declared frame count")
        finally:self.cap.release()
    def close(self):self.cap.release()
def device():
    if torch.cuda.is_available():return torch.device("cuda")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available():return torch.device("mps")
    return torch.device("cpu")
