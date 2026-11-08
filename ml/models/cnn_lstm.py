import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torchvision import models
class CNNBackbone(nn.Module):
    def __init__(self,name="resnet50",pretrained=True,fine_tune="frozen"):
        super().__init__()
        if name not in ("resnet18","resnet50"):raise ValueError("Unsupported backbone")
        weights=(models.ResNet50_Weights.DEFAULT if name=="resnet50" else models.ResNet18_Weights.DEFAULT) if pretrained else None
        self.net=getattr(models,name)(weights=weights);self.dim=self.net.fc.in_features;self.net.fc=nn.Identity();self.fine_tune=fine_tune
        if fine_tune not in ("frozen","last","full"):raise ValueError("Invalid fine-tune mode")
        for p in self.net.parameters():p.requires_grad=fine_tune=="full"
        if fine_tune=="last":
            for p in self.net.layer4.parameters():p.requires_grad=True
    def train(self,mode=True):
        super().train(mode)
        if self.fine_tune!="full":
            self.net.eval()
            if self.fine_tune=="last":self.net.layer4.train(mode)
        return self
    def forward(self,x):return self.net(x)
class PhaseModel(nn.Module):
    def __init__(self,backbone="resnet50",pretrained=True,fine_tune="frozen",model="lstm",hidden=512,layers=2,dropout=.3,bidirectional=False,classes=7,**_):
        super().__init__();self.cnn=CNNBackbone(backbone,pretrained,fine_tune);self.kind=model;self.bidirectional=bidirectional
        if model not in ("cnn","lstm"):raise ValueError("Invalid model")
        self.temporal=nn.LSTM(self.cnn.dim,hidden,layers,batch_first=True,dropout=dropout if layers>1 else 0,bidirectional=bidirectional) if model=="lstm" else None
        dim=hidden*(2 if bidirectional else 1) if self.temporal else self.cnn.dim
        self.head=nn.Sequential(nn.Linear(dim,256),nn.ReLU(),nn.Dropout(dropout),nn.Linear(256,classes))
    def forward(self,x,lengths=None,state=None):
        if x.ndim==5:
            b,t,c,h,w=x.shape;x=self.cnn(x.reshape(b*t,c,h,w)).reshape(b,t,-1)
        if self.temporal:
            if lengths is not None:
                packed=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
                packed,state=self.temporal(packed,state);x,_=pad_packed_sequence(packed,batch_first=True,total_length=x.shape[1])
            else:x,state=self.temporal(x,state)
        return self.head(x),state
