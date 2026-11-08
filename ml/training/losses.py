import torch
from torch import nn
class FocalLoss(nn.Module):
    def __init__(self,weight=None,gamma=2.):super().__init__();self.weight=weight;self.gamma=gamma
    def forward(self,logits,targets):
        mask=targets!=-100;logits=logits[mask];targets=targets[mask]
        logp=logits.log_softmax(-1).gather(1,targets[:,None]).squeeze(1);loss=-(1-logp.exp()).pow(self.gamma)*logp
        if self.weight is not None:loss=loss*self.weight[targets]
        return loss.mean()
