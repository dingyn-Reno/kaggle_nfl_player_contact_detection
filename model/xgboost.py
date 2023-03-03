import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        pass
    def forward(self, input):
        x=self.f1(input)
        x=nn.ReLu()(x)
        x=self.GAP(x)
        x=nn.Flatten()(x)
        x=nn.Softmax()(x)
        return x

if __name__=='__main__':
    model=LeNet()
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
