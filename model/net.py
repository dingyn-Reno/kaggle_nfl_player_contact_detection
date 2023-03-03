import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.f1=nn.Conv2d(1, 10, kernel_size=5)
        self.l1=nn.Linear(490, 10)
        self.GAP=nn.AvgPool2d(kernel_size=2)

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
