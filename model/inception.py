import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class Model(nn.Module):
    def __init__(self,bm='adv_inception_v3'):
        super(Model, self).__init__()
        self.backbone = timm.create_model(bm, pretrained=True, num_classes=500, in_chans=13)
        self.mlp = nn.Sequential(
            nn.Linear(18, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(64, 64),
            # nn.LayerNorm(64),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.fc = nn.Linear(64+500*2, 1)

    def forward(self, img, feature):
        # print(img.shape)
        b, c, h, w = img.shape
        img = img.reshape(b*2, c//2, h, w)
        img = self.backbone(img).reshape(b, -1)
        # print(img.shape)
        feature = self.mlp(feature)
        y = self.fc(torch.cat([img, feature], dim=1))
        return y