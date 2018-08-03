import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50
from torchvision.models.inception import inception_v3

class SDEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(SDEncoder, self).__init__()

        self.backbone = resnet50(pretrained=False, num_classes=512)
        # self.backbone = inception_v3(pretrained=True)#, num_classes=emb_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, emb_dim)

    def forward(self, x, norm=True):


        if len(x.shape) < 4:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = F.relu(self.bn1(self.backbone(x)))
        x = self.fc1(x)
        # out = self.layer2(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        if norm:
            return F.normalize(x)
        return x
