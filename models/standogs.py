import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50
from torchvision.models.inception import inception_v3

class SDEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(SDEncoder, self).__init__()

        self.backbone = resnet50(pretrained=False, num_classes=emb_dim)
        # self.backbone = inception_v3(pretrained=True)#, num_classes=emb_dim)

        # self.fc1 = nn.Linear(1000, emb_dim)

    def forward(self, x, norm=False):


        if len(x.shape) < 4:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        out = self.backbone(x)
        # out = self.layer2(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        if norm:
            return F.normalize(out)
        return out
