import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50
from torchvision.models.inception import inception_v3


class MNISTEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(MNISTEncoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(7 * 7 * 64, emb_dim)

    def forward(self, x, norm=False):
        if len(x.shape) < 4:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        if norm:
            return F.normalize(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, emb_dim, fc_dim=None, norm=True):
        super(ResNetEncoder, self).__init__()
        # self.backbone = inception_v3(pretrained=True)#, num_classes=emb_dim)

        self.fc_dim = fc_dim
        self.norm = norm

        if fc_dim:
            self.backbone = resnet50(pretrained=False, num_classes=fc_dim)
            self.bn1 = nn.BatchNorm1d(fc_dim)
            self.fc1 = nn.Linear(fc_dim, emb_dim)
        else:
            self.backbone = resnet50(pretrained=False, num_classes=emb_dim)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.backbone(x)

        if self.fc_dim:
            x = F.relu(self.bn1(x))
            x = self.fc1(x)

        if self.norm:
            x = F.normalize(x)

        return x
