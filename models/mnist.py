import torch.nn as nn
import torch.nn.functional as F


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
