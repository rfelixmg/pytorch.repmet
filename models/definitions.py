import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
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
    def __init__(self, emb_dim, type=18, fc_dim=None, norm=True, pretrained=True, lock=False):
        super(ResNetEncoder, self).__init__()

        self.fc_dim = fc_dim
        self.norm = norm
        self.pretrained = pretrained

        if type == 18:
            if self.pretrained:
                self.backbone = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
                if lock:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                ll_size = 512
            elif fc_dim:
                self.backbone = resnet18(pretrained=False, num_classes=fc_dim)
            else:
                self.backbone = resnet18(pretrained=False, num_classes=emb_dim)
        elif type == 50:
            if self.pretrained:
                self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-1])
                if lock:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                ll_size = 2048
            elif fc_dim:
                self.backbone = resnet50(pretrained=False, num_classes=fc_dim)
            else:
                self.backbone = resnet50(pretrained=False, num_classes=emb_dim)

        if self.pretrained:
            if fc_dim:
                self.fc1 = nn.Linear(ll_size, fc_dim)
                self.bn1 = nn.BatchNorm1d(fc_dim)
                self.fc2 = nn.Linear(fc_dim, emb_dim)
            else:
                self.fc1 = nn.Linear(ll_size, emb_dim)
        else:
            if fc_dim:
                self.bn1 = nn.BatchNorm1d(fc_dim)
                self.fc1 = nn.Linear(fc_dim, emb_dim)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.backbone(x)

        if self.pretrained:
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            if self.fc_dim:
                x = F.relu(self.bn1(x))
                x = self.fc2(x)
        elif self.fc_dim:
            x = F.relu(self.bn1(x))
            x = self.fc1(x)

        if self.norm:
            x = F.normalize(x)

        return x


class InceptionEncoder(nn.Module):

    def __init__(self, emb_dim, fc_dim=None, norm=True, pretrained=True, lock=False, transform_input=False):
        super(InceptionEncoder, self).__init__()

        # TODO: fix inception backbone as suggested:
        # https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/49

        self.fc_dim = fc_dim
        self.norm = norm
        self.pretrained = pretrained

        if self.pretrained:
            self.backbone = InceptionBackbone(transform_input=transform_input)

            if lock:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            if fc_dim:
                self.fc1 = nn.Linear(2048, fc_dim)
                self.bn1 = nn.BatchNorm1d(fc_dim)
                self.fc2 = nn.Linear(fc_dim, emb_dim)
            else:
                self.fc1 = nn.Linear(2048, emb_dim)

        else:

            if fc_dim:
                self.backbone = inception_v3(pretrained=False, num_classes=fc_dim, transform_input=transform_input)
                self.bn1 = nn.BatchNorm1d(fc_dim)
                self.fc1 = nn.Linear(fc_dim, emb_dim)
            else:
                self.backbone = inception_v3(pretrained=False, num_classes=emb_dim, transform_input=transform_input)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        if self.pretrained:
            x = self.backbone(x)

            x = self.fc1(x)
            if self.fc_dim:
                x = F.relu(self.bn1(x))
                x = self.fc2(x)
        elif self.fc_dim:
            x = self.backbone(x)
            x = F.relu(self.bn1(x))
            x = self.fc1(x)

        if self.norm:
            x = F.normalize(x)

        return x


class InceptionBackbone(nn.Module):
    def __init__(self, transform_input=False):
        super(InceptionBackbone, self).__init__()

        inception = inception_v3(pretrained=True, transform_input=transform_input)
        self.transform_input = inception.transform_input
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e

        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)

        return x