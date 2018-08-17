from models.definitions import MNISTEncoder, ResNetEncoder, InceptionEncoder


def load_net(net_name):
    if net_name == 'mnist_default':
        net = MNISTEncoder(emb_dim=2)
    elif net_name == 'resnet18_e512':
        net = ResNetEncoder(type=18, emb_dim=512, fc_dim=None, norm=False)
    elif net_name == 'resnet18_e512_fc512':
        net = ResNetEncoder(type=18, emb_dim=512, fc_dim=512, norm=False)
    elif net_name == 'resnet18_e1024_fc1024':
        net = ResNetEncoder(type=18, emb_dim=1024, fc_dim=1024, norm=False)
    elif net_name == 'resnet18_e1024_fc1024_norm':
        net = ResNetEncoder(type=18, emb_dim=1024, fc_dim=1024, norm=True)
    elif net_name == 'resnet18_e1024_pt':
        net = ResNetEncoder(type=18, emb_dim=1024, norm=False, pretrained=True)
    elif net_name == 'resnet18_e1024_fc2048_norm_pt':
        net = ResNetEncoder(type=18, emb_dim=1024, fc_dim=2048, norm=True, pretrained=True)
    elif net_name == 'resnet18_e1024_fc2048_norm':
        net = ResNetEncoder(type=18, emb_dim=1024, fc_dim=2048, norm=True, pretrained=False)
    elif net_name == 'resnet50_e1024_fc2048_norm_pt':
        net = ResNetEncoder(type=50, emb_dim=1024, fc_dim=2048, norm=True, pretrained=True)

    elif net_name == 'resnet50_e512':
        net = ResNetEncoder(type=50, emb_dim=512, fc_dim=None, norm=False)
    elif net_name == 'resnet50_e512_fc512':
        net = ResNetEncoder(type=50, emb_dim=512, fc_dim=512, norm=False)
    elif net_name == 'resnet50_e1024_fc1024':
        net = ResNetEncoder(type=50, emb_dim=1024, fc_dim=1024, norm=False)
    elif net_name == 'resnet50_e1024_fc1024_norm':
        net = ResNetEncoder(type=50, emb_dim=1024, fc_dim=1024, norm=True)
    elif net_name == 'inceptionv3_e1024_pt':
        net = InceptionEncoder(emb_dim=1024, norm=False, pretrained=True)
    elif net_name == 'inceptionv3_fc2048_e1024_pt_l':
        net = InceptionEncoder(emb_dim=1024, fc_dim=2048, norm=True, pretrained=True, lock=True)
    elif net_name == 'inceptionv3_fc2048_e1024_pt_ul':
        net = InceptionEncoder(emb_dim=1024, fc_dim=2048, norm=True, pretrained=True, lock=True)
    else:
        return None
    return net
