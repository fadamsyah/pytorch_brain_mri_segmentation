import torch

def UNet(in_channels=3, out_channels=1, init_features=32, pretrained=False):
    return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                          in_channels=in_channels, out_channels=out_channels,
                          init_features=init_features, pretrained=pretrained)