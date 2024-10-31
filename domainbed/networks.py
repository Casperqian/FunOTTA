# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from domainbed.model.SpGraph import VisionTransformer
from domainbed.lib import misc
from domainbed.lib import wide_resnet
from domainbed.lib import big_transfer
from domainbed.lib import vision_transformer
from torchvision.models import (ResNet18_Weights, ResNet50_Weights,
                                EfficientNet_B0_Weights,
                                EfficientNet_B3_Weights,
                                EfficientNet_B4_Weights)


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SqueezeLastTwo(nn.Module):
    """A module which squeezes the last two dimensions, ordinary squeeze can be a problem for batch size 1"""

    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth'] - 2)
        ])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        #self.output = nn.Linear(hparams['mlp_width'], n_outputs)

        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = self.squeezeLastTwo(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['backbone'] == 'resnet18':
            self.network = torchvision.models.resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1)
            self.n_outputs = 512
            self.disable_bn = True
        elif hparams['backbone'] == 'resnet50':
            self.network = torchvision.models.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1)
            self.n_outputs = 2048
            self.disable_bn = True
        elif hparams['backbone'] == 'resnet18-BN':
            self.network = torchvision.models.resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1)
            self.n_outputs = 512
            self.disable_bn = False
        elif hparams['backbone'] == 'resnet50-BN':
            self.network = torchvision.models.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1)
            self.n_outputs = 2048
            self.disable_bn = False
        if self.disable_bn:
            self.network = remove_batch_norm_from_resnet(self.network)

        self.network.fc = Identity()

        if self.disable_bn:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.disable_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class EfficientNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams):
        super(EfficientNet, self).__init__()
        if hparams['backbone'] == 'efficientnet-b0':
            self.network = torchvision.models.efficientnet_b0(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.n_outputs = 1280
            self.disable_bn = False
        elif hparams['backbone'] == 'efficientnet-b4':
            self.network = torchvision.models.efficientnet_b3(
                weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.n_outputs = 1000
            self.disable_bn = False

        if self.disable_bn:
            self.network = remove_batch_norm_from_resnet(self.network)

        self.network.classifier = Identity()

        if self.disable_bn:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.disable_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ViT(torch.nn.Module):

    def __init__(self, input_shape, hparams):
        super(ViT, self).__init__()
        if hparams['backbone'] == 'ViT-S_16':
            self.network = timm.create_model(
                'vit_small_patch16_224',
                pretrained=False,
                checkpoint_path=
                '/home/zengqian/vit_small_patch16.augreg_in1k.bin')
            self.n_outputs = 384

        self.network.head = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)


class Graph(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams):
        super(Graph, self).__init__()
        if hparams['backbone'] == 'SpGraph':
            self.network = VisionTransformer(image_size=224,
                                             patch_size=16,
                                             in_channels=3,
                                             num_classes=1000,
                                             embed_dim=384,
                                             num_heads=6,
                                             depth=12,
                                             emb_dropout=0.,
                                             dropout=0.,
                                             proj_drop=0.,
                                             attn_drop=0.,
                                             drop_path=0.)
            self.n_outputs = 384
            self.disable_bn = False

        if hasattr(self.network, 'head'):
            self.network.head = Identity()

        if self.disable_bn:
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.disable_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif 'resnet' in hparams['backbone']:
        return ResNet(input_shape, hparams)
    elif 'efficientnet' in hparams['backbone']:
        return EfficientNet(input_shape, hparams)
    elif 'Graph' in hparams['backbone']:
        return Graph(input_shape, hparams)
    elif input_shape[1:3] == (224, 224) and 'ViT-' in hparams['backbone']:
        return ViT(input_shape, hparams)
    # elif input_shape[1:3] == (224, 224) and hparams['backbone'] in ['B_16', 'B_32', 'L_16', 'L_32']:
    #     return vision_transformer.ViT(input_shape, hparams)
    # elif input_shape[1:3] == (224, 224) and 'dino' in hparams['backbone']:
    #     return vision_transformer.DINO(input_shape, hparams)
    # elif input_shape[1:3] == (224, 224) and 'DeiT-' in hparams['backbone']:
    #     return ViT(input_shape,hparams)
    # elif input_shape[1:3] == (224, 224) and 'HViT' in hparams['backbone']:
    #     return vision_transformer.HybridViT(input_shape, hparams)
    # elif input_shape[1:3] == (224, 224) and 'Mixer' in hparams['backbone']:
    #     return mlp_mixer.MLPMixer(input_shape, hparams)
    # elif input_shape[1:3] == (224, 224) and 'BiT' in hparams['backbone']:
    #     return big_transfer.BiT(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2), torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(), torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)
