from typing import Any, Callable, List, Optional, Dict, TypeVar

from torch import Tensor

import torch
import torch.nn as nn
from snetx.snn import algorithm as snnalgo
from snetx.models import _shortcuts
from snetx.models._shortcuts import BasicBlock_A, BasicBlock_B, BasicBlock_C
from snetx.models._shortcuts import Bottleneck_A, Bottleneck_B, Bottleneck_C
from snetx.models._shortcuts import conv1x1, conv1x1_norm
from snetx import models


__all__ = [
    "VGG9",
    "msresnet18",
    "sewresnet18",
    "msresnet19",
    "sewresnet19",
]

class VGG9(nn.Module):
    def __init__(self, neuron, neuron_cfg, in_channels=2, norm_layer=None):
        super(VGG9, self).__init__()
        pool = snnalgo.Tosnn(nn.AvgPool2d(2))
        # self.voting = snnalgo.Tosnn(nn.AvgPool1d(10, 10))
        self.voting = nn.AvgPool1d(10, 10)
        self.features = nn.Sequential(
            _shortcuts.conv3x3_norm(in_channels, 64, norm_layer=norm_layer),  
            neuron(**neuron_cfg),
            _shortcuts.conv3x3_norm(64, 64, norm_layer=norm_layer),
            neuron(**neuron_cfg),
            pool,
            _shortcuts.conv3x3_norm(64, 128, norm_layer=norm_layer),
            neuron(**neuron_cfg),
            _shortcuts.conv3x3_norm(128, 128, norm_layer=norm_layer),
            neuron(**neuron_cfg),
            pool,
            _shortcuts.conv3x3_norm(128, 256, norm_layer=norm_layer),
            neuron(**neuron_cfg),
            _shortcuts.conv3x3_norm(256, 256, norm_layer=norm_layer),
            neuron(**neuron_cfg),
            _shortcuts.conv3x3_norm(256, 256, norm_layer=norm_layer),
            neuron(**neuron_cfg),
            pool,
        )
        self.fc1 =  nn.Sequential(nn.Dropout(0.25), nn.Linear(256 * 6 * 6, 1024))
        self.fc2 =  nn.Sequential(nn.Dropout(0.25), nn.Linear(1024, 100), self.voting)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.fc2(self.fc1(x))
        return x


BasicBlocks = {
    'A': BasicBlock_A,
    'B': BasicBlock_B,
    'C': BasicBlock_C
}

Bottlenecks = {
    'A': Bottleneck_A,
    'B': Bottleneck_B,
    'C': Bottleneck_C,
}

class ResNet1(nn.Module):
    def __init__(
        self,
        neuron,
        n_config,
        block,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        type: str = 'A',
        feature: Callable = models.ms_resnet.classic_feature
    ) -> None:
        super().__init__()
        self.type = type
        if norm_layer is None:
            norm_layer = snnalgo.tdBN2d
            # norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.feature = feature(self.type, norm_layer, self.inplanes, neuron, n_config)
        self.layer1 = self._make_layer(neuron, n_config, block, 128, layers[0])
        self.layer2 = self._make_layer(neuron, n_config, block, 256, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(neuron, n_config, block, 512, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        
        out_layers = []
        if self.type == 'A':
            out_layers.extend([neuron(**n_config), snnalgo.Tosnn(nn.AdaptiveAvgPool2d((1, 1)))])
        elif self.type == 'B':
            out_layers.extend([norm_layer(512 * block.expansion), neuron(**n_config), snnalgo.Tosnn(nn.AdaptiveAvgPool2d((1, 1)))])
        else:
            out_layers.append(snnalgo.Tosnn(nn.AdaptiveAvgPool2d((1, 1))))
            
        self.out_layer = nn.Sequential(*out_layers)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if (isinstance(m, Bottleneck_A) or isinstance(m, Bottleneck_C)) and m.conv3[-1].weight is not None:
                    nn.init.constant_(m.conv3[-1].weight, 0)  # type: ignore[arg-type]
                elif (isinstance(m, BasicBlock_A) or isinstance(m, BasicBlock_C)) and m.conv2[-1].weight is not None:
                    nn.init.constant_(m.conv2[-1].weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        neuron,
        n_config,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.type == 'A':
                downsample = nn.Sequential(
                    neuron(**n_config),
                    conv1x1_norm(self.inplanes, planes * block.expansion, stride, norm_layer=norm_layer)
                )
            elif self.type == 'B':
                if norm_layer:
                    downsample = [snnalgo.Tosnn(norm_layer(self.inplanes))]
                else:
                    downsample = []
                downsample += [neuron(**n_config), conv1x1(self.inplanes, planes * block.expansion, stride)]
                downsample = nn.Sequential(*downsample)
            elif self.type == 'C':
                downsample = conv1x1_norm(self.inplanes, planes * block.expansion, stride, norm_layer=norm_layer)
            else:
                raise ValueError('')

        layers = []
        layers.append(
            block(
                neuron, n_config, self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    neuron,
                    n_config,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.feature(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.out_layer(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet1(
    type,
    neuron,
    n_config,
    block,
    layers: List[int],
    **kwargs: Any,
) -> ResNet1:

    model = ResNet1(neuron, n_config, block, layers, type=type, **kwargs)

    return model


def msresnet18(type, neuron, n_config, **kwargs: Any) -> ResNet1:
    return _resnet1(type, neuron, n_config, BasicBlocks[type], [3, 3, 2], **kwargs)

class MsResNet19(nn.Module):
    def __init__(self, type, neuron, n_config, drop=0.2, **kwargs):
        super().__init__()
        num_classes = kwargs.get('num_classes', 1000)
        kwargs['num_classes'] = 1000
        self.resnet = msresnet18(type, neuron, n_config, **kwargs)
        self.fc = nn.Sequential(
            # neuron(**n_config),
            nn.Dropout(p=drop),
            nn.Linear(kwargs['num_classes'], num_classes)
        )
        
    def forward(self, x):
        return self.fc(self.resnet(x))
    
def msresnet19(type, neuron, n_config, dropout=0.2, **kwargs):
    return MsResNet19(type, neuron, n_config, dropout, **kwargs)


#############


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def _conv3x3_norm(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, norm_layer=None
    ) -> nn.Sequential:
    """3x3 convolution with padding"""
    layers = [snnalgo.Tosnn(conv3x3(in_planes, out_planes, stride, groups, dilation))]
    if norm_layer:
        layers.append(norm_layer(out_planes))
    
    return nn.Sequential(*layers)

def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def _conv1x1_norm(in_planes: int, out_planes: int, stride: int = 1, norm_layer=None):
    """1x1 convolution"""
    layers = [snnalgo.Tosnn(conv1x1(in_planes, out_planes, stride=stride))]
    if norm_layer:
        layers.append(norm_layer(out_planes))
    
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        neuron,
        n_config,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = snnalgo.tdBN2d
            # norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _conv3x3_norm(inplanes, planes, stride, norm_layer=norm_layer)
        self.sn1 = neuron(**n_config)
        self.conv2 = _conv3x3_norm(planes, planes, norm_layer=norm_layer)
        self.sn2 = neuron(**n_config)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
            
        out = self.conv1(x)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.sn2(out)
        
        if self.downsample:
            identity = self.downsample(identity)
        
        out += identity

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        neuron,
        n_config,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = snnalgo.tdBN2d
            # norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _conv1x1_norm(inplanes, width, norm_layer=norm_layer)
        self.sn1 = neuron(**n_config)
        self.conv2 = _conv3x3_norm(width, width, stride, groups, dilation, norm_layer=norm_layer)
        self.sn2 = neuron(**n_config)
        self.conv3 = _conv1x1_norm(width, planes * self.expansion, norm_layer=norm_layer)
        self.sn3 = neuron(**n_config)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.sn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        neuron,
        n_config,
        block,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        feature: Callable = models.sew_resnet.classic_feature
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = snnalgo.tdBN2d
            # norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.feature = feature(norm_layer, self.inplanes, neuron, n_config)
        self.layer1 = self._make_layer(neuron, n_config, block, 128, layers[0])
        self.layer2 = self._make_layer(neuron, n_config, block, 256, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(neuron, n_config, block, 512, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.avgpool = snnalgo.Tosnn(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     if m.affine:
            #         nn.init.constant_(m.weight, 1)
                    # nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.conv2[-1].weight is not None:
                    nn.init.constant_(m.conv2[-1].weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.conv3[-1].weight is not None:
                    nn.init.constant_(m.conv3[-1].weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        neuron,
        n_config,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    _conv1x1_norm(self.inplanes, planes * block.expansion, stride, norm_layer=norm_layer),
                    neuron(**n_config)
                )

        layers = []
        layers.append(
            block(
                neuron, n_config, self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    neuron,
                    n_config,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.feature(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    neuron,
    n_config,
    block,
    layers: List[int],
    **kwargs: Any,
) -> ResNet:

    model = ResNet(neuron, n_config, block, layers, **kwargs)

    return model


def sewresnet18(neuron, n_config, **kwargs: Any) -> ResNet:
    return _resnet(neuron, n_config, BasicBlock, [3, 3, 2], **kwargs)


class SewResNet19(nn.Module):
    def __init__(self, neuron, n_config, drop=0.2, **kwargs):
        super().__init__()
        num_classes = kwargs.get('num_classes', 1000)
        kwargs['num_classes'] = 1000
        self.resnet = sewresnet18(neuron, n_config, **kwargs)
        self.fc = nn.Sequential(
            # neuron(**n_config),
            nn.Dropout(p=drop),
            nn.Linear(kwargs['num_classes'], num_classes)
        )
        
    def forward(self, x):
        return self.fc(self.resnet(x))
    
def sewresnet19(neuron, n_config, dropout=0.2, **kwargs):
    return SewResNet19(neuron, n_config, dropout, **kwargs)