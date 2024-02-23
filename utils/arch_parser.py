from snetx.models import sew_resnet
from snetx.models import ms_resnet
from snetx.models import vgg
from . import myModels
from . import myVGG

def get_network(cmd_args, neuron, neuron_cfg, norm_layer=None):
    module, network = cmd_args.arch.split('+')
    args = ('A', neuron, neuron_cfg) if 'ms' in cmd_args.arch else (neuron, neuron_cfg)
    kwargs = dict(norm_layer=norm_layer, num_classes=cmd_args.num_classes)
    if 'DVS' not in cmd_args.dataset:
        if 'ms' in cmd_args.arch:
            kwargs['feature'] = ms_resnet.cifar10_feature
        elif 'sew' in cmd_args.arch:
            kwargs['feature'] = sew_resnet.cifar10_feature
        else:
            kwargs['in_channels'] = 3
    else:
        if 'ms' in cmd_args.arch:
            kwargs['feature'] = ms_resnet.cifar10dvs_feature
        elif 'sew' in cmd_args.arch:
            kwargs['feature'] = sew_resnet.cifar10dvs_feature
        else:
            kwargs['in_channels'] = 2
    if 'vgg' in network or '19' in network:
        kwargs['dropout'] = cmd_args.drop

    if module == 'sew_resnet':
        return sew_resnet.__dict__[network](*args, **kwargs)
    elif module == 'ms_resnet':
        return ms_resnet.__dict__[network](*args, **kwargs)
    elif module == 'vgg':
        return vgg.__dict__[network](*args, **kwargs)
    elif module == 'myModels':
        return myModels.__dict__[network](*args, **kwargs)
    elif module == 'myVGG':
        return myVGG.__dict__[network](*args, **kwargs)
    else:
        raise NotImplementedError(cmd_args.arch)