import os
from typing import Union, Optional, List, Iterator

import cupy as cp
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from torch.utils.dlpack import to_dlpack as tens2dlpack

cuda_threads = 256
cuda_blocks = lambda n: (n + cuda_threads - 1) // cuda_threads
_CURPATH = os.path.abspath(__file__)[:-9]

with open(os.path.join(_CURPATH, 'cuda/momentum_neuron_cupy_kernel.cu'), 'r') as f:
    CU_SOURCE_CODE_RAW_STRING = f.read()


def ten2cpy(ten: torch.Tensor) -> cp.ndarray:
    if hasattr(cp, 'core'):
        return cp.core.dlpack.fromDlpack(tens2dlpack(ten))
    else:
        return cp.from_dlpack(tens2dlpack(ten))

###### learnable_lb, mt
class MomentumNeuronFunc0(torch.autograd.Function):
    funclists = ['cudaMomentumNeuronForwardKernel0<float>', 'cudaMomentumNeuronBackwardKernel0<float>']
    
    cu_module = cp.RawModule(code=CU_SOURCE_CODE_RAW_STRING,
                             options=('-std=c++11', '-I ' + _CURPATH,),
                             name_expressions=funclists)
    
    neuron_FP = cu_module.get_function(funclists[0])
    neuron_BP = cu_module.get_function(funclists[1])
    
    @staticmethod
    def forward(ctx, x, tau, th, momentum, lamb, suro, alpha):
        x = x.contiguous()
        ctx.suro, ctx.alpha = suro, alpha
        ctx.N, ctx.T, ctx.D = x.shape[0], x.shape[1], x[0][0].numel()
        out1, out2, out3, out4 = [torch.zeros_like(x) for i in range(4)]
        with cp.cuda.Device(x.get_device()):
            MomentumNeuronFunc0.neuron_FP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(x), ten2cpy(out1), ten2cpy(out2), ten2cpy(out3), ten2cpy(out4),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(momentum[0].item()), cp.float32(lamb[0].item()), 
                cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        ctx.save_for_backward(out1, out2, out3, tau, th, momentum, lamb)
        
        return out4
        
    @staticmethod
    def backward(ctx, gradout):
        saved1, saved2, saved3, tau, th, momentum, lamb = ctx.saved_tensors
        gradin1 = torch.zeros_like(gradout)
        gradin2, gradin3 = [torch.zeros_like(gradout[:, 0]) for i in range(2)]
        with cp.cuda.Device(gradout.get_device()):
            MomentumNeuronFunc0.neuron_BP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(gradin1), ten2cpy(gradin2), ten2cpy(gradin3), ten2cpy(gradout.contiguous()), ten2cpy(saved1), ten2cpy(saved2), ten2cpy(saved3),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(momentum[0].item()), cp.float32(lamb[0].item()), 
                cp.int32(ctx.suro), cp.float32(ctx.alpha), cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        
        dlamb = gradin2.sum().view([1])
        dm = gradin3.sum().view([1])
        return gradin1, None, None, dm, dlamb, None, None  


###### learnable_lb
class MomentumNeuronFunc1(torch.autograd.Function):
    funclists = ['cudaMomentumNeuronForwardKernel1<float>', 'cudaMomentumNeuronBackwardKernel1<float>']
    
    cu_module = cp.RawModule(code=CU_SOURCE_CODE_RAW_STRING,
                             options=('-std=c++11', '-I ' + _CURPATH,),
                             name_expressions=funclists)
    
    neuron_FP = cu_module.get_function(funclists[0])
    neuron_BP = cu_module.get_function(funclists[1])
    
    @staticmethod
    def forward(ctx, x, tau, th, momentum, lamb, suro, alpha):
        x = x.contiguous()
        ctx.suro, ctx.alpha = suro, alpha
        ctx.N, ctx.T, ctx.D = x.shape[0], x.shape[1], x[0][0].numel()
        out1, out2, out3 = [torch.zeros_like(x) for i in range(3)]
        with cp.cuda.Device(x.get_device()):
            MomentumNeuronFunc1.neuron_FP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(x), ten2cpy(out1), ten2cpy(out2), ten2cpy(out3),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(momentum[0].item()), cp.float32(lamb[0].item()), 
                cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        ctx.save_for_backward(out1, out2, tau, th, momentum, lamb)
        
        return out3
        
    @staticmethod
    def backward(ctx, gradout):
        saved1, saved2, tau, th, momentum, lamb = ctx.saved_tensors
        gradin1 = torch.zeros_like(gradout) 
        gradin2 = torch.zeros_like(gradout[:, 0])
        with cp.cuda.Device(gradout.get_device()):
            MomentumNeuronFunc1.neuron_BP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(gradin1), ten2cpy(gradin2), ten2cpy(gradout.contiguous()), ten2cpy(saved1), ten2cpy(saved2),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(momentum[0].item()), cp.float32(lamb[0].item()), 
                cp.int32(ctx.suro), cp.float32(ctx.alpha), cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        
        dlamb = gradin2.sum().view([1])
        return gradin1, None, None, None, dlamb, None, None  

   
###### learnable_mt
class MomentumNeuronFunc2(torch.autograd.Function):
    funclists = ['cudaMomentumNeuronForwardKernel2<float>', 'cudaMomentumNeuronBackwardKernel2<float>']
    
    cu_module = cp.RawModule(code=CU_SOURCE_CODE_RAW_STRING,
                             options=('-std=c++11', '-I ' + _CURPATH,),
                             name_expressions=funclists)
    
    neuron_FP = cu_module.get_function(funclists[0])
    neuron_BP = cu_module.get_function(funclists[1])
    
    @staticmethod
    def forward(ctx, x, tau, th, momentum, lamb, suro, alpha):
        x = x.contiguous()
        ctx.suro, ctx.alpha = suro, alpha
        ctx.N, ctx.T, ctx.D = x.shape[0], x.shape[1], x[0][0].numel()
        out1, out2, out3, out4 = [torch.zeros_like(x) for i in range(4)]
        with cp.cuda.Device(x.get_device()):
            MomentumNeuronFunc2.neuron_FP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(x), ten2cpy(out1), ten2cpy(out2), ten2cpy(out3), ten2cpy(out4),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(momentum[0].item()), cp.float32(lamb[0].item()), 
                cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        ctx.save_for_backward(out1, out2, out3, tau, th, momentum, lamb)
        
        return out4
        
    @staticmethod
    def backward(ctx, gradout):
        saved1, saved2, saved3, tau, th, momentum, lamb = ctx.saved_tensors
        gradin1 = torch.zeros_like(gradout) 
        gradin2 = torch.zeros_like(gradout[:, 0])
        with cp.cuda.Device(gradout.get_device()):
            MomentumNeuronFunc2.neuron_BP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(gradin1), ten2cpy(gradin2), ten2cpy(gradout.contiguous()), ten2cpy(saved1), ten2cpy(saved2), ten2cpy(saved3),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(momentum[0].item()), cp.float32(lamb[0].item()), 
                cp.int32(ctx.suro), cp.float32(ctx.alpha), cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        
        dm = gradin2.sum().view([1])
        return gradin1, None, None, dm, None, None, None  



###### unlearnable
class MomentumNeuronFunc3(torch.autograd.Function):
    funclists = ['cudaMomentumNeuronForwardKernel3<float>', 'cudaMomentumNeuronBackwardKernel3<float>']
    
    cu_module = cp.RawModule(code=CU_SOURCE_CODE_RAW_STRING,
                             options=('-std=c++11', '-I ' + _CURPATH,),
                             name_expressions=funclists)
    
    neuron_FP = cu_module.get_function(funclists[0])
    neuron_BP = cu_module.get_function(funclists[1])
    
    @staticmethod
    def forward(ctx, x, tau, th, momentum, lamb, suro, alpha):
        x = x.contiguous()
        ctx.suro, ctx.alpha = suro, alpha
        ctx.N, ctx.T, ctx.D = x.shape[0], x.shape[1], x[0][0].numel()
        out1, out2, out3 = [torch.zeros_like(x) for i in range(3)]
        with cp.cuda.Device(x.get_device()):
            MomentumNeuronFunc3.neuron_FP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(x), ten2cpy(out1), ten2cpy(out2), ten2cpy(out3),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(momentum[0].item()), cp.float32(lamb[0].item()), 
                cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        ctx.save_for_backward(out1, out2, tau, th, momentum, lamb)
        
        return out3
        
    @staticmethod
    def backward(ctx, gradout):
        saved1, saved2, tau, th, momentum, lamb = ctx.saved_tensors
        gradin1 = torch.zeros_like(gradout)
        with cp.cuda.Device(gradout.get_device()):
            MomentumNeuronFunc3.neuron_BP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(gradin1), ten2cpy(gradout.contiguous()), ten2cpy(saved1), ten2cpy(saved2),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(momentum[0].item()), cp.float32(lamb[0].item()), 
                cp.int32(ctx.suro), cp.float32(ctx.alpha), cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        
        return gradin1, None, None, None, None, None, None  



class MomentumLIF(nn.Module):
    r""" STBP based momentum neuron model(MomentumLIF).
    
    Args:
        tau (float): Exponential attenuation coefficient of the membrane potential in LIF model. Default: 2.0.
    """
    def __init__(
        self, 
        tau=2., 
        th=1.,
        momentum=0.,
        lamb=0.,
        learnable_mt=False,
        learnable_lb=False,
        regular=False,
        suro=4,
        alpha=lambda : 1.,
    ) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor([tau]), requires_grad=False)
        self.th = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.momentum = nn.Parameter(torch.tensor([momentum]), requires_grad=learnable_mt)
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=learnable_lb)
        self.regular = regular
        self.suro, self.alpha = suro, alpha
        self.momentum_neuron_func = self.analysis_momentum_func()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # momentum = self.momentum.sigmoid() if self.regular else self.momentum
        lamb = self.lamb.sigmoid() if self.regular else self.lamb

        return self.momentum_neuron_func.apply(
            x, self.tau, self.th, self.momentum, lamb, self.suro, self.alpha()
        )

    def analysis_momentum_func(self):
        MNFunc = {
            0: MomentumNeuronFunc0,
            1: MomentumNeuronFunc1,
            2: MomentumNeuronFunc2,
            3: MomentumNeuronFunc3,
        }
        return MNFunc[int(not self.lamb.requires_grad) * 2 + int(not self.momentum.requires_grad)]