import os
from typing import Union, Optional, List, Iterator
import math
import cupy as cp
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from torch.utils.dlpack import to_dlpack as tens2dlpack

__all__ = ['ParametricLIF', 'mParametricLIF', 'bmParametricLIF']

cuda_threads = 256
cuda_blocks = lambda n: (n + cuda_threads - 1) // cuda_threads
_CURPATH = os.path.abspath(__file__)[:-9]

with open(os.path.join(_CURPATH, 'cuda/parametric_neuron_cupy_kernel.cu'), 'r') as f:
    CU_SOURCE_CODE_RAW_STRING = f.read()


def ten2cpy(ten: torch.Tensor) -> cp.ndarray:
    if hasattr(cp, 'core'):
        return cp.core.dlpack.fromDlpack(tens2dlpack(ten))
    else:
        return cp.from_dlpack(tens2dlpack(ten))

class ParametricNeuronFunc(torch.autograd.Function):
    funclists = ['cudaParametricNeuronForwardKernel<float>', 'cudaParametricNeuronBackwardKernel<float>']
    
    cu_module = cp.RawModule(code=CU_SOURCE_CODE_RAW_STRING,
                             options=('-std=c++11', '-I ' + _CURPATH,),
                             name_expressions=funclists)
    
    neuron_FP = cu_module.get_function(funclists[0])
    neuron_BP = cu_module.get_function(funclists[1])

    @staticmethod
    def forward(ctx, x, tau, th, suro, alpha):
        x = x.contiguous()
        ctx.suro, ctx.alpha = suro, alpha
        ctx.N, ctx.T, ctx.D = x.shape[0], x.shape[1], x[0][0].numel()
        out2, out3 = [torch.zeros_like(x) for i in range(2)]
        with cp.cuda.Device(x.get_device()):
            ParametricNeuronFunc.neuron_FP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(x), ten2cpy(out2), ten2cpy(out3),
                cp.float32(tau[0].item()), cp.float32(th[0].item()),
                cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        ctx.save_for_backward(out2, tau, th)
        
        return out3

    @staticmethod
    def backward(ctx, gradout):
        saved2, tau, th, = ctx.saved_tensors
        gradin1 = torch.zeros_like(gradout)
        gradin2 = torch.zeros_like(gradout[:, 0])
        with cp.cuda.Device(gradout.get_device()):
            ParametricNeuronFunc.neuron_BP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(gradin1), ten2cpy(gradin2), ten2cpy(gradout.contiguous()), ten2cpy(saved2),
                cp.float32(tau[0].item()), cp.float32(th[0].item()),
                cp.int32(ctx.suro), cp.float32(ctx.alpha), cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        dtau = gradin2.sum().view([1])
        return gradin1, dtau, None, None, None

class ParametricLIF(nn.Module):
    def __init__(
        self, 
        tau=2.0,
        th=1.,
        suro=4,
        alpha=lambda : 1.,
        **kwargs
    ) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor([math.log(1. / (tau - 1.))]))
        self.th = nn.Parameter(torch.tensor([th]), requires_grad=True)
        self.suro = suro
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ParametricNeuronFunc.apply(
            x, self.tau.sigmoid(), self.th, self.suro, self.alpha()
        )


class mParametricNeuronFunc(torch.autograd.Function):
    funclists = ['cudamParametricNeuronForwardKernel<float>', 'cudamParametricNeuronBackwardKernel<float>']
    
    cu_module = cp.RawModule(code=CU_SOURCE_CODE_RAW_STRING,
                             options=('-std=c++11', '-I ' + _CURPATH,),
                             name_expressions=funclists)
    
    neuron_FP = cu_module.get_function(funclists[0])
    neuron_BP = cu_module.get_function(funclists[1])

    @staticmethod
    def forward(ctx, x, tau, th, lamb, suro, alpha):
        x = x.contiguous()
        ctx.suro, ctx.alpha = suro, alpha
        ctx.N, ctx.T, ctx.D = x.shape[0], x.shape[1], x[0][0].numel()
        out1, out2, out3 = [torch.zeros_like(x) for i in range(3)]
        with cp.cuda.Device(x.get_device()):
            mParametricNeuronFunc.neuron_FP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(x), ten2cpy(out1), ten2cpy(out2), ten2cpy(out3),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(lamb[0].item()),
                cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        ctx.save_for_backward(out1, out2, tau, th, lamb)

        return out3

    @staticmethod
    def backward(ctx, gradout):
        saved1, saved2, tau, th, lamb = ctx.saved_tensors
        gradin1 = torch.zeros_like(gradout)
        gradin2 = torch.zeros_like(gradout[:, 0])
        gradin3 = torch.zeros_like(gradout[:, 0])
        with cp.cuda.Device(gradout.get_device()):
            mParametricNeuronFunc.neuron_BP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(gradin1), ten2cpy(gradin2), ten2cpy(gradin3), ten2cpy(gradout.contiguous()), ten2cpy(saved1), ten2cpy(saved2),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(lamb[0].item()),
                cp.int32(ctx.suro), cp.float32(ctx.alpha), cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        dtau = gradin3.sum().view([1])
        dlamb = gradin2.sum().view([1])
        return gradin1, dtau, None, dlamb, None, None

class mParametricLIF(nn.Module):
    def __init__(
        self, 
        tau=2.0,
        lamb=0.2,
        th=1.,
        suro=4,
        alpha=lambda : 1.,
        learnable_lb=True,
        **kwargs
    ) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor([math.log(1. / (tau - 1.))]))
        self.th = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=learnable_lb)
        self.suro = suro
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mParametricNeuronFunc.apply(
            x, self.tau.sigmoid(), self.th, self.lamb, self.suro, self.alpha()
        )


class bmParametricNeuronFunc(torch.autograd.Function):
    funclists = ['cudabmParametricNeuronForwardKernel<float>', 'cudabmParametricNeuronBackwardKernel<float>']
    
    cu_module = cp.RawModule(code=CU_SOURCE_CODE_RAW_STRING,
                             options=('-std=c++11', '-I ' + _CURPATH,),
                             name_expressions=funclists)
    
    neuron_FP = cu_module.get_function(funclists[0])
    neuron_BP = cu_module.get_function(funclists[1])

    @staticmethod
    def forward(ctx, x, tau, th, lamb, suro, alpha):
        x = x.contiguous()
        ctx.suro, ctx.alpha = suro, alpha
        ctx.N, ctx.T, ctx.D = x.shape[0], x.shape[1], x[0][0].numel()
        out1, out2, out3 = [torch.zeros_like(x) for i in range(3)]
        with cp.cuda.Device(x.get_device()):
            bmParametricNeuronFunc.neuron_FP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(x), ten2cpy(out1), ten2cpy(out2), ten2cpy(out3),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(lamb[0].item()),
                cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        ctx.save_for_backward(out1, out2, tau, th, lamb)

        return out3

    @staticmethod
    def backward(ctx, gradout):
        saved1, saved2, tau, th, lamb = ctx.saved_tensors
        gradin1 = torch.zeros_like(gradout)
        gradin2 = torch.zeros_like(gradout[:, 0])
        gradin3 = torch.zeros_like(gradout[:, 0])
        with cp.cuda.Device(gradout.get_device()):
            bmParametricNeuronFunc.neuron_BP((cuda_blocks(ctx.N * ctx.D),), (cuda_threads,), (
                ten2cpy(gradin1), ten2cpy(gradin2), ten2cpy(gradin3), ten2cpy(gradout.contiguous()), ten2cpy(saved1), ten2cpy(saved2),
                cp.float32(tau[0].item()), cp.float32(th[0].item()), cp.float32(lamb[0].item()),
                cp.int32(ctx.suro), cp.float32(ctx.alpha), cp.int32(ctx.N), cp.int32(ctx.T), cp.int32(ctx.D)
            ))
        dtau = gradin3.sum().view([1])
        dlamb = gradin2.sum().view([1])
        return gradin1, dtau, None, dlamb, None, None


class bmParametricLIF(nn.Module):
    def __init__(
        self, 
        tau=2.0,
        lamb=0.8,
        th=1.,
        suro=4,
        alpha=lambda : 1.,
        learnable_lb=True,
        **kwargs
    ) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor([math.log(1. / (tau - 1.))]))
        self.th = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=learnable_lb)
        self.suro = suro
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bmParametricNeuronFunc.apply(
            x, self.tau.sigmoid(), self.th, self.lamb, self.suro, self.alpha()
        )