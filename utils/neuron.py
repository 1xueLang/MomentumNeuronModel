import math

from typing import Iterator
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from snetx.snn.algorithm import PiecewiseQuadratic, arc_tan

__all__ = ['MomentumLIF', 'bMomentumLIF']


class bMomentumLIF(nn.Module):
    r""" STBP based balanced momentum neuron model(bMomentumLIF).
    
    Args:
        tau (float): Exponential attenuation coefficient of the membrane potential in LIF model. Default: 2.0.
    """
    def __init__(
        self, 
        tau=2.0,
        th=1.,
        momentum=0.,
        lamb=1.,
        learnable_mt=False,
        learnable_lb=False,
        regular=False,
        sg=PiecewiseQuadratic.apply,
        alpha=lambda : 1.0,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.thresholds = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.momentum = nn.Parameter(torch.tensor([momentum]), requires_grad=learnable_mt)
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=learnable_lb)
        self.regular = regular
        self.spiking = sg
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # momentum = self.momentum.sigmoid() if self.regular else self.momentum
        lamb = self.lamb.sigmoid() if self.regular else self.lamb
        out = torch.zeros_like(x)
        u = 0.
        u_last = 0.
        m = 0.
        s = 0.
        for i in range(x.shape[1]):
            u = u_last / self.tau + x[:, i]
            s = self.spiking(u - self.thresholds, self.alpha())
            m = self.momentum * m + (1. - self.momentum) * (u - u_last)
            u_last = (u * lamb + m * (1. - lamb)) * (1. - s)
            out[:, i] = s
        
        return out

    
class MomentumLIF(nn.Module):
    r""" STBP based momentum neuron model(MomentumLIF).
    
    Args:
        tau (float): Exponential attenuation coefficient of the membrane potential in LIF model. Default: 2.0.
    """
    def __init__(
        self, 
        tau=2.0,
        th=1.,
        momentum=0.,
        lamb=0.,
        learnable_mt=False,
        learnable_lb=False,
        reset=False,
        regular=False,
        sg=PiecewiseQuadratic.apply,
        alpha=lambda : 1.0,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.thresholds = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.momentum = nn.Parameter(torch.tensor([momentum]), requires_grad=learnable_mt)
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=learnable_lb)
        self.reset = reset
        self.regular = regular
        self.spiking = sg
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # momentum = self.momentum.sigmoid() if self.regular else self.momentum
        lamb = self.lamb.sigmoid() if self.regular else self.lamb
        out = torch.zeros_like(x)
        u = 0.
        u_last = torch.tensor([0.]).to(x)
        m = 0.
        s = 0.
        for i in range(x.shape[1]):
            u = u_last / self.tau + x[:, i]
            s = self.spiking(u - self.thresholds, self.alpha())
            m = self.momentum * m + (1. - self.momentum) * (u - u_last)
            u_last = (u + m * lamb) * (1. - s)
            out[:, i] = s
        
        return out

class ParametricLIF(nn.Module):
    def __init__(
        self, 
        tau=2.0,
        th=1.,
        sg=PiecewiseQuadratic.apply,
        alpha=lambda : 1.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor([math.log(1. / (tau - 1.))]))
        self.thresholds = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.spiking = sg
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        u = 0.
        s = 0.
        for i in range(x.shape[1]):
            u = u * self.tau.sigmoid() + x[:, i]
            s = self.spiking(u - self.thresholds, self.alpha())
            u = u * (1. - s)
            out[:, i] = s
        
        return out

class mParametricLIF(nn.Module):
    def __init__(
        self, 
        tau=2.0,
        lamb=0.2,
        th=1.,
        sg=PiecewiseQuadratic.apply,
        alpha=lambda : 1.,
        learnable_lb=True,
        **kwargs
    ) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor([math.log(1. / (tau - 1.))]))
        self.th = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=learnable_lb)
        self.spiking = sg
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        u = 0.
        last = 0.
        s = 0.
        for i in range(x.shape[1]):
            u = last * self.tau.sigmoid() + x[:, i]
            s = self.spiking(u - self.th, self.alpha())
            last = (u + self.lamb * (u - last)) * (1. - s)
            out[:, i] = s
        
        return out

class bmParametricLIF(nn.Module):
    def __init__(
        self, 
        tau=2.0,
        lamb=0.8,
        th=1.,
        sg=PiecewiseQuadratic.apply,
        alpha=lambda : 1.,
        learnable_lb=True,
        **kwargs
    ) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor([math.log(1. / (tau - 1.))]))
        self.th = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=learnable_lb)
        self.spiking = sg
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        u = 0.
        last = 0.
        s = 0.
        for i in range(x.shape[1]):
            u = last * self.tau.sigmoid() + x[:, i]
            s = self.spiking(u - self.th, self.alpha())
            last = (u * self.lamb + (1. - self.lamb) * (u - last)) * (1. - s)
            out[:, i] = s
        
        return out