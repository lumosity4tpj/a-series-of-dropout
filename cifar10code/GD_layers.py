# -*- coding: utf-8 -*-
"""
@author: lumosity
"""

import torch
import torch.nn as nn
import math
import numpy as np
import warnings
from torch._jit_internal import weak_script_method,weak_script,weak_module
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.linear import Linear as _Linear


@weak_module
class GaussianDropoutWfc(_Linear):
    def __init__(self, in_features, out_features, bias=True, p=0.5, deterministic_test=False, eps=1e-8):
        super(GaussianDropoutWfc,self).__init__(in_features, out_features, bias)
        self.deterministic_test = deterministic_test
        self.eps = eps
        self.alpha = torch.Tensor(in_features,out_features).fill_(p/(1.-p))

    @weak_script_method
    def forward(self,input):
        mu = F.linear(input, self.weight, self.bias)
        if self.deterministic_test:
            assert self.training == False,"Flag deterministic is True. This should not be used in training."
            return mu
        else:
            assert self.alpha.t().shape == self.weight.shape
            weight = self.alpha.t().to(self.weight.device) * (self.weight**2)
            si = torch.sqrt(F.linear(input**2, weight, None) + self.eps)
            eps = torch.randn(*mu.size()).to(input.device)
            assert si.shape == eps.shape
            return mu + eps*si

@weak_module
class GaussianDropoutWcnn(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, 
                 p=0.5, deterministic_test=False, eps=1e-8):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GaussianDropoutWcnn, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')
        self.deterministic_test = deterministic_test
        self.eps = eps
        self.alpha = torch.Tensor(in_channels,out_channels,*kernel_size).fill_(p/(1.-p))

    @weak_script_method
    def forward(self,input):
        mu = F.conv2d(input, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
        if self.deterministic_test:
            assert self.training == False,"Flag deterministic is True. This should not be used in training."
            return mu
        else:
            assert self.alpha.permute(1,0,2,3).contiguous().shape == self.weight.shape
            weight = self.alpha.permute(1,0,2,3).contiguous().to(self.weight.device) * (self.weight**2)
            si = torch.sqrt(F.conv2d(input**2, weight, None, self.stride,
                        self.padding, self.dilation, self.groups) + self.eps)
            eps = torch.randn(*mu.size()).to(input.device)
            assert si.shape == eps.shape
            return mu + eps*si


############
@weak_module
class GaussianDropoutSfc(_Linear):
    def __init__(self, in_features, out_features, bias=True, p=0.5, deterministic_test=False, eps=1e-8):
        super(GaussianDropoutSfc,self).__init__(in_features, out_features, bias)
        self.deterministic_test = deterministic_test
        self.eps = eps
        self.alpha = torch.Tensor(in_features).fill_(p/(1.-p))

    @weak_script_method
    def forward(self,input):
        if self.deterministic_test:
            assert self.training == False,"Flag deterministic is True. This should not be used in training."
            return F.linear(input, self.weight, self.bias)
        else:
            si = input * torch.sqrt(self.alpha + self.eps).to(input.device)
            eps = torch.randn(*input.size()).to(input.device)
            assert si.shape == eps.shape
            input_new = input + si * eps
            return F.linear(input_new, self.weight, self.bias)

@weak_module
class GaussianDropoutScnn(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, 
                 p=0.5, deterministic_test=False, eps=1e-8):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GaussianDropoutScnn, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')
        self.deterministic_test = deterministic_test
        self.eps = eps        
        self.alpha = torch.Tensor(in_channels).fill_(p/(1.-p))

    @weak_script_method
    def forward(self,input):
        if self.deterministic_test:
            assert self.training == False,"Flag deterministic is True. This should not be used in training."
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            si = input * torch.sqrt(self.alpha.unsqueeze(-1).unsqueeze(-1) + self.eps).to(input.device)
            eps = torch.randn(*input.size()).to(input.device)
            assert si.shape == eps.shape
            input_new = input + si * eps
            return F.conv2d(input_new, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


class GaussianDropout(nn.Module):
    def __init__(self, in_features, p=0.5, eps=1e-8):
        super(GaussianDropout, self).__init__()
        self.eps = eps
        self.alpha = torch.Tensor(in_features).fill_(p/(1.-p))

    def forward(self, input):
        if not self.training:
            return input
        else:
            si = input * torch.sqrt(self.alpha.unsqueeze(-1).unsqueeze(-1) + self.eps).to(input.device)
            eps = torch.randn(*input.size()).to(input.device)
            assert si.shape == eps.shape
            return input + eps*si