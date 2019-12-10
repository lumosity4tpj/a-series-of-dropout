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
class VariationalDropoutfc(_Linear):
    def __init__(self, in_features, out_features, bias=True, p=0.5, dropout_type='A', deterministic_test=False, deterministic_limit=True, eps=1e-8):
        super(VariationalDropoutfc,self).__init__(in_features, out_features, bias)
        self.deterministic_test = deterministic_test
        self.eps = eps        
        self.dropout_type = dropout_type
        self.deterministic_limit = deterministic_limit
        if dropout_type == 'B':
            # every weight
            log_alpha = torch.Tensor(in_features,out_features).fill_(np.log(p/(1. - p)))
        elif dropout_type == 'A':
            # every hidden layer
            log_alpha = torch.Tensor(in_features).fill_(np.log(p/(1. - p)))
        else:
            raise RuntimeError("Variational type must be 'A' or 'B'.")
        self.log_alpha = nn.Parameter(log_alpha)
        
    def kl(self):
        c1, c2, c3 = 1.16145124, -1.50204118, 0.58629921
        C = -(c1+c2+c3)
        if self.deterministic_limit == True:
            log_alpha = torch.clamp(self.log_alpha.data, -8., 0)
        else:
            log_alpha = self.log_alpha
        alpha = log_alpha.exp()
        return -torch.sum(0.5 * torch.log(alpha) + c1 * alpha + c2 * (alpha**2) + c3 * (alpha**3) + C)

    @weak_script_method
    def forward(self,input):
        if self.deterministic_test:
            assert self.training == False,"Flag deterministic is True. This should not be used in training."
            return F.linear(input, self.weight, self.bias)
        else:
            if self.deterministic_limit == True:
                log_alpha = torch.clamp(self.log_alpha.data, -8., 0)
            else:
                log_alpha = self.log_alpha
            if self.dropout_type == 'B':
                log_sigma2 = log_alpha.to(self.weight.device) + torch.log((self.weight.t())**2 + self.eps)
                mu = F.linear(input, self.weight, self.bias)
                weight = log_sigma2.exp().t()
                si = torch.sqrt(F.linear(input**2, weight, None) + self.eps)
                eps = torch.randn(*mu.size()).to(input.device)
                assert si.shape == eps.shape
                return mu + eps*si
            else:
                si = input * torch.sqrt(log_alpha.exp() + self.eps).to(input.device)
                eps = torch.randn(*input.size()).to(input.device)
                assert si.shape == eps.shape
                input_new = input + si * eps
                return F.linear(input_new, self.weight, self.bias)

@weak_module
class VariationalDropoutcnn(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, 
                 p=0.5, dropout_type='A', deterministic_test=False, deterministic_limit=True, eps=1e-8):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(VariationalDropoutcnn, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')
        self.deterministic_test = deterministic_test
        self.eps = eps        
        self.dropout_type = dropout_type
        self.deterministic_limit = deterministic_limit
        if dropout_type == 'B':
            # every weight
            log_alpha = torch.Tensor(in_channels,out_channels,*kernel_size).fill_(np.log(p/(1. - p)))
        elif dropout_type == 'A':
            # every hidden layer
            log_alpha = torch.Tensor(in_channels).fill_(np.log(p/(1. - p)))
        else:
            raise RuntimeError("Variational type must be 'A' or 'B'.")
        self.log_alpha = nn.Parameter(log_alpha)

    def kl(self):
        c1, c2, c3 = 1.16145124, -1.50204118, 0.58629921
        C = -(c1+c2+c3)
        if self.deterministic_limit == True:
            log_alpha = torch.clamp(self.log_alpha.data, -8., 0)
        else:
            log_alpha = self.log_alpha
        alpha = log_alpha.exp()
        return -torch.sum(0.5 * torch.log(alpha) + c1 * alpha + c2 * (alpha**2) + c3 * (alpha**3) + C)


    @weak_script_method
    def forward(self,input):
        if self.deterministic_test:
            assert self.training == False,"Flag deterministic is True. This should not be used in training."
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            if self.deterministic_limit == True:
                log_alpha = torch.clamp(self.log_alpha.data, -8., 0)
            else:
                log_alpha = self.log_alpha
            if self.dropout_type == 'B':
                log_sigma2 = log_alpha.to(self.weight.device) + torch.log((self.weight.permute(1,0,2,3).contiguous())**2 + self.eps)
                mu = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
                weight = log_sigma2.exp().permute(1,0,2,3).contiguous()
                si = torch.sqrt(F.conv2d(input**2, weight, None, self.stride,
                            self.padding, self.dilation, self.groups) + self.eps)
                eps = torch.randn(*mu.size()).to(input.device)
                assert si.shape == eps.shape
                return mu + eps*si
            else:
                si = input * torch.sqrt(log_alpha.exp().unsqueeze(-1).unsqueeze(-1) + self.eps).to(input.device)
                eps = torch.randn(*input.size()).to(input.device)
                assert si.shape == eps.shape
                input_new = input + si * eps
                return F.conv2d(input_new, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)