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
class VariationalDropoutHierarchicalfc(_Linear):
    def __init__(self, in_features, out_features, bias=True, deterministic_test=False, deterministic_compress=False, threshold=3, eps=1e-8):
        super(VariationalDropoutHierarchicalfc,self).__init__(in_features, out_features, bias)
        self.deterministic_test = deterministic_test
        self.deterministic_compress = deterministic_compress
        self.threshold = threshold
        self.eps = eps
        # log_alpha = torch.Tensor(in_features,out_features).fill_(np.log(1.))
        log_alpha = torch.Tensor(out_features).fill_(np.log(1.))
        self.log_alpha = nn.Parameter(log_alpha)
        self.softplus = nn.Softplus()

    @staticmethod
    def compute_log_sigma2(log_alpha, theta, eps=1e-8):
        return log_alpha + torch.log(theta**2 + eps)

    def get_log_alpha(self):
        return self.log_alpha

    def kl(self):
        log_alpha = torch.clamp(self.log_alpha, -8., 8.)
        return -torch.sum(-0.5*self.softplus(-log_alpha))

    @weak_script_method
    def forward(self,input):
        log_alpha = torch.clamp(self.log_alpha, -8., 8.)
        if self.deterministic_test:
            assert self.training == False,"Flag deterministic is True. This should not be used in training."
            if self.deterministic_compress:
                clip_mask = log_alpha > self.threshold # torch.ByteTensor
                weight = self.weight.masked_fill(clip_mask.t(),0)
                bias = self.bias.masked_fill(clip_mask.t(),0)
                return F.linear(input, weight, bias)
            else:
                return F.linear(input, self.weight, self.bias)
        else:
            # log_sigma2 = self.compute_log_sigma2(log_alpha,self.weight.t())
            mu = F.linear(input, self.weight, self.bias)
            # weight = log_sigma2.exp().t()
            # si = torch.sqrt(F.linear(input**2, weight, None) + self.eps)
            si = mu * torch.sqrt(log_alpha.exp() + self.eps)
            eps = torch.randn(*mu.size()).to(input.device)
            assert si.shape == eps.shape
            return mu + eps*si

@weak_module
class VariationalDropoutHierarchicalcnn(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, 
                 deterministic_test=False, deterministic_compress=False, threshold=3, eps=1e-8):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(VariationalDropoutHierarchicalcnn, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')
        self.deterministic_test = deterministic_test
        self.deterministic_compress = deterministic_compress
        self.threshold = threshold
        self.eps = eps        
        # log_alpha = torch.Tensor(in_channels,out_channels,*kernel_size).fill_(np.log(1.))
        log_alpha = torch.Tensor(out_channels).fill_(np.log(1.))
        self.log_alpha = nn.Parameter(log_alpha)
        self.softplus = nn.Softplus()

    @staticmethod
    def compute_log_sigma2(log_alpha, theta, eps=1e-8):
        return log_alpha + torch.log(theta**2 + eps)

    def get_log_alpha(self):
        return self.log_alpha
    
    def kl(self):
        log_alpha = torch.clamp(self.log_alpha, -8., 8.)
        return -torch.sum(-0.5*self.softplus(-log_alpha))

    @weak_script_method
    def forward(self,input):
        log_alpha = torch.clamp(self.log_alpha, -8., 8.)
        if self.deterministic_test:
            assert self.training == False,"Flag deterministic is True. This should not be used in training."
            if self.deterministic_compress:
                clip_mask = log_alpha > self.threshold # torch.ByteTensor
                weight = self.weight.masked_fill(clip_mask.t(),0)
                bias = self.bias.masked_fill(clip_mask.t(),0)
                return F.conv2d(input, weight, bias, self.stride,
                                self.padding, self.dilation, self.groups)
            else:
                return F.conv2d(input, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
        else:
            mu = F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
            # log_sigma2 = self.compute_log_sigma2(log_alpha,self.weight.permute(1,0,2,3).contiguous())
            # weight = log_sigma2.exp().permute(1,0,2,3).contiguous()
            # si = torch.sqrt(F.conv2d(input**2, weight, None, self.stride,
                        # self.padding, self.dilation, self.groups) + self.eps)
            si = mu*torch.sqrt(log_alpha.exp().unsqueeze(-1).unsqueeze(-1) + self.eps)
            eps = torch.randn(*mu.size()).to(input.device)
            assert si.shape == eps.shape
            return mu + eps*si