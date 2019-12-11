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
class VariationalDropout(nn.Module):
    def __init__(self, in_features, p=0.5, deterministic_test=False, deterministic_limit=True, deterministic_sparse=False, eps=1e-8, fc=False):
        super(VariationalDropout, self).__init__()
        self.deterministic_test = deterministic_test
        self.eps = eps        
        self.deterministic_limit = deterministic_limit
        self.deterministic_sparse = deterministic_sparse
        self.fc = fc
        log_alpha = torch.Tensor(in_features).fill_(np.log(p/(1. - p)))

        self.log_alpha = nn.Parameter(log_alpha)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def kl(self):
        if self.deterministic_limit == True:
            c1, c2, c3 = 1.16145124, -1.50204118, 0.58629921
            C = -(c1+c2+c3)
            if self.deterministic_limit == True:
                log_alpha = torch.clamp(self.log_alpha, -8., 0)
            else:
                log_alpha = self.log_alpha
            alpha = log_alpha.exp()
            return -torch.sum(0.5 * torch.log(alpha) + c1 * alpha + c2 * (alpha**2) + c3 * (alpha**3) + C)
        else:
            if self.deterministic_sparse == True:
                k1, k2, k3 = 0.63576, 1.8732, 1.48695
                return -torch.sum(k1 * self.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * self.softplus(-self.log_alpha) - k1)

            else:
                return -torch.sum(-0.5*self.softplus(-self.log_alpha))

    @weak_script_method
    def forward(self,input):
        if self.deterministic_test:
            assert self.training == False,"Flag deterministic is True. This should not be used in training."
            return input
        else:
            if self.deterministic_limit == True:
                log_alpha = torch.clamp(self.log_alpha, -8., 0) ############Todo:好像截断了，导致梯度不能传递？
            else:
                log_alpha = self.log_alpha
            if self.fc == True:
                si = input * torch.sqrt(log_alpha.exp() + self.eps).to(input.device)
            else:
                si = input * torch.sqrt(log_alpha.exp().unsqueeze(-1).unsqueeze(-1) + self.eps).to(input.device)
            eps = torch.randn(*input.size()).to(input.device)
            assert si.shape == eps.shape
            return input + eps*si


class GaussianDropout(nn.Module):
    def __init__(self, in_features, p=0.5, deterministic_test=False, eps=1e-8, fc=False):
        super(GaussianDropout, self).__init__()
        self.deterministic_test = deterministic_test
        self.eps = eps        
        self.fc = fc
        self.alpha = torch.Tensor(in_features).fill_(p/(1.-p))

    def forward(self, input):
        if self.deterministic_test:
            assert self.training == False,"Flag deterministic is True. This should not be used in training."
            return input
        else:
            if self.fc == True:
                si = input * torch.sqrt(self.alpha + self.eps).to(input.device)
            else:
                si = input * torch.sqrt(self.alpha.unsqueeze(-1).unsqueeze(-1) + self.eps).to(input.device)
            eps = torch.randn(*input.size()).to(input.device)
            assert si.shape == eps.shape
            return input + eps*si
