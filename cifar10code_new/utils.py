#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import torch
import torch.nn as nn
import math
from torch._jit_internal import weak_script_method,weak_script,weak_module
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

def setup_seed(seed):
    torch.manual_seed(seed)#cpu
    torch.cuda.manual_seed(seed)#gpu
    np.random.seed(seed)#numpy
    random.seed(seed)#random and transforms
    torch.backends.cudnn.deterministic = True#cudnn

def weights_init(m):
    # for m in x.modules():
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #         if m.bias is not None:
    #             m.bias.data.zero_()
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.weight.data.fill_(1)
    #         m.bias.data.zero_()
    #     elif isinstance(m, nn.Linear):
    #         m.weight.data.normal_(0, 0.01)
    #         m.bias.data.zero_()
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
        # nn.init.xavier_normal_(m.weight.data,gain=1)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)
    elif classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0,0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)   