# -*- coding: utf-8 -*-
"""
@author: lumosity
"""
import torch
import torch.nn as nn
import math
import numpy as np
import warnings
import torch.nn.functional as F
from VD_new_layers import VariationalDropout,GaussianDropout

#########################################################################################################################
class NETGaussianDropoutWang(nn.Module):
    def __init__(self,scale=1.0):
        super(NETGaussianDropoutWang,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            nn.Conv2d(3,int(32*scale),kernel_size=3,stride=2),
            GaussianDropout(int(32*scale),p=0.5,deterministic_test=(self.training!=True),fc=False),
            nn.Softplus(),
            nn.Conv2d(int(32*scale),int(64*scale),kernel_size=3,stride=2),
            GaussianDropout(int(64*scale),p=0.5,deterministic_test=(self.training!=True),fc=False),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            nn.Linear(int(64*scale)*7*7,int(128*scale)),
            GaussianDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),fc=True),
            nn.Linear(int(128*scale),int(128*scale)),
            GaussianDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),fc=True),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

class NETGaussianDropoutSrivastava(nn.Module):
    def __init__(self,scale=1.0):
        super(NETGaussianDropoutSrivastava,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            GaussianDropout(3,p=0.5,deterministic_test=(self.training!=True),fc=False),
            nn.Conv2d(3,int(32*scale),kernel_size=3,stride=2),
            nn.Softplus(),
            GaussianDropout(int(32*scale),p=0.5,deterministic_test=(self.training!=True),fc=False),
            nn.Conv2d(int(32*scale),int(64*scale),kernel_size=3,stride=2),
            nn.Softplus(),
            GaussianDropout(int(64*scale),p=0.5,deterministic_test=(self.training!=True),fc=False),
        )
        self.fc = nn.Sequential(
            # GaussianDropout(int(64*scale)*7*7,p=0.5,deterministic_test=(self.training!=True),fc=True),
            nn.Linear(int(64*scale)*7*7,int(128*scale)),
            GaussianDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),fc=True),
            nn.Linear(int(128*scale),int(128*scale)),
            GaussianDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),fc=True),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

#########################################################################################################################
class NETVariationalDropoutA(nn.Module):
    def __init__(self,scale=1.0):
        super(NETVariationalDropoutA,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            VariationalDropout(3,p=0.5,deterministic_test=(self.training!=True),deterministic_limit=True,deterministic_sparse=False,fc=False),
            nn.Conv2d(3,int(32*scale),kernel_size=3,stride=2),
            nn.Softplus(),
            VariationalDropout(int(32*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=True,deterministic_sparse=False,fc=False),
            nn.Conv2d(int(32*scale),int(64*scale),kernel_size=3,stride=2),
            nn.Softplus(),
            # VariationalDropout(int(64*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=True,deterministic_sparse=False,fc=False),
        )
        self.fc = nn.Sequential(
            VariationalDropout(int(64*scale)*7*7,p=0.5,deterministic_test=(self.training!=True),deterministic_limit=True,deterministic_sparse=False,fc=True),
            nn.Linear(int(64*scale)*7*7,int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=True,deterministic_sparse=False,fc=True),
            nn.Linear(int(128*scale),int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=True,deterministic_sparse=False,fc=True),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

class EffectNETVariationalDropoutA(nn.Module):
    def __init__(self,scale=1.0):
        super(EffectNETVariationalDropoutA,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            VariationalDropout(3,p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=False,fc=False),
            nn.Conv2d(3,int(32*scale),kernel_size=3,stride=2),
            nn.Softplus(),
            VariationalDropout(int(32*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=False,fc=False),
            nn.Conv2d(int(32*scale),int(64*scale),kernel_size=3,stride=2),
            nn.Softplus(),
            # VariationalDropout(int(64*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=False,fc=False),
        )
        self.fc = nn.Sequential(
            VariationalDropout(int(64*scale)*7*7,p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=False,fc=True),
            nn.Linear(int(64*scale)*7*7,int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=False,fc=True),
            nn.Linear(int(128*scale),int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=False,fc=True),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

class EffectSparseNETVariationalDropoutA(nn.Module):
    def __init__(self,scale=1.0):
        super(EffectSparseNETVariationalDropoutA,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            VariationalDropout(3,p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=True,fc=False),
            nn.Conv2d(3,int(32*scale),kernel_size=3,stride=2),
            nn.Softplus(),
            VariationalDropout(int(32*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=True,fc=False),
            nn.Conv2d(int(32*scale),int(64*scale),kernel_size=3,stride=2),
            nn.Softplus(),
            # VariationalDropout(int(64*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=True,fc=False),
        )
        self.fc = nn.Sequential(
            VariationalDropout(int(64*scale)*7*7,p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=True,fc=True),
            nn.Linear(int(64*scale)*7*7,int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=True,fc=True),
            nn.Linear(int(128*scale),int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=True,fc=True),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

##############################################################################################################################
class NETVariationalDropoutB(nn.Module):
    def __init__(self,scale=1.0):
        super(NETVariationalDropoutB,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            nn.Conv2d(3,int(32*scale),kernel_size=3,stride=2),
            VariationalDropout(int(32*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=True,deterministic_sparse=False,fc=False),
            nn.Softplus(),
            nn.Conv2d(int(32*scale),int(64*scale),kernel_size=3,stride=2),
            VariationalDropout(int(64*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=True,deterministic_sparse=False,fc=False),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            nn.Linear(int(64*scale)*7*7,int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=True,deterministic_sparse=False,fc=True),
            nn.Linear(int(128*scale),int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=True,deterministic_sparse=False,fc=True),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

class EffectNETVariationalDropoutB(nn.Module):
    def __init__(self,scale=1.0):
        super(EffectNETVariationalDropoutB,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            nn.Conv2d(3,int(32*scale),kernel_size=3,stride=2),
            VariationalDropout(int(32*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=False,fc=False),
            nn.Softplus(),
            nn.Conv2d(int(32*scale),int(64*scale),kernel_size=3,stride=2),
            VariationalDropout(int(64*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=False,fc=False),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            nn.Linear(int(64*scale)*7*7,int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=False,fc=True),
            nn.Linear(int(128*scale),int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=False,fc=True),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

class EffectSparseNETVariationalDropoutB(nn.Module):
    def __init__(self,scale=1.0):
        super(EffectSparseNETVariationalDropoutB,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            nn.Conv2d(3,int(32*scale),kernel_size=3,stride=2),
            VariationalDropout(int(32*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=True,fc=False),
            nn.Softplus(),
            nn.Conv2d(int(32*scale),int(64*scale),kernel_size=3,stride=2),
            VariationalDropout(int(64*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=True,fc=False),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            nn.Linear(int(64*scale)*7*7,int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=True,fc=True),
            nn.Linear(int(128*scale),int(128*scale)),
            VariationalDropout(int(128*scale),p=0.5,deterministic_test=(self.training!=True),deterministic_limit=False,deterministic_sparse=True,fc=True),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y