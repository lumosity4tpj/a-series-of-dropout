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
from GD_layers import GaussianDropoutWfc,GaussianDropoutWcnn,GaussianDropoutSfc,GaussianDropoutScnn,GaussianDropout
from VD_layers import VariationalDropoutfc,VariationalDropoutcnn
from VDH_layers import VariationalDropoutHierarchicalfc,VariationalDropoutHierarchicalcnn
from VDS_layers import VariationalDropoutSparsefc,VariationalDropoutSparsecnn
from VD_effect_layers import VariationalDropoutcnne,VariationalDropoutfce

#########################################################################################################################
class NETNoDropout(nn.Module):
    def __init__(self,scale=1.0):
        super(NETNoDropout,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            nn.Conv2d(3,int(32*scale),kernel_size=3,stride=2),
            nn.Softplus(),
            nn.Conv2d(int(32*scale),int(64*scale),kernel_size=3,stride=2),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            nn.Linear(int(64*scale)*7*7,int(128*scale)),
            nn.Linear(int(128*scale),int(128*scale)),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

class NETBernoulliDropout(nn.Module):
    def __init__(self,scale=1.0):
        super(NETBernoulliDropout,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(3,int(32*scale),kernel_size=3,stride=2),
            nn.Softplus(),
            nn.Dropout(p=0.5),
            nn.Conv2d(int(32*scale),int(64*scale),kernel_size=3,stride=2),
            nn.Softplus(),
            nn.Dropout(p=0.5),
        )
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(int(64*scale)*7*7,int(128*scale)),
            nn.Dropout(p=0.5),
            nn.Linear(int(128*scale),int(128*scale)),
            nn.Dropout(p=0.5),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

class NETGaussianDropoutWang(nn.Module):
    def __init__(self,scale=1.0):
        super(NETGaussianDropoutWang,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            GaussianDropoutWcnn(3,int(32*scale),kernel_size=3,stride=2,p=0.5,deterministic_test=(self.training!=True)),
            nn.Softplus(),
            GaussianDropoutWcnn(int(32*scale),int(64*scale),kernel_size=3,stride=2,p=0.5,deterministic_test=(self.training!=True)),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            GaussianDropoutWfc(int(64*scale)*7*7,int(128*scale),p=0.5,deterministic_test=(self.training!=True)),
            GaussianDropoutWfc(int(128*scale),int(128*scale),p=0.5,deterministic_test=(self.training!=True)),
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
            GaussianDropoutScnn(3,int(32*scale),kernel_size=3,stride=2,p=0.5,deterministic_test=(self.training!=True)),
            nn.Softplus(),
            GaussianDropoutScnn(int(32*scale),int(64*scale),kernel_size=3,stride=2,p=0.5,deterministic_test=(self.training!=True)),
            nn.Softplus(),
            # GaussianDropout(int(64*scale),p=0.5),
        )
        self.fc = nn.Sequential(
            # nn.Linear(int(64*scale)*7*7,int(128*scale)),
            GaussianDropoutSfc(int(64*scale)*7*7,int(128*scale),p=0.5,deterministic_test=(self.training!=True)),
            GaussianDropoutSfc(int(128*scale),int(128*scale),p=0.5,deterministic_test=(self.training!=True)),
            GaussianDropoutSfc(int(128*scale),10,p=0.5,deterministic_test=(self.training!=True)),
            # nn.Linear(int(128*scale),10),
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
            VariationalDropoutcnn(3,int(32*scale),kernel_size=3,stride=2,p=0.5,dropout_type='A',deterministic_test=(self.training!=True),deterministic_limit=True),
            nn.Softplus(),
            VariationalDropoutcnn(int(32*scale),int(64*scale),kernel_size=3,stride=2,p=0.5,dropout_type='A',deterministic_test=(self.training!=True),deterministic_limit=True),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            VariationalDropoutfc(int(64*scale)*7*7,int(128*scale),p=0.5,dropout_type='A',deterministic_test=(self.training!=True),deterministic_limit=True),
            VariationalDropoutfc(int(128*scale),int(128*scale),p=0.5,dropout_type='A',deterministic_test=(self.training!=True),deterministic_limit=True),
            VariationalDropoutfc(int(128*scale),10,p=0.5,dropout_type='A',deterministic_test=(self.training!=True),deterministic_limit=True),
            # nn.Linear(int(128*scale),10),
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
            VariationalDropoutcnne(3,int(32*scale),kernel_size=3,stride=2,p=0.5,dropout_type='A',deterministic_test=(self.training!=True),deterministic_limit=False),
            nn.Softplus(),
            VariationalDropoutcnne(int(32*scale),int(64*scale),kernel_size=3,stride=2,p=0.5,dropout_type='A',deterministic_test=(self.training!=True),deterministic_limit=False),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            VariationalDropoutfce(int(64*scale)*7*7,int(128*scale),p=0.5,dropout_type='A',deterministic_test=(self.training!=True),deterministic_limit=False),
            VariationalDropoutfce(int(128*scale),int(128*scale),p=0.5,dropout_type='A',deterministic_test=(self.training!=True),deterministic_limit=False),
            VariationalDropoutfce(int(128*scale),10,p=0.5,dropout_type='A',deterministic_test=(self.training!=True),deterministic_limit=False),
            # nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

class NETVariationalDropoutB(nn.Module):
    def __init__(self,scale=1.0):
        super(NETVariationalDropoutB,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            VariationalDropoutcnn(3,int(32*scale),kernel_size=3,stride=2,p=0.5,dropout_type='B',deterministic_test=(self.training!=True),deterministic_limit=True),
            nn.Softplus(),
            VariationalDropoutcnn(int(32*scale),int(64*scale),kernel_size=3,stride=2,p=0.5,dropout_type='B',deterministic_test=(self.training!=True),deterministic_limit=True),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            VariationalDropoutfc(int(64*scale)*7*7,int(128*scale),p=0.5,dropout_type='B',deterministic_test=(self.training!=True),deterministic_limit=True),
            VariationalDropoutfc(int(128*scale),int(128*scale),p=0.5,dropout_type='B',deterministic_test=(self.training!=True),deterministic_limit=True),
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
            VariationalDropoutcnne(3,int(32*scale),kernel_size=3,stride=2,p=0.5,dropout_type='B',deterministic_test=(self.training!=True),deterministic_limit=False),
            nn.Softplus(),
            VariationalDropoutcnne(int(32*scale),int(64*scale),kernel_size=3,stride=2,p=0.5,dropout_type='B',deterministic_test=(self.training!=True),deterministic_limit=False),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            VariationalDropoutfce(int(64*scale)*7*7,int(128*scale),p=0.5,dropout_type='B',deterministic_test=(self.training!=True),deterministic_limit=False),
            VariationalDropoutfce(int(128*scale),int(128*scale),p=0.5,dropout_type='B',deterministic_test=(self.training!=True),deterministic_limit=False),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

#########################################################################################################################
class NETVariationalDropoutSparse(nn.Module):
    def __init__(self,scale=1.0):
        super(NETVariationalDropoutSparse,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            VariationalDropoutSparsecnn(3,int(32*scale),kernel_size=3,stride=2,deterministic_test=(self.training!=True),deterministic_compress=False),
            nn.Softplus(),
            VariationalDropoutSparsecnn(int(32*scale),int(64*scale),kernel_size=3,stride=2,deterministic_test=(self.training!=True),deterministic_compress=False),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            VariationalDropoutSparsefc(int(64*scale)*7*7,int(128*scale),deterministic_test=(self.training!=True),deterministic_compress=False),
            VariationalDropoutSparsefc(int(128*scale),int(128*scale),deterministic_test=(self.training!=True),deterministic_compress=False),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

class CompressNETVariationalDropoutSparse(nn.Module):
    def __init__(self,scale=1.0):
        super(CompressNETVariationalDropoutSparse,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            VariationalDropoutSparsecnn(3,int(32*scale),kernel_size=3,stride=2,deterministic_test=(self.training!=True),deterministic_compress=True),
            nn.Softplus(),
            VariationalDropoutSparsecnn(int(32*scale),int(64*scale),kernel_size=3,stride=2,deterministic_test=(self.training!=True),deterministic_compress=True),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            VariationalDropoutSparsefc(int(64*scale)*7*7,int(128*scale),deterministic_test=(self.training!=True),deterministic_compress=True),
            VariationalDropoutSparsefc(int(128*scale),int(128*scale),deterministic_test=(self.training!=True),deterministic_compress=True),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

class NETVariationalDropoutHierarchical(nn.Module):
    def __init__(self,scale=1.0):
        super(NETVariationalDropoutHierarchical,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            VariationalDropoutHierarchicalcnn(3,int(32*scale),kernel_size=3,stride=2,deterministic_test=(self.training!=True),deterministic_compress=False),
            nn.Softplus(),
            VariationalDropoutHierarchicalcnn(int(32*scale),int(64*scale),kernel_size=3,stride=2,deterministic_test=(self.training!=True),deterministic_compress=False),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            VariationalDropoutHierarchicalfc(int(64*scale)*7*7,int(128*scale),deterministic_test=(self.training!=True),deterministic_compress=False),
            VariationalDropoutHierarchicalfc(int(128*scale),int(128*scale),deterministic_test=(self.training!=True),deterministic_compress=False),
            # VariationalDropoutHierarchicalfc(int(128*scale),10,deterministic_test=(self.training!=True),deterministic_compress=False),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y

class CompressNETVariationalDropoutHierarchical(nn.Module):
    def __init__(self,scale=1.0):
        super(CompressNETVariationalDropoutHierarchical,self).__init__()
        self.scale = scale

        self.cnn = nn.Sequential(
            VariationalDropoutHierarchicalcnn(3,int(32*scale),kernel_size=3,stride=2,deterministic_test=(self.training!=True),deterministic_compress=True),
            nn.Softplus(),
            VariationalDropoutHierarchicalcnn(int(32*scale),int(64*scale),kernel_size=3,stride=2,deterministic_test=(self.training!=True),deterministic_compress=True),
            nn.Softplus(),
        )
        self.fc = nn.Sequential(
            VariationalDropoutHierarchicalfc(int(64*scale)*7*7,int(128*scale),deterministic_test=(self.training!=True),deterministic_compress=True),
            VariationalDropoutHierarchicalfc(int(128*scale),int(128*scale),deterministic_test=(self.training!=True),deterministic_compress=True),
            nn.Linear(int(128*scale),10),
            # nn.Softmax(),
        )

    def forward(self,x):
        x1 = self.cnn(x)
        x2 = x1.view(-1,int(64*self.scale)*7*7)
        y = self.fc(x2)
        return y
