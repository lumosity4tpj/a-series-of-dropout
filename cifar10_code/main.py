# -*- coding: utf-8 -*-
"""
@author: lumosity
"""
import os
import argparse
import torch
from utils import setup_seed
from solver import CNN
import random

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type = int, \
                        default = 32,
                        help = 'the size of input image(default:32)')
    # the params when training
    parser.add_argument('--epochs', type = int, \
                        default = 100,
                        help = 'number of epochs(default:1000)')
    parser.add_argument('--batch_size', type = int, \
                        default = 1000, 
                        help = 'the batch size(default:1000)')
    parser.add_argument('--lr', type = float, \
                        default = 0.001,
                        help = 'the learning rate(default:0.001)')
    # the params of path
    parser.add_argument('--data_dir', type = str, \
                        default = '/root/data/lumosity/Dropout/cifar10_code/data_cifar10',
                        help = 'the path of rgb train txt')
    parser.add_argument('--net_pth', type = str, \
                        default = ' ',
                        help = 'the path of loading the model')
    parser.add_argument('--save_pth', type = str, \
                        default = '/root/data/lumosity/Dropout/cifar10_code/model',
                        help = 'the path of saving the model')
    # the other params
    parser.add_argument('--ngpu', type = int, \
                        default = 1,
                        help = 'the number of gpus available')
    parser.add_argument('--gpu_id', type = int, \
                        default = 0,
                        help = 'the id of gpu available')
    parser.add_argument('--dropout_type', type = str, \
                        default = 'NETfcGaussianDropoutfcnosoftplus',
                        help = 'dropout type')
    parser.add_argument('--scale', type = float, \
                        default = 2.0,
                        help = 'the width scale of model')
    return parser.parse_args()

if __name__ == '__main__':
    opt = args()
    # for i in range(5):
    for opt.scale in [1.0,1.5,2.0]:
        manual_Seed = random.randint(1,10000)
        env = '%s_%.1f_%s'%(opt.dropout_type,opt.scale,manual_Seed)
        setup_seed(manual_Seed)
        torch.backends.cudnn.benchmark = True
        if not os.path.exists(opt.save_pth):
            os.makedirs(opt.save_pth)
        path = '%s/%.1f'%(opt.dropout_type,opt.scale)
        if not os.path.exists(path):
            os.makedirs(path)
        if opt.ngpu > 1:
            device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        else:
            device = torch.device('cuda:%d'%(opt.gpu_id) if torch.cuda.is_available else 'cpu')
        cnn = CNN(device,opt.ngpu,opt.data_dir,opt.img_size,opt.batch_size,opt.dropout_type,opt.scale,manual_Seed)
        cnn.train(opt.epochs,env,opt.lr,opt.save_pth,opt.net_pth)