# -*- coding: utf-8 -*-
"""
@author: lumosity
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torchvision import datasets,transforms
import visdom
import time
import os
from tqdm import tqdm
# from tensorboardX import SummaryWriter
from utils import weights_init
from VD_new_layers import VariationalDropout
from models_new import NETGaussianDropoutSrivastava,NETGaussianDropoutWang,NETVariationalDropoutA,NETVariationalDropoutB,EffectNETVariationalDropoutA,EffectNETVariationalDropoutB

class CNN(object):
    def __init__(self,device,ngpu,data_dir,img_size,batch_size,dropout_type,scale,seed):
        self.ngpu = ngpu
        self.device = device
        self.img_size,self.batch_size = img_size,batch_size
        self.data_dir = data_dir
        self.dropout_type = dropout_type
        self.scale = scale
        self.seed = seed

    def train(self,epochs,env,lr,save_pth,net_pth=' '):
        # vis = visdom.Visdom(env=env)
        # assert vis.check_connection()
        # train_loss_win = vis.line(np.arange(10))
        # test_loss_win = vis.line(np.arange(10))
        # train_acc_win = vis.line(np.arange(10))
        # test_acc_win = vis.line(np.arange(10))
        # lr_win = vis.line(np.arange(10))
        # alpha_win = vis.line(X=np.column_stack((np.array(0),np.array(0))),Y=np.column_stack((np.array(0),np.array(0))))
        iter_count = 0
        model = self.load_model(net_pth)
        
        # dataloader
        transform = {
            'train':transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]),
            'val':transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]),
        }
        dataset = {
            'train':datasets.CIFAR10(root=os.path.join(self.data_dir,'train'),train=True,transform=transform['train'],download=True),
            'val':datasets.CIFAR10(root=os.path.join(self.data_dir,'val'),train=False,transform=transform['val'],download=True),
            }
        dataloader = {x:torch.utils.data.DataLoader(dataset[x],batch_size=self.batch_size,shuffle=True,num_workers=0) for x in ['train','val']}
        datasize = {x:len(dataset[x]) for x in ['train','val']}

        optimizer = optim.Adam(model.parameters(),lr=lr)
        criterion = nn.CrossEntropyLoss()
        start_t = time.time()
        best_train_epoch_correct = 0.
        best_test_epoch_correct = 0.
        alpha_list = [] 
        for epoch in range(epochs):
            for phase in ['train','val']:
                epoch_start_t = time.time()
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_correct = 0

                for i,(x,label) in enumerate(dataloader[phase]):
                    batch_size = x.size(0)
                    optimizer.zero_grad()
                    x,label = x.to(self.device),label.to(self.device)
                    
                    with torch.set_grad_enabled(phase=='train'):
                        y = model(x)
                        _,preds = torch.max(y,1)
                        assert label.size(0) == y.size(0)
                        loss = criterion(y,label)
                        kl = 0
                        if self.dropout_type == 'NETVariationalDropoutA' or 'NETVariationalDropoutB' or 'EffectNETVariationalDropoutA' or 'EffectNETVariationalDropoutB':
                            for name, module in model.named_modules():
                                if isinstance(module,VariationalDropout):
                                    kl += module.kl()
                            # if phase == 'train':
                            #     m = []
                            #     for i in model.named_parameters():
                            #         if i[0].split('.')[-1] == 'alpha':
                            #             t = torch.mean(i[1])
                            #             m.append(t)
                            #     alpha_list.append(m)
                        else:
                            assert self.dropout_type == 'NETNoDropout' or 'NETBernoulliDropout' or 'NETGaussianDropoutWang' or 'NETGaussianDropoutSrivastava','dropout type error'
                        loss = loss + kl/50000

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item()*batch_size
                    running_correct += (preds==label).sum()
                
                epoch_loss = running_loss/datasize[phase]
                epoch_correct = running_correct.double()/datasize[phase]
                epoch_t = time.time() - epoch_start_t
                print('[mode:%s,\t{%d}/{%d}],\ttime:%.0fm %.0fs,\tepoch_loss:%.4f,\tepoch_correct:%.4f'%(phase,epoch,epochs,epoch_t//60,epoch_t%60,epoch_loss,epoch_correct))

                with torch.no_grad():
                    if phase == 'train':
                        if epoch_correct >= best_train_epoch_correct:
                            best_train_epoch_correct = epoch_correct
                        # vis.line(Y=np.array([epoch_loss]),
                        #         X=np.array([iter_count]),
                        #         update='append',
                        #         win=train_loss_win,
                        #         opts=dict(legend=['train_loss']))
                        # vis.line(Y=np.array([epoch_correct.item()]),
                        #         X=np.array([iter_count]),
                        #         update='append',
                        #         win=train_acc_win,
                        #         opts=dict(legend=['train_acc']))
                        print('train_best_epoch_correct:',best_train_epoch_correct.item())
                    else:
                        if epoch_correct >= best_test_epoch_correct:
                            best_test_epoch_correct = epoch_correct
                        # vis.line(Y=np.array([epoch_loss]),
                        #         X=np.array([iter_count]),
                        #         update='append',
                        #         win=test_loss_win,
                        #         opts=dict(legend=['test_loss']))
                        # vis.line(Y=np.array([epoch_correct.item()]),
                        #         X=np.array([iter_count]),
                        #         update='append',
                        #         win=test_acc_win,
                        #         opts=dict(legend=['test_acc']))
                        print('val_best_epoch_correct:',best_test_epoch_correct.item())
                        iter_count += 1

            # self.save_model(model,epoch,save_pth)
        with open('%s/%.1f/%d_end.txt'%(self.dropout_type,self.scale,self.seed),'w') as f:
            f.writelines([str(epoch_correct.item())])
        with open('%s/%.1f/%d_best.txt'%(self.dropout_type,self.scale,self.seed),'w') as f:
            f.writelines([str(best_test_epoch_correct.item())])
        
        self.save_model(model,epoch,save_pth)
        total_t = time.time() - start_t
        print('Training finsh in %.0f h %.0f m %.0f s'%(total_t//3600,(total_t%3600)//60,total_t%60))

    def load_model(self,net_pth):        
        # if self.dropout_type == 'NETNoDropout':
        #     net = NETNoDropout(scale=self.scale)
        # elif self.dropout_type == 'NETBernoulliDropout':
        #     net = NETBernoulliDropout(scale=self.scale)
        if self.dropout_type == 'NETGaussianDropoutWang':
            net = NETGaussianDropoutWang(scale=self.scale)
        elif self.dropout_type == 'NETGaussianDropoutSrivastava':
            net = NETGaussianDropoutSrivastava(scale=self.scale)
        elif self.dropout_type == 'NETVariationalDropoutA':
            net = NETVariationalDropoutA(scale=self.scale)
        elif self.dropout_type == 'NETVariationalDropoutB':
            net = NETVariationalDropoutB(scale=self.scale)
        elif self.dropout_type == 'EffectNETVariationalDropoutA':
            net = EffectNETVariationalDropoutA(scale=self.scale)
        elif self.dropout_type == 'EffectNETVariationalDropoutB':
            net = EffectNETVariationalDropoutB(scale=self.scale)
        else:
            print('please check your dropout type')
        if net_pth != ' ':
            net.load_state_dict(torch.load(net_pth,map_location=lambda storage,loc:storage))##gpu -> cpu
            print('Load from: ',net_pth)
        else:
            # net.apply(weights_init)
            print('no model & init weight')
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            net = nn.DataParallel(net,list(range(self.ngpu)))
        else:
            net.to(self.device)
        return net

    def save_model(self,net,epoch,model_dir):
        path = '%s/%s/%.1f/%d'%(model_dir,self.dropout_type,self.scale,self.seed)
        if not os.path.exists(path):
            os.makedirs(path)
        if (self.device.type == 'cuda') and (self.ngpu > 1): 
            torch.save(net.module.state_dict(),'%s/epoch_%d.pth'%(path,epoch))
        else:
            torch.save(net.state_dict(),'%s/epoch_%d.pth'%(path,epoch))
        print('the model has saved(epoch:%d)'%(epoch))