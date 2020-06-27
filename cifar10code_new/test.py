import numpy as np 
import glob
import os

def read_data_compute_mean_final(txt_dir):
    for i in [1.0,1.5,2.0]:
        try:
            rootdir = os.path.join(txt_dir,str(i))
            line = glob.glob(rootdir+'/*end.txt')
            # print(line)
            a = 0.
            for j in line:
                b = np.loadtxt(j)
                # print(b)
                a += float(b)
            # print(len(line))
            mean = a/len(line)
            print('%s_end:%f'%(rootdir,mean))
        except:
            pass

def read_data_compute_mean_best(txt_dir):
    for i in [1.0,1.5,2.0]:
        try:
            rootdir = os.path.join(txt_dir,str(i))
            line = glob.glob(rootdir+'/*best.txt')
            # print(line)
            a = 0.
            for j in line:
                b = np.loadtxt(j)
                # print(b)
                a += float(b)
            # print(len(line))
            mean = a/len(line)
            print('%s_best:%f'%(rootdir,mean))
        except:
            pass

if __name__ == '__main__':
    dir1 = '/root/data/lumosity/Dropout/cifar10_code_new/'
    dir2 = ['NETGaussianDropoutSrivastava','NETGaussianDropoutWang']
    for m in dir2:
        txt_dir = dir1 + m
        read_data_compute_mean_best(txt_dir)
        read_data_compute_mean_final(txt_dir)