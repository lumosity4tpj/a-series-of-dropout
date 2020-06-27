根据实验发现：

1. 对VDH和VDS如果使用log_sigma2作为参数优化，Hierarchical跟Sparse会很快的稀疏化(Hierarchical更快)，导致Hierarchical不能正确分类(10%)，Sparse则收敛很慢(48%)，但后面做压缩实验在mnist的时候，VDS使用的是log_sigma2，效果还不错，但可能的原因是给了一个学习率衰减?
2. 故VDH和VDS也改为使用log_alpha为参数优化，~~在cifar10上，稀疏能力都一般，但Hierarchical的稀疏能力没有Sparse好~~
3. 对type A 协同：在最后一个conv后和在第一个fc前差不多
4. 对type B 独立：最初new的code效果更好，区别在于一个是对B（new，类似于做了局部重采样），一个是对W(维度更高)，~~所以原因在于此~~~~？~~(更改代码之后结果就相近了)，但在Hierarchical时对B会直接导致log_alpha下降
5. VD中效果type A比type B差，原文又没直接说明cifar10上的结果选用(但论文在mnist上type A更好)，但显然结果相反了？(之前有人复现好像也有此问题)

A: (第一行)在fc前 (第二行)在conv后

B: (第一行)对W (第二行)对B

batchsize=1000 adam默认参数 lr=0.001 epoch=100 一次结果

|                                | Scale(1.0)                     | Scale(1.5)                      | Scale(2.0)                      |
| ------------------------------ | ------------------------------ | ------------------------------- | ------------------------------- |
| Bernoulli                      | 56.32/55.67<br/>56.23/54.96    | 57.22/56.11<br/>56.55/56.21     | 58.05/57.13<br/>58.21/57.63     |
| GaussianS                      | 49.33/48.99                    | 50.93/49.92                     | 52.21/50.75                     |
| GaussianW                      | 56.57/56.35                    | 58.34/57.52                     | 58.52/57.60                     |
| NoDropout                      | 58.53/57.09                    | 58.49/55.51                     | 59.61/56.87                     |
| VariationalA                   | 55.85/55.31                    | 54.63/54.58                     | 54.88/54.41                     |
| VariationalB                   | 56.45/56.45                    | 57.98/56.59                     | 58.30/56.93                     |
| VariationalSparseb对W          | 56.16/55.32                    | 55.38/54.91                     | 56.72/56.10                     |
| VariationalHierarchicalb       | 55.41/55.41<br>**61.60/60.68** | 56.62/56.20<br/>**62.90/62.10** | 55.85/54.97<br/>**62.38/62.02** |
| VariationalHierarchicalA在fc前 | 60.80/59.96                    | 60.73/59.95                     | 60.01/60.01                     |
| VariationalHierarchicalB对W    | 56.70/56.19                    | 57.07/56.26                     | 56.30/55.99                     |

  

new:

|                             | Scale(1.0)                  | Scale(1.5)                  | Scale(2.0)                  |
| --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| GaussianS                   | 49.53/48.64<br/>48.90/48.39 | 51.69/50.53<br/>51.48/51.13 | 51.05/50.22<br/>51.68/50.78 |
| GaussianW                   | 53.86/53.63                 | 56.51/55.75                 | 57.39/57.02                 |
| VariationalA                | 55.42/55.00<br/>58.34/57.57 | 54.32/53.80<br/>55.94/55.12 | 55.41/54.34<br/>55.03/53.80 |
| VariationalB对B             | 59.8/59.64                  | 62.10/**62.10**             | 61.88/60.54                 |
| VariationalHierarchicalA    | 59.78/59.78<br/>60.25/59.69 | 60.59/60.50<br/>59.48/58.64 | 61.22/60.47<br/>59.14/58.04 |
| VariationalHierarchicalB对B | **61.77/61.08**             | **62.14**/61.58             | **62.71/61.30**             |
| VariationalSparseA          | 59.97/59.40<br/>60.17/60.12 | 60.25/60.01<br/>59.60/59.23 | 59.88/59.47<br/>60.24/58.51 |
| VariationalSparseB对B       | **62.38/61.34**             | **62.56/62.29**             | **63.28/62.38**             |

 

Bernoulli/GaussianS：[Dropout: A Simple Way to Prevent Neural Networks from Overﬁtting](https://www.semanticscholar.org/paper/34f25a8704614163c4095b3ee2fc969b60de4698)

GaussianW：[Fast dropout training](https://www.semanticscholar.org/paper/ec92efde21707ddf4b81f301cd58e2051c1a2443)

VariationalA/B：[Variational Dropout and the Local Reparameterization Trick](https://www.semanticscholar.org/paper/f0ddb2bc6e5464d992ddbcdfdc7e894150fc81f2)

VariationalHierarchical：[Variational Bayesian Dropout with a Hierarchical Prior](https://arxiv.org/abs/1811.07533)

VariationalSparse： [Variational Dropout Sparsiﬁes Deep NeuralNetworks](https://www.semanticscholar.org/paper/34cc3ceae5c3f7c8acbb89f2bff63f9d452b00d5)



还未完全按照论文参数设置

LeNet-300-100

|          | acc  | sparsity per layer | compression ratio |
| -------- | ---- | ------------------ | ----------------- |
| SparseVD | 98.6 | 97.0-95.6-61.2     | 32                |



TODO：

模型压缩