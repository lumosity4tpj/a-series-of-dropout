### cifar-10分类：

> 参数设置均为：batchsize=1000  adam默认参数  lr=0.001  epoch=100  一次结果
>
> A：指VD论文中提出的协同，对于不同行，指在conv的尺寸reshape为fc的尺寸时，第一行指在先reshape再fc前dropout，第二行是指在conv后dropout再reshape
>
> B：指VD论文中提出的独立，对于不同行，第一行指对W先加噪声再做fc/conv(维度更高)，第二行是指先做fc/conv再对输出B进行加噪声(局部重采样)



第一个版本，使用log_alpha作为更新参数，将网络层跟dropout层结合起来

对于fc层，shape为[out_features,in_features]

对于conv层，shape为[out_channels,in_channels,*kernel_size]



|                                | Scale(1.0)                     | Scale(1.5)                      | Scale(2.0)                      |
| ------------------------------ | ------------------------------ | ------------------------------- | ------------------------------- |
| Bernoulli                      | 56.32/55.67<br/>56.23/54.96    | 57.22/56.11<br/>56.55/56.21     | 58.05/57.13<br/>58.21/57.63     |
| GaussianS                      | 56.06/55.27                    | 57.24/57.17                     | 57.87/55.94                     |
| GaussianW                      | 56.57/56.35                    | 58.34/57.52                     | 58.52/57.60                     |
| NoDropout                      | 58.53/57.09                    | 58.49/55.51                     | 59.61/56.87                     |
| VariationalA                   | 55.85/55.31                    | 54.63/54.58                     | 54.88/54.41                     |
| VariationalB                   | 56.45/56.45                    | 57.98/56.59                     | 58.30/56.93                     |
| VariationalHierarchicalB       | 55.41/55.41<br>**61.60/60.68** | 56.62/56.20<br/>**62.90/62.10** | 55.85/54.97<br/>**62.38/62.02** |
| VariationalHierarchicalA在fc前 | 60.80/59.96                    | 60.73/59.95                     | 60.01/60.01                     |
| VariationalHierarchicalB对W    | 56.70/56.19                    | 57.07/56.26                     | 56.30/55.99                     |

  

第二个版本，使用log_alpha作为更新参数，将网络层跟dropout层分开(第一个版本应该更合理点)

对于fc层，shape为[in_features]，即上一个版本的out_features

对于conv层，shape为[in_channels]，即上一个版本的out_channels



|                             | Scale(1.0)                  | Scale(1.5)                  | Scale(2.0)                  |
| --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| GaussianS                   | 56.06/55.27                 | 57.24/57.17                 | 57.87/55.94                 |
| GaussianW                   | 53.86/53.63                 | 56.51/55.75                 | 57.39/57.02                 |
| VariationalA                | 55.42/55.00<br/>58.34/57.57 | 54.32/53.80<br/>55.94/55.12 | 55.41/54.34<br/>55.03/53.80 |
| VariationalB对B             | 59.8/59.64                  | 62.10/**62.10**             | 61.88/60.54                 |
| VariationalHierarchicalA    | 59.78/59.78<br/>60.25/59.69 | 60.59/60.50<br/>59.48/58.64 | 61.22/60.47<br/>59.14/58.04 |
| VariationalHierarchicalB对B | **61.77/61.08**             | **62.14**/61.58             | **62.71/61.30**             |

 

由上述两个实验可以得知：

- 对于type A：在最后一个conv后和在第一个fc前结果差不多

- 对于type B：对B局部重采样结果更好(符合预期)，且差距较大

- 可以得出在cifar10上分类结果的效果上VariationalHierarchical>Variational>Bernoulli/GaussianS/Nodropout/GaussianW

  我觉得是符合预期的，但跟VariationalHierarchical论文比有一定的差距

主要存在的问题：

- 使用log_sigma或者log_sigma2作为更新参数，会出现Hierarchical不能收敛，VD收敛很慢等问题，VariationalSparse代码中使用的是log_sigma，Hierarchical参考的代码使用的是log_alpha

  可能的原因：对于初值的设置很敏感，这在其他人复现和实验中也有说到，我更改log_sigma2的初值，对于Hierarchical要么loss为nan，要么损失不变

- VD复现的实验type A比type B差，在原文cifar10中未涉及，但在mnist上是type A更好

- mnist分类上目前则完全比较不出效果好坏，准确率差距太小，可能有某个细节就导致一些波动



### mnist压缩

> epoch 200 batchsize 100 adam默认参数
>
> 学习率为no dropout的为2e-4在前100个epoch线性下降到1e-4(它代码中是这样，但这样跟有dropout的设置不同)，有dropout的为从1e-3经过200个epoch线性下降到0
>
> kl损失的系数均为为前5个epoch为0，然后经过15个epoch线性增加到1



Lenet5：

|                    | sparsity per layer  | acc   | compression ratio |
| ------------------ | ------------------- | ----- | ----------------- |
|                    |                     |       |                   |
| LeNetconvNoDropout |                     | 0.991 |                   |
| LeNetconvSVD       | 57.2-90.4-99.0-90.1 | 0.993 | 61.39             |

Lenet300-100：

|                  | sparsity per layer | acc    | compression ratio |
| ---------------- | ------------------ | ------ | ----------------- |
| LeNetfcHVD       | 98.0-96.7-76.6     | 0.9827 | 45.58             |
| LeNetfcNoDropout |                    | 0.9808 |                   |
| LeNetfcSVD       | 98.0-97.1-74.7     | 0.983  | 46.25             |

目前的问题：

- 压缩比效果跟论文差距较大，且HVD跟SVD没明显差距
- HVD卷积部分出现loss nan的情况



我的感受：在实际应用中，不同的dropout它的压缩或者其他任务的结果，可能并不能简单的分出好坏，更多的应该是通过调整超参数或者相关模型结构

这是选取LeNetfcSVD中fc1的可视化结果：



![937_fc1_hot](https://github.com/lumosity4tpj/a-series-of-dropout/blob/master/pics/937_fc1_hot.png)

![937_fc1](https://github.com/lumosity4tpj/a-series-of-dropout/blob/master/pics/937_fc1.png)





Bernoulli/GaussianS：[Dropout: A Simple Way to Prevent Neural Networks from Overﬁtting](https://www.semanticscholar.org/paper/34f25a8704614163c4095b3ee2fc969b60de4698)

GaussianW：[Fast dropout training](https://www.semanticscholar.org/paper/ec92efde21707ddf4b81f301cd58e2051c1a2443)

VariationalA/B：[Variational Dropout and the Local Reparameterization Trick](https://www.semanticscholar.org/paper/f0ddb2bc6e5464d992ddbcdfdc7e894150fc81f2)

VariationalHierarchical：[Variational Bayesian Dropout with a Hierarchical Prior](https://arxiv.org/abs/1811.07533)

VariationalSparse： [Variational Dropout Sparsiﬁes Deep NeuralNetworks](https://www.semanticscholar.org/paper/34cc3ceae5c3f7c8acbb89f2bff63f9d452b00d5)

