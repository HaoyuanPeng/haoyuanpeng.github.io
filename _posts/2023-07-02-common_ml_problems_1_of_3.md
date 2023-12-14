---
layout: post
title: 试答机器学习理论知识中的一些常见问题 (1 / 3)
date: 2023-07-02 16:51:00 +0800
description: 试答机器学习理论知识中的一些常见问题  (1 / 3)
tags: [Machine Learning]
relatedposts: false
---

有网友总结了机器学习理论知识中的一些常见问题，链接见：

1. [https://www.1point3acres.com/bbs/thread-713903-1-1.html](https://www.1point3acres.com/bbs/thread-713903-1-1.html)
2. [https://www.1point3acres.com/bbs/thread-714090-1-1.html](https://www.1point3acres.com/bbs/thread-714090-1-1.html)
3. [https://www.1point3acres.com/bbs/thread-714558-1-1.html](https://www.1point3acres.com/bbs/thread-714558-1-1.html)

现试答如下，欢迎批评指正。本文为第一部分，共三部分，每一部分与上述三个链接一一对应。

#### **overfitting / underfiting是指的什么**

_overfitting指的是模型过度拟合了训练集的样本，包括其中的噪声，表现为训练集的Loss很低，但验证集的Loss较高；
underfitting指的是因为参数量、特征选择、训练进度等原因，未能充分学习训练集中样本的特征，表现为训练集和验证集的loss都较高。_

#### **bias/variance trade off是指的什么**

_bias指的是模型的预测结果与Ground Truth之差的期望，variance指的是同一个模型在不同的训练数据下，对同一个样本的预测结果的方差。一般而言，随着模型复杂度和特征复杂度的增加，bias会下降，而variance会上升，因此需要找到一个合适的复杂度，使得bias^2 + variance最小。_

#### **过拟合一般有哪些预防手段**

_数据方面：增加训练数据量；对训练数据进行数据增强；减少训练数据的特征数量；_

_模型和训练策略方面：使用单独的验证集或使用交叉验证来观察是否发生了过拟合；使用Regularization控制模型参数的scale；使用Early Stopping控制训练进度；使用Ensemble降低Variance。_

#### **Generative和Discriminative的区别**

_对于输入特征x和样本y，generative model学习P(x,y)，而discriminative model学习P(y\|x)_

#### **Give a set of ground truths and 2 models, how do you be confident that one model is better than another?**

_确定合适的metric，并使用k-fold cross validation对模型效果进行验证。_

#### **L1 vs L2, which one is which and difference**

_L1 Norm是将所有参数的绝对值之和作为正则项，L2 Norm是将所有参数的平方和作为正则项。L1对outlier更Robust，且有稀疏性；L2计算起来更方便，且有唯一解。_

#### **Lasso / Ridge 的解释（Prior分别是什么）**

_Lasso是指加入系数的L1 Norm作为正则项，而Ridge是加入系数的L2 Norm作为正则项。L1的prior是认为参数w的先验概率满足拉普拉斯分布，而L2的prior是认为参数w的先验满足高斯分布。_

#### **Lasso / Ridge 的推导**

_Lasso:_

$$\begin{aligned}
L(w)=\|Xw - Y\|^2 + \alpha \sum \|w_j\|  \\
\frac{\partial L(w)}{\partial w} = 2X^TXw - 2X^TY + \alpha C \\
w = (X^TX)^{-1}(X^TY-\frac{\alpha}{2}C)

\end{aligned}$$

_Ridge:_

$$\begin{aligned}
L(w)=\|Xw - Y\|^2 + \alpha \sum \|w_j\|^2 \\
\frac{\partial L(w)}{\partial w} = 2X^TXw - 2X^TY + 2\alpha w \\
(X^X+\alpha I)w = X^TY \\
w = (X^TX + \alpha I)^{-1}X^TY

\end{aligned}$$


#### **为什么L1比L2稀疏？**

_定义损失函数与某个权重参数x的函数为L(x)。_
_当施加L1 Norm的时候，$$f(x)=L(x)+C|x|$$，则$$f'(0-)=L'(0)-C，f'(0+)=L'(0)+C$$，要使0成为f的极值点，则需要$$f'(0-)*f'(0+)<0$$，解得$$C>|L'(0)|$$。也就是说，只要正则项的系数C大于Loss function在参数x为0的时候的导数的绝对值，则x的最优解就是0。_

_当施加L2 Norm的时候，$$f(x)=L(x)+Cx^2$$，则$$f'(0)=L'(0)$$，要使x=0为极值点，则需要$$L'(0)=0$$，条件很难满足。_

#### **为什么regularization works？**

_因为过拟合的曲线考虑到了训练集上每个样本点的噪声，因此会发生“突变”的现象，即斜率会变得较大。对线性模型来说，斜率其实就是参数的值，推广到多维复杂模型，参数的值越大，越容易发生突变。而加入正则项之后，Loss function与参数值相关，优化器要降低loss，就要降低参数的值，从而降低突变的可能性，因此可以抑制过拟合现象。_

_对于Lasso而言，因为将权重惩罚到了0，相当于降低了模型使用的特征数量即复杂度。_

#### **为什么regularization用L1、L2，而不用L3、L4？**

_因为L1和L2的效果已经得到了充分验证，且L1能产生稀疏特征。相比较而言，L3和L4的计算复杂度更大，效果也没有充分验证。_

#### **Precision and Recall, trade-off**

_Precision表示True Positive占所有预测的Positive的比例，衡量模型的产生False Positive的程度。
Recall表示True Positive占所有正样本的比例，衡量模型产生False Negative的程度。_ 

_可用通过F Score，基于Precision和Recall对实际任务的权重，对Precision和Recall进行Trade Off._

#### **label不平衡时用什么metric**

_可以用各个类别，或者关注的正样本类别（二分类）的P、R、F1作为metric，在二分类场景下，也可以用ROC-AUC作为metric。_

#### **分类问题该用什么metric，and why？**

_Accuracy、Top-K acc、P、R、F、混淆矩阵，P-R Curve、ROC-AUC_

#### **confusion matrix**

_一个矩阵，matrix\[i\]\[j\]表示实际为第i类，预测为第j类的样本数量。_

#### **AUC的解释（The probability of ranking a randomly selected positive sample higher ...)**

_ROC曲线是二分类模型在不同分类阈值下的效果曲线，横坐标为FPR，即假阳率；纵坐标为召回率。FPR=0时，Recall=0，FPR=1时，Recall=1。因此ROC曲线
连接了(0,0)和(1, 1)两点。ROC-AUC是指曲线下的面积，面积越大，表示模型在相同的FPR下，Recall越高，因此模型的效果越好。同时，ROC-AUC也正好等于
随机挑选一个正样本和一个负样本，正样本的得分（即预测为正样本的概率）比负样本高的概率。_

_ROC-AUC的概率意义的证明暂缺。_

#### **true positive rate, false positive rate, ROC**

_TPR = Recall，FPR = FP / (FP + TN)；ROC为横坐标为FPR，纵坐标为TPR的曲线，可以衡量模型的效果。ROC仅对排序敏感，而对具体的预测概率值不敏感。
两个模型可能ROC-AUC相同，但在不同的FPR区间表现的效果差异较大（MisLeading)。同时，当用户对FPR和TPR有不同的重视程度时，这个指标无法满足。_

#### **Log-loss是什么，什么时候用Log-loss**

_Log-loss就是Likelihood的负对数，即$$-logP(Y_{gt}\vert X)$$，在分类场景，尤其是关注类别置信度的分类场景时适用。_

#### **场景相关的问题，比如ranking design的时候用什么metric，推荐的时候用什么等？**

_排序任务的指标：_
- _MAP：将排序后，所有relevant position处的precision进行平均_
- _MRR：将Top-1所在rank的倒数作平均_
- _NDCG：DCG就是每个位置的相关性，除以相应位置的对数（排序越往后，重要性越低）。NDCG就是DCG除以DCG的理论最大值。_

#### **用MSE做loss的Logistic Regression是convex problem吗？**

_不是。证明如下：_

$$\begin{aligned}
令f(x) = (y - \hat y)^2, 其中\hat y = \frac{1}{1 + e^{-\theta x}} \\
则有\frac{\partial f}{\partial \theta} = \frac{\partial f}{\partial \hat y} \frac{\partial \hat y}{\partial \theta} \\
=-2(y-\hat y)\hat y(1-\hat y)x \\
=-2(y\hat y - y\hat y ^ 2 - \hat y^2 + \hat y^3)x

\frac{\partial^2 f}{\partial \theta^2} \\
= -2(y\hat y - y\hat y ^ 2 - \hat y^2 + \hat y^3)x \cdot {\hat y (1 - \hat y)x} \\  
\\ \because \hat y \in (0, 1)
\\ \therefore x^2\hat y (1- \hat y) > 0
\end{aligned}$$

_在$$y=0$$和$$y=1$$时分别求上式>0的$$\hat y$$的取值范围，可得上式有正有负，所以是non convex的。_


#### **解释并写出MSE的公式，什么时候用到MSE？**
_MSE就是Mean Squared Error，即每个样本的预测值与真实值之差的平方的平均值。在回归任务中，假设真实值和观测值的误差服从高斯分布，则会用到MSE，如果误差服从拉普拉斯分布，则用MAE。_

#### **Linear Regression最小二乘法和MSE的关系**
_最小二乘法就是最小化MSE的一种求解方法。_

#### **什么是relative entropy / cross entropy，以及KL Divergence，他们的intuition**
_Relative entropy就是KL Divergence，衡量两个概率分布的差异。_$$D_{KL}(P \| Q) = \sum_i P(i)ln\frac{P(i)}{Q(i)}$$

_KL Divergence和交叉熵：$$D_{KL}(P \| Q) = -S(P) + H(P, Q)$$_

#### **Logistic Regression的Loss是什么？**
_Negative Log Likelihood._

#### **Logistic Regression的Loss推导**

$$\begin{aligned}
令\beta表示模型参数，x_i表示输入，y_i表示正确结果，\delta = \frac{1}{1 + e^{-\beta^Tx_i}}，则有 \\
p(y_i\vert x_i, \beta) = \delta^{y_i}(1-\delta)^{1-y_i} \\

NLL = -\sum ln(\delta^{y_i}(1-\delta)^{1-y_i}) \\

\frac{\partial NLL}{\partial \beta} = -\sum (y_i \cdot \frac{1}{\delta} \cdot \delta (1-\delta)x_i + (1-y_i) \cdot \frac{1}{1-\delta} \cdot -\delta (1- \delta)x_i )\\
= -\sum(\delta - y_i)x_i \\
\end{aligned}$$

#### **SVM的Loss是什么**
_Hinge Loss. $$L = max(0, 1 - ty)$$，其中t为目标值，取值为1或-1，y为预测值。不鼓励模型过度自信。_

#### **Multiclass Logistic Regression，然后问了一个为什么用cross entropy做Loss function。**
_K个1 vs. others分类器。在Logistics Regression中，优化cross entropy等于优化NLL._

#### **Decision Tree split nodes时的优化目标是啥**
_信息增益。_

#### **DNN为什么要有bias term，bias term的intuition是什么？**
_如果没有bias，则分类超平面则必须经过原点。Bias可以对分类超平面进行平移。_

#### **什么是Back propagation?**
_Back Propagation是指误差反向传播，是一种对神经网络进行梯度下降的算法，即用链式法则计算损失函数对每层权重的梯度，并据此来更新权重。_

#### 梯度消失和梯度爆炸是什么？怎么解决？
_梯度消失和梯度爆炸是训练深层神经网络或者Vanilla RNN时存在的问题。使用Back Propagation算法进行链式求导时，对于靠近输入端的权重，其梯度是每一层权重以及每一层的激活函数的梯度的乘积数，如果梯度较小（如
使用了不合适的激活函数，如Sigmoid，其导数的取值范围是(0, 0.25))，会导致梯度的值逐层减小，最终消失。梯度消失表现为随着模型训练，loss下降很慢，尤其是离输出层近的权重改变幅度大，离输出层远的权重改变幅度小甚至为0。反之，如果梯度较大，会导致
梯度逐层指数增加，产生梯度爆炸，表现为loss值和模型权重剧烈变化，甚至变成NaN。_

_解决梯度消失问题的方法主要有：更换激活函数为Relu或其变种、增加Residual Connection和使用Batch Norm。_

_解决梯度爆炸问题的方法主要有：梯度裁剪、加入正则项、更好的初始化策略等。_

#### **神经网络初始化能不能把weights都设置成0？**
_不能，也不能设为相同的值，否则会导致每一层Hidden Layer中各个节点的值完全相同，根据反向传播算法，每个参数更新的值也相同，无法正常学习特征。_

#### **DNN和Logistic Regression区别**
_可以把Logistic Regression看成一个只有一层的神经网络，只能学习特征各自的权重（有可解释性），并将特征的加权和通过sigmoid函数进行二分类。_

#### **你为什么觉得DNN拟合能力比Logistic Regression强？**
_因为DNN可以学习特征的组合关系，而Logistic Regression只能学习特征各自的权重。_

#### **How to do hyper-parameter tuning in DL?**
_可以通过Grid Search的方法进行，也有一些AutoML的方法，如贝叶斯优化。_

#### **Deep Learning有哪些预防over fitting的方法？**
_Weight Decay、Dropout、Pretrain + Finetune、Early Stopping、数据增强等。_

#### **什么是Dropout？Why it works? Dropout的流程是什么？**
_Dropout是一种在训练深度神经网络时防止过拟合的方法。在训练阶段的前向过程中，通过将hidden layer的值直接设为0的方式，忽略一部分的隐藏层神经元。这部分神经元不产生梯度传播。在推理阶段则保留全部的隐藏层神经元，并将神经元的输出进行Rescale。_

_Dropout有效果，一方面是在训练的时候迫使每个神经元的取值不依赖于任何一个特定的其它神经元，降低了网络的复杂度。如果把不同的dropout看成是不同的子网络，也体现一种集成学习的思想。_

#### **什么是Batch Norm？Why it works? BN的流程是什么？**
_BN是一种用于神经网络的Regularization技术，目的是解决神经网络中的Internal covariate shift的问题，即随着神经网络的训练，权重的分布发生变化，导致输出给下一层的特征的分布发生变化，而这个变化需要下一层的网络权重额外学习。_

_BN包括两个阶段，归一化和scale and shift。它学习两个参数，$$\gamma$$和$$\beta$$。在训练阶段，对每个mini batch的输入，BN层计算其均值和标准差，并将其归一化到标准正态分布。然后通过使用学习到的$$\gamma$$和$$\beta$$进行缩放和平移。_

_在推理阶段，因为只有一个样本的输入，无法计算均值和方差，所以在训练的时候会同时记录训练样本的均值和方差。训练样本的均值和方差一般通过滑动平均的方式来更新。_

#### **common activation functions是什么以及每个的优缺点**
_Sigmoid：可以将输入映射到0和1之间的连续输出，可以解释为概率并用于二分类问题。缺点：容易造成梯度消失，且输出不是以0为中心。_

_tanh：将输入映射到-1到1之间，产生以0为中心的输出。缺点：计算开销大，且输入远离0的时候，梯度也接近0，容易造成梯度消失。_

_ReLU：将负的输入映射为0，正的输入保持不变，简单有效，计算速度快，产生稀疏特征。缺点：输入为负的神经元梯度为0。_

_Leaky ReLU: ReLU的改进，输入为负的时候引入一个小的斜率，可以允许梯度的传播。_

_还有一些基于ReLU的改进，如引入可学习的参数来控制斜率等。_

#### **为什么需要Non-Linear activation functions？**
_如果没有非线性激活函数，则DNN退化成对输入特征的线性变换，不能产生非线性的决策边界。_

#### **Different Optimizers （SGD, RMSProp, Momentum, Adagrad, Adam）的区别**
_SGD是基本的梯度下降算法，每次迭代通过当前样本的梯度更新模型参数。$$\theta_{t+1} = \theta_t - \eta \nabla J(\theta)$$ 。SGD（包括Mini-Batch GD)有两个问题：1.容易陷在局部最小值（saddle point或者plateau)。 2.学习率始终固定。动量方法应对问题1，而自适应学习率方法应对问题2。_

_Momentum在SGD的基础上增加了动量概念，即在参数更新时，既利用当前Batch的梯度，也保留原有的更新方向，即_

$$\begin{aligned}
m_{t+1}=\mu \cdot m_t + \eta \nabla J(\theta); \\
\theta_{t+1} = \theta_t - m_{t+1}
\end{aligned}$$

_Nestrov accelerate gradient是Momentum的变种，具体做法是计算梯度时，先在$$\theta$$上减去上一时刻的动量_

$$\begin{aligned}
m_{t+1}=\mu \cdot m_t + \eta \nabla J(\theta - \mu \cdot m_t); \\
\theta_{t+1} = \theta_t - m_{t+1}
\end{aligned}$$

_Adagrad是一种自适应学习率算法，目的是对学习率进行动态调整。在更新梯度时，对每个参数，将全局学习率除以一个基于梯度累计范数的约束项，这样前期的梯度更新量较大，后期梯度更新量较小。频繁更新的参数梯度更新量较小，稀疏更新的参数梯度更新量较大。_

$$
\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t, i} + \epsilon}} \cdot \nabla J(\theta_{t,i})
$$

_Adagrad的问题是到了训练中后期，因为约束项分母过大，导致参数更新量太小，无法学习。RMSProp将不断累计的梯度更新量之和改为RMS形式的梯度移动平均。_

$$\begin{aligned}
E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1-\rho) \cdot g_t^2 \\
\theta_{t+1} = \theta_{t} - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot \nabla J(\theta_{t})  = \theta_t - \frac{\eta}{RMS[g]_t}\cdot \nabla J(\theta_{t}) 
\end{aligned}$$

_Adadelta将RMSProp中的固定学习率参数$$\eta$$转换为上一时刻的$$\Delta \theta$$的RMS，不需要指定固定的学习率。(Pytorch中的adadelta优化器有学习率参数lr，但它是可选参数，默认值是1.0，相当于对自适应梯度再做一次scale）。_

$$\theta_{t+1} = \theta_t - \frac{RMS[\Delta(\theta)]_{t-1}}{RMS[g]_t}\cdot \nabla J(\theta_{t})$$ 

_Adam优化器可以看作是动量和自适应学习率的结合，类似RMSProp + Momentum。Adam保留过去梯度的指数衰减平均值，也保留梯度的平方的指数衰减平均值：_

$$\begin{aligned}
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \\
v_t = \beta_2 v_{t-1} + (1-\beta_2)g^2_t \\
\hat m_t = \frac{m_t}{1-\beta^t_1} \\
\hat v_t = \frac{v_t}{1-\beta^t_2} \\
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat v_t + \epsilon}} \cdot \hat m_t
\end{aligned}$$

_目前主流的优化器是AdamW，即增加了WeightDecay的Adam优化器。AdamW优化器将参数权重的decay从梯度计算中分离出去，单独基于L2正则化计算权重衰减。_

#### **Batch和SGD的优缺点，Batch size的影响**
_Batch GD（区别于Mini Batch GD）是一次性迭代所有的训练样本，而SGD是每次迭代一个训练样本，Mini-Batch GD是两者的折中，通过超参数Batch Size控制每次迭代的样本数量。_

_BGD优点是较为稳定，缺点是每次迭代完整的数据集，速度较慢，且无法应用到大数据集上；_

_SGD的优点是计算成本低，内存需求小，缺点是受噪声影响大，更新方向的方差较大，收敛过程不稳定且对学习率敏感。_

_Mini Batch作为两者的折中，是主流的方案，可以通过一个mini-batch中的样本近似整体梯度，减少更新方向的方差，更新更稳定，也可以通过并行化来提升计算效率。_

_Batch Size的影响主要是内存/显存的占用量，同时，batch size发生变化时，学习率也要相应调整。_

#### **learning rate过大和过小对模型的影响**
_学习率设置过大时，容易出现梯度爆炸的问题，也容易产生Loss的震荡。学习率设置过小时，会使得模型的收敛速度太慢，且容易陷入局部最小值无法跳出。主流的做法会对学习率进行warmup和衰减。_

#### **Problem of Plateau, Saddle point**
_Problem of Plateau是指目标函数在一个区域内非常平坦，梯度接近0。而saddle point指的是目标函数在某个区域梯度为0，但并非局部最小值（在某些维度上升，某些维度下降）。可以通过基于动量的优化器，以及自适应学习率解决这两个问题。_

#### **When transfer learning makes sense.**
_迁移学习是将在一个任务上学习到的知识（模型参数）应用到另一个任务上，当新的任务有监督数据较少，且和原任务有类似的原始输入特征时有效。如把基于ImageNet训练的CNN权重作为其它分类任务的
初始值，或者NLP领域的Pretrain + Finetune以及Prompt Tuning两个Paradigm，都能看成是迁移学习的成功应用。_