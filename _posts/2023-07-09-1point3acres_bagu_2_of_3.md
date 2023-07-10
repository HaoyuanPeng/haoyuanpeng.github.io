---
layout: post
title: 试答一亩三分地上的机器学习八股文 (2 / 3)
date: 2023-07-10 18:00:00 +0800
description: 试答一亩三分地上的机器学习八股文 (2 / 3)
tags: [Machine Learning]
relatedposts: false
---

一亩三分地论坛上有网友总结了常见的机器学习八股文，链接见：

1. [https://www.1point3acres.com/bbs/thread-713903-1-1.html](https://www.1point3acres.com/bbs/thread-713903-1-1.html)
2. [https://www.1point3acres.com/bbs/thread-714090-1-1.html](https://www.1point3acres.com/bbs/thread-714090-1-1.html)
3. [https://www.1point3acres.com/bbs/thread-714558-1-1.html](https://www.1point3acres.com/bbs/thread-714558-1-1.html)


现试答如下，欢迎批评指正。本文为第二部分，共三部分，每一部分与上述三个链接一一对应。

#### **Linear Regression的基本假设是什么？**
1. _自变量X和y线性相关_
2. _每个样本独立同分布_
3. _Residual的均值为0_
4. _Residual的方差在自变量的取值范围内是常数_

#### **What will happen when we have co-related variables, how to solve?**
_模型会对输入x的微小变化过于敏感。如果两个自变量完全线性相关，则无法求解。可以通过Pearson系数或者VIF检测特征之间的相关性，并对特征进行合并或删除；或者使用Ridge、Lasso等正则化。_

#### **Explain regression coefficient**
_其它自变量保持不变时，某个自变量每变化1单位，则因变量的期望值的变化。_

#### **What's the relationship between minimizing squared error and maximizing the likelihood?**
_当噪声满足高斯分布时，最小化MSE等价于最大化Likelihood。_

#### **How could you minimize the inter-correlation between variables with Linear Regression?**
_Feature selection and removing highly-correlated features (or PCA), Regularization (Ridge or Lasso)._

#### **If the relationship between y and x is not linear, can linear regression solve that?**
_传统线性回归不行，但可以在特征中加入其他项，如多项式项、对数项、交叉特征等，使得因变量与输入近似满足线性关系。_

#### **Why use interaction variables?**
_建模输入特征之间的相互作用，这也是非线性关系的一种。同时可以提供更丰富的可解释性。_

#### **K-means clustering: explain the algorithm in detail, whether it will converge, global or local optimums, how to stop?**
_算法：_
1. _选择K个起始点（随机选择或者用Kmeans++选择），作为K个cluster的中心；_
2. _每次迭代：计算每个样本点到各个cluster中心的距离，将其分到对应的中心；_
3. _计算每个cluster的均值，作为新的中心；_
4. _重复2～3步骤，直到每个聚类中心在两次迭代中保持不变，或达到最大迭代次数。_

_K-means会收敛到局部最优，且要求每个cluster是凸的。_

#### **EM算法是什么？**
_EM算法是一种在概率模型包含不能观察的隐变量的情形下，对参数作最大似然估计的算法。算法通过两个步骤交替计算：_
1. _根据参数的假设值，给出未知变量的期望估计，并用在隐变量上；_
2. _根据隐变量的估计值，给出当前参数的极大似然估计。_

#### **GMM是什么，和K-means的关系**
_GMM也是一种聚类算法，假设数据是由多个高斯分布组成的混合模型，每个高斯分布表示一个cluster。算法预测通过EM算法来估计模型参数，并得出每个样本点属于每个高斯分布的概率。GMM不要求每个聚类簇的形状是凸的。_

_K-means可以看作GMM的一个特例，即每个高斯分布的协方差矩阵是各向同性的。_

#### **How regression / classification trees split nodes?**
- _Regression Tree: 一般以MSE为最小化的目标函数。对选定的特征，遍历其不同取值，计算相应的目标函数，并选择能够使其最小化的分裂点。_
- _Classification Tree: 遍历选定的特征的不同取值，计算相应的信息增益（即节点的熵 - 各分裂的子节点的熵的加权平均），选择能够使信息增益最大的分裂点。_

#### **How to prevent over-fitting in DT?**
- _剪枝：综合考虑树的节点个数和目标函数，对树进行剪枝；_
- _提前停止：根据叶节点的样本数量，或者信息增益的值，提前停止分裂；_
- _限制最大深度；_
- _ensemble。_

#### **How to do regularization in DT?**
_同上_

#### **Difference between boosting and bagging**
_Boosting和Bagging都是将弱模型组合成强模型的方法。_
- _Bagging方法中，弱模型并行地独立训练，并在最后预测时通过投票或者均值等方式集成。Bagging旨在降低模型的variance，解决过拟合问题，代表是随机森林。_
- _Boosting中，弱模型串行地组成pipeline。Boosting旨在降低模型的bias，提升模型效果，代表是GBDT。_

#### **GBDT和random forest区别，pros and cons**
_GBDT是典型的Boosting方法，串行地训练N个DT，每个DT拟合之前的DT的残差。随机森林是典型的Bagging方法，通过bootstrapping对训练数据集进行不同的采样，并选择不同的特征子集，并行地同时训练N个DT。_

#### **Will random forest reduce bias or variance? Why?**
_variance。因为随机森林中的每个模型仅使用部分的特征，且通过bootstrap的方法对训练样本进行采样，每个DT的训练样本各不相同。最后，通过投票集成也可以降低variance。_

#### **和Discriminative模型相比，Generative模型更容易over-fitting还是under-fitting?**
_在相同的数据条件下，判别模型只需要建模条件分布，而不需要对完整的数据分布进行建模；同时生成模型为了更准确地拟合训练数据中的分布，更有可能关注到数据的内在结构和噪声，而判别模型只需要关注类别边界。因此，生成模型更容易过拟合。_

#### **Naive-Bayes的原理，基础假设是什么？**
_Naive Bayes的基础假设是特征之间的条件独立，即在给定类别的条件下，各输入特征之间相互独立。_

_Naive Bayes的原理是，利用Bayes定理，基于P(类别）、P(特征)、P(特征\|类别)，计算出P(类别\|特征)。_

#### **LDA/QDA是什么，基础假设是什么？**
_LDA试图找到一个低维空间，使得将不同类别的样本投影到这个空间上时，同类样本之间的距离小，不同类样本之间的距离大，从而可以使用线性分类器进行分类。LDA的基础假设是不同类别的样本的特征的协方差矩阵相同。_

_QDA暂缺。_

#### **Logistic Regression和SVM的差别**
_Logistic Regression采用Log Loss，即最小化Negative Log Likelihood，而SVM采用Hinge Loss。_

#### **Explain SVM，如何引入非线性**
_SVM是一种二分类算法，原理是在特征空间中找到一个最优超平面，将不同类别的样本分隔开来。最优的超平面指的是离最近的样本点最远的超平面，距离超平面最近的样本点称为支持向量。_

_SVM可以通过特征的非线性变换以及引入组合特征来将非线性分类问题转换为线性分类问题，也可以通过核函数将输入特征映射到高维空间来实现非线性划分。_

#### **Explain PCA**
_PCA是一种特征降维的方法。PCA首先对数据进行标准化处理，然后计算协方差矩阵，并对协方差矩阵进行特征值分解，选择具有最大值的前k个特征向量。然后将原始数据通过选取的k个特征向量进行线性变换来映射到k维的低维空间。_

_PCA假设数据的方差集中在少数的k个维度上，要评估数据是否满足这个假设。_

#### **Explain kernel methods, why to use.**
_核函数将输入特征映射到高维空间，从而解决非线性划分的问题。同时，核函数可以计算样本在高维空间的内积，而无需显式计算样本在高维空间的表示。_

_常见的核函数包括多项式核函数、RBF核函数、Sigmoid核函数等。_

#### **怎么把SVM的output按照概率输出？**

_暂缺_

#### **Explain KNN**
_KNN是一种有监督分类算法，通过训练样本中，与输入样本距离最近的k个样本的标签，来对输入样本的标签进行预测。_

#### **怎么处理imbalanced data**
1. _对类别较多的样本进行降采样，对类别较少的样本进行过采样；_
2. _为不同类别在损失函数中设置不同的权重；_
3. _将任务转换为异常检测任务；_
4. _使用集成学习来降低模型对多数样本的过拟合。_
5. _使用合适的评价指标，在不平衡数据上对模型进行评价，如precision、recall、F值等。_

#### **High-dim classification有什么问题，如何处理？**
1. _维度灾难问题：特征选择或降维；_
2. _过拟合问题：正则化、交叉验证、ensemble；_
3. _计算复杂度问题：特征选择或降维。_

#### **Missing data如何处理**
1. _删除有缺失值的样本，用剩下的样本训练模型作为baseline，并评价缺失值填充的效果。_
2. _特殊值填充、平均值填充、众数填充、基于距离最近的k个其他样本赋值、插值、模型预测。_

#### **How to do feature selection**
1. _Pearson相关系数；_
2. _卡方特征选择；_
3. _方差选择；_
4. _基于互信息选择；_
5. _基于L1正则产生稀疏特征；_
6. _使用PCA或者LDA进行降维。_

#### **How to capture feature interaction?**
1. _原始特征的多项式扩展，如x1 * x2, x1^2等；_
2. _交叉特征：如gender_age；_
3. _通过深度学习模型自动学习。_