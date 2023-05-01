## 什么是机器学习

在进行特定编程的情况下，给予计算机学习能力的领域。 —— Arthur Samual

主要两种类型的机器学习算法
- supervised learning
- unsupervised learning 

## 监督学习 supervised learning

X --> Y mappings
input --> output label

learns from being given "right answers"

![|500](files/supervised%20learning%20input%20to%20output.png)

我们数据集中的每个样本都有相应的“正确答案”，再根据这些样本作出预测

### Regression 回归 

- predict a number 
- infinitely many possible numbers

### Classification 分类

- predict categories
- small number of possible outputs

*术语class/category都表示类别*

## 无监督学习 unsupervised learning

Given data is not associated with any output labels $Y$

无监督学习中没有任何的标签

我们已知数据集，却不知如何处理，也未告知每个数据点是什么

Our job is to find some structure or some pattern or just find some interesting in the unlabeled data.

**聚类算法** clustering

针对数据集，无监督学习就能判断出数据有两个不同的聚集簇。这是一个，那是另一个，二者不同。无监督学习算法可能会把这些数据分成两个不同的簇。所以叫做聚类算法。

