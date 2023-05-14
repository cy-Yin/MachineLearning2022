
some advice on how to build machine learning systems

![|600](files/DebuggingALearningAlgorithm.png)
![|600](files/MachineLearningDiagnostic.png)

## Evaluating and choosing models

### Evaluating a model

将数据划分为两部分：**训练集**(training set)和**测试集**(test set)

一般training set可以取到70%，test set可以取到30%，或者80%-20%的比例。

用 $m_\text{train}$ 表示训练集的样本数量，用 $m_\text{test}$ 表示测试集的样本数量

用 $x^{(i)}$ 表示第$i$个训练集样本（$i = 1, 2, \dots, m_\text{train}$），用 $x_\text{test}^{(i)}$ 表示第$i$个测试集样本（$i = 1, 2, \dots, m_\text{test}$）

1. Train/Test procedure for linear regression (with squared error cost)
- Fit parameters by minimizing cost function $J(\vec{w},b)$
$$
\min_{\vec{w},b} J(\vec{w},b) = \min_{\vec{w},b} [\frac{1}{2m_\text{train}}\sum_{i=1}^{m_\text{train}}(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m_\text{train}}\sum_{j=1}^{n}w_j^2]
$$
- Compute test error:
$$J_\text{test}(\vec{w},b) = \frac{1}{2m_\text{test}}\sum_{i=1}^{m_\text{test}}(f_{\vec{w},b}(\vec{x}^{(i)}_\text{test}) - y^{(i)}_\text{test})^2$$
- Compute training error:
$$J_\text{train}(\vec{w},b) = \frac{1}{2m_\text{train}}\sum_{i=1}^{m_\text{train}}(f_{\vec{w},b}(\vec{x}^{(i)}_\text{train}) - y^{(i)}_\text{train})^2$$

对于overfitting的model
- $J_\text{train}(\vec{w},b)$ will be very low
- $J_\text{test}(\vec{w},b)$ will be high

在出现这种情况时，就需要对model进行调整

2. Train/Test procedure for classification problem
- Fit parameters by minimizing cost function $J(\vec{w},b)$, E.g.,
$$
J(\vec{w},b) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)} * \log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1 - y^{(i)}) * \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2
$$
- Compute test error:
$$J_\text{test}(\vec{w},b) = -\frac{1}{m_\text{test}}\sum_{i=1}^{m_\text{test}}[y^{(i)}_\text{test} * \log(f_{\vec{w},b}(\vec{x}^{(i)}_\text{test})) + (1 - y^{(i)}_\text{test}) * \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}_\text{test}))]$$
- Compute training error:
$$J_\text{train}(\vec{w},b) = -\frac{1}{m_\text{train}}\sum_{i=1}^{m_\text{train}}[y^{(i)}_\text{train} * \log(f_{\vec{w},b}(\vec{x}^{(i)}_\text{train})) + (1 - y^{(i)}_\text{train}) * \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}_\text{train}))]$$

评估model好坏的方法与上面一致。

对于classification问题，还有一种评估的方法，就是计算 the fraction of the test set and the fraction of the train set that the algorithm has misclassified.
$$
\hat{y} =
\left
\{
\begin{aligned} 
& 1 & \text{if}\ \ f_{\vec{w},b}(\vec{x}^{(i)}) \geq 0.5 \\ 
& 0 & \text{if}\ \ f_{\vec{w},b}(\vec{x}^{(i)}) < 0.5
\end{aligned} 
\right. 
$$

Then count $\hat{y} \neq y$

$J_\text{test}(\vec{w},b)$ is the fraction of the test set that has been misclassified.

$J_\text{train}(\vec{w},b)$ is the fraction of the training set that has been misclassified.

### Model selection and training / cross validation / test sets

Once parameters $\vec{w}, b$ are fit to the training set, the training error $J_\text{train}(\vec{w},b)$ is likely lower than the actual generalization error.

$J_\text{test}(\vec{w},b)$ is better estimate of how well the model will generalize to new data than $J_\text{train}(\vec{w},b)$.

The problem is that $J_\text{test}(\vec{w},b)$ is likely to be an optimistic estimate of generalization error.

![|650](files/ModelSelection.png)

因为测试集也只是整个数据集的一小部分，所以测试集给出的最优解也会产生误差，此时需要采用一个更加合理的方法。

因为选择的model，比如多项式的阶数$d$，和$\vec{w}, b$一样也是一个需要学习的参数，这个参数$d$原本是靠test set找到的，无法再用test set来评估$d$的好坏，所以容易出现overfit。
此时需要引入一个新的子集。

将整个数据集分为三部分：**训练集**(training set)、**交叉验证集**(cross-validation set)、**测试集**(test set)

![|650](files/TrainingSetAndCrossValidationSetAndTestSet.png)

交叉验证集 是指一个额外的数据集，用于检查不同模型的有效性和真实性。

*交叉验证集有时候也叫做**验证集**(validation set)或者**开发集**(development set or dev set for short)*

- Training error
$$J_\text{train}(\vec{w},b) = \frac{1}{2m_\text{train}}\sum_{i=1}^{m_\text{train}}(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2$$
- Cross validation error
$$J_\text{cv}(\vec{w},b) = \frac{1}{2m_\text{cv}}\sum_{i=1}^{m_\text{cv}}(f_{\vec{w},b}(\vec{x}^{(i)}_\text{cv}) - y^{(i)}_\text{cv})^2$$
- Test error
$$J_\text{test}(\vec{w},b) = \frac{1}{2m_\text{test}}\sum_{i=1}^{m_\text{test}}(f_{\vec{w},b}(\vec{x}^{(i)}_\text{test}) - y^{(i)}_\text{test})^2$$


在有了交叉验证集后，对于不同的模型，首先用训练集得到最优的参数，然后分别带入交叉验证集计算Cost function，比较不同的模型，选择有着最低的交叉验证误差的model。

最后，如果要estimate the generalization error of how well this model will do on new data，即评估这个模型的泛化能力，此时就需要选择测试集进行此操作。

*可以理解为多分出来了一个训练集来训练新的参数，在这里就是和model有关的多项式的阶数$d$*

在选择神经网络的结构时同理，可以用Cross validation set来进行neural network architecture的选择。

总结：先训练集对不同的模型训练出最优参数，然后验证集使用这些参数，挑出有最小的交叉验证误差的模型作为最优的模型，最后测试集用训练集给出的参数具体地评估该模型的泛化能力。
 
*好比三个人参加比赛，训练集就是训练三个人的比赛能力到最优，验证集就是赛前模拟测试决定最佳人选去参加比赛，测试集就是正式参加比赛量化能力值*

## Bias and variance

### Diagnosing bias and variance

high bias -- underfitting:
- $J_\text{train}$ is high
- $J_\text{cv}$ is high
high variance -- overfitting:
- $J_\text{train}$ is low
- $J_\text{cv}$ is high
just right:
- $J_\text{train}$ is low
- $J_\text{cv}$ is low

![|600](files/BiasAndVariance.png)

有时也会出现 high bias and high variance 的情况，此时
- $J_\text{train}$ will be high
- $J_\text{cv} \gg J_\text{train}$

### Regularization and bias / variance

![|600](files/RegularizationAndBiasOrVariance.png)

Choosing the regularization parameter $\lambda$

![|600](files/ChoosingRegularizationParameter.png)

![|600](files/BiasAndVarianceAsFunctionOfRegularizationParameter.png)
