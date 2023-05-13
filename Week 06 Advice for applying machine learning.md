
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