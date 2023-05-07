输出变量$y$只能取一小部分可能值中的一个，而非像Regression中无限范围的数字

Linear Regression 不是一个解决 Classification 问题的好方法，我们需要用到 逻辑回归（Logistic Regression）。

"binary classifcation" 二元分类：$y$ can only be one of the **two** values

the two classes:
- false, $0$, "negative class"
- ture, $1$, "positive class"

## Logistic Regression

### sigmoid function 

the sigmoid function algorithm (or the logistic function) is
$$g(z) = \frac{1}{1 + e^{-z}}$$

the outputs are between 0 and 1 ($0 < g(z) < 1$)

![|375](files/SigmoidFunction.png)

用sigmoid function实现逻辑回归需要两步：
1. $z = f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$
2. $g(z) = 1 / (1 + e^{-z})$ （这里第二步变为非线性回归）

所以整个逻辑回归的模型$f$为
$$f_{\vec{w},b}(\vec{x}) = g(\vec{w}\cdot\vec{x} + b) = \frac{1}{1 + e^{-(\vec{w}\cdot\vec{x} + b)}}$$

What the logistic regression model does is it inputs features or set of features $X$ and outputs a number between $0$ and $1$.

可以认为逻辑回归模型输出的是概率 the "probability" that class is $1$, given input $\vec{x}$, parameters $\vec{w}$, $b$
$$f_{\vec{w},b}(\vec{x}) = P(y = 1|\vec{x};\vec{w},b)$$

Set $0.5$ as the **threshold**. Is $f_{\vec{w},b}(\vec{x}) \geq 0.5$ (or $z = \vec{w}\cdot\vec{x} + b \geq 0$) ?
- Yes: $\hat{y} = 1$
- No: $\hat{y} = 0$

### Decision boundary 决策边界

Decision boundary 定义为 $z = \vec{w} \cdot \vec{x} + b = 0$

### Cost function for Logistic Regression

recall the cost function for linear regression:
$$J(\vec{w},b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})^2$$
but choose the logistic regression model $f = 1 / [1 + e^{-(\vec{w}\cdot\vec{x} + b)}]$, the cost function will be non-convex(非凸的), and there are lots of local minima that you cannot use gradient descent (fig as follow).

![|500](files/LogisticSquaredErrorCostFunction.png)

重新选择Cost function如下:

首先对单个样本定义**损失函数**(Loss function)：$\mathcal{L}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})$ for the $i^{\text{th}}$ example.

Define the loss on single training set $\mathcal{L}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})$

Choose the form of the **loss function** as
$$
\mathcal{L}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})=
\left
\{
\begin{aligned} 
& -\log(f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if}\ \ y^{(i)} = 1 \\ 
& -\log(1 - f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if}\ \ y^{(i)} = 0
\end{aligned} 
\right.
$$

因为 $0 < f_{\vec{w},b}(\vec{x}) = \frac{1}{1 + e^{-(\vec{w}\cdot\vec{x} + b)}} < 1$，所以 $0 < -\log(f) < +\infty$；
- 当 $y^{(i)} = 1$ 时，由 $y=-\log(x)$ 的函数图像：
	- 在 $f_{\vec{w},b}(\vec{x}^{(i)})$ 很接近$1$时，$L = -\log(f_{\vec{w},b}(\vec{x}^{(i)}))$ 损失大于$0$而很接近$0$
	- 在 $f_{\vec{w},b}(\vec{x}^{(i)})$ 等于$0.5$时，$L = -\log(f_{\vec{w},b}(\vec{x}^{(i)}))$ 损失变大但没那么大
	- 在 $f_{\vec{w},b}(\vec{x}^{(i)})$ 趋于$0$时，$L = -\log(f_{\vec{w},b}(\vec{x}^{(i)}))$ 趋于$+\infty$，变得非常大
	- 即 Loss is lowest when $f_{\vec{w},b}(\vec{x}^{(i)})$ predicts close to true label $y^{(i)}$; and the further prediction $f_{\vec{w},b}(\vec{x}^{(i)})$ is from target $y^{(i)}$, the higher the loss.
- 当 $y^{(i)} = 0$ 时，由 $y=-\log(1-x)$ 的函数图像，同理可得：
	- 在 $f_{\vec{w},b}(\vec{x}^{(i)})$ 很接近$0$时，$L = -\log(f_{\vec{w},b}(\vec{x}^{(i)}))$ 损失大于$0$而很接近$0$
	- 在 $f_{\vec{w},b}(\vec{x}^{(i)})$ 等于$0.5$时，$L = -\log(f_{\vec{w},b}(\vec{x}^{(i)}))$ 损失变大但没那么大
	- 在 $f_{\vec{w},b}(\vec{x}^{(i)})$ 趋于$1$时，$L = -\log(f_{\vec{w},b}(\vec{x}^{(i)}))$ 趋于$+\infty$，变得非常大
	- 即 Loss is lowest when $f_{\vec{w},b}(\vec{x}^{(i)})$ predicts close to true label $y^{(i)}$; and the further prediction $f_{\vec{w},b}(\vec{x}^{(i)})$ is from target $y^{(i)}$, the higher the loss.

![|575](files/LogisticCostFunction.png)

When the true label is $1$, the algorithm is strongly incentivized not to predict something too close to $0$.

在选取合适的Loss function后，定义整体的Cost function如下：
$$J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m}\mathcal{L}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})$$

此时新的Cost function就变为一个平滑的凸函数，易于进行gradient descent

![|625](files/Pasted%20image%2020230507222644.png)

（*注：其中左边是新的Cost function，可以看到非常平滑；右边是由之前的平方误差函数做出的Cost function，波动大，到处是局部最小值，不利于梯度下降*）

### Simplified Cost Function

