## Tensorflow implementation

假设我们要辨别手写数字0和1。

我们构建的神经网络如下图所示。

![|350](files/TrainANeuralNetworkInTensorflow.png)

我们首先给出代码，然后逐步解释其中的细节。

```Python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy


model = Sequential([
	Dense(units=25, activation='sigmoid'),
	Dense(units=15, activation='sigmoid'),
	Dense(units=1, activation='sigmoid')
])

model.compile(loss=BinaryCrossentropy())

model.fit(X, Y, epochs=100)
```

这里我们引入`BinaryCrossentropy()`函数（二元分类交叉熵）作为loss function参与model的compile过程。
这个函数就是之前对于二元分类问题定义的损失函数loss function
$$\mathcal{L}(f_{\vec{w},b}(\vec{x}^{(i)}),y^{(i)})=
-y^{(i)} * \log(f_{\vec{w},b}(\vec{x}^{(i)})) -(1-y^{(i)}) * \log(1 - f_{\vec{w},b}(\vec{x}^{(i)}))$$

“交叉熵”

对于回归问题Regression(predicting numbers and not categories)，通常使用平方误差函数。其在Tensorflow中可以用`MeanSquaredError()`

```Python
from tensorflow.keras.losses import MeanSquaredError

model.compile(loss=MeanSquaredError())
```

在`fit`函数中，我们指定`epochs=100`，即想要运行100步梯度下降算法

![|700](files/ModelTrainingSteps.png)

在梯度下降算法中，最主要的一点就是计算偏导项。在Tensorflow中，我们使用**反向传播**(back-propagation)算法进行偏导项的计算。

## Activation function 激活函数

### Alternatives to the sigmoid activation

神经网络中，除了Sigmoid函数，还有一个常用的激活函数
$$
g(z) = \max(0, z) = 
\left
\{
\begin{aligned} 
& 0 & \text{if}\ \ z < 0 \\ 
& z & \text{if}\ \ z \geq 0
\end{aligned} 
\right.
$$

这个激活函数被命名为**ReLU**函数，是rectified linear unit的缩写。

也有**线性激活函数**（其实相当于没有用任何激活函数）：
$$g(z) = z$$

### How to choose different activation functions

Choosing $g(z)$ for output layer（现在先讨论最后一层输出层的激活函数）
- 如果要处理输出为0或1的**二元分类问题**，选择 **Sigmoid函数**
- 如果要处理回归问题，要预测**输出是正还是负**，选择 **线性激活函数**
- 如果要处理回归问题，但是**输出的值是非负数**，选择 **ReLU函数**

对于中间的隐藏层，**ReLU函数**是迄今为止神经网络中最常见的选择，除非二元分类问题才会用到**Sigmoid函数**，但即使是二元分类问题隐藏层也是用**ReLU**更多。

*因为Sigmoid函数在$z$很大和$z$很小时都会变化很缓慢，导致Cost function在进行梯度下降时效率很低，而ReLU函数只在$z<0$时变化缓慢，所以在进行梯度下降时速度会更快*

### Why do we need activation function?

激活函数的作用是**非线性输出**，如果没有激活函数，所有的神经元做的只是线性变换，则只能应对线性回归模型，且整个神经网络与只用一个神经元本质上没有区别（因为矩阵乘法）。

Don't use linear activation function in hidden layers (using ReLU activation function should do just fine) !!!

## Softmax

### Multi-class 多分类问题

多分类问题指分类问题可以有两个以上的输出标签，而不仅仅是0或1。例如，分辨数字（0到9，而不仅仅是0和1）。

Softmax Regression 是logistic regression的推广，它可以用来解决多分类问题，划分两个以上的类别的决策边界。

