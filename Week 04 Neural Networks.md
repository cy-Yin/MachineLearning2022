Neural networks algorithms originally try to mimic the brain.

A **layer** is a grouping of neurons which takes us input the same or similar features and that in turn outputs a few numbers together.

A layer can have multiple neurons or a single neuron.

- 第一个输入样本值的层称为**输入层**。
- 中间层也称为**隐藏层**（中间层可以有很多层）。
- 最后一个神经网络层也叫**输出层**。

每一层的输出值也叫**激活值**(activation values)。

神经网络中每一层的每个神经元可以访问上一层传递给该层的所有参数/变量的值

每一层的输入、输出的一个或者多个参数可以并在一起以**向量化**的形式表示。

“多层感知机”

神经网络的例子：

- Face recognition
![|625](files/FaceRecognition.png)

- Car classification
![|625](files/CarClassification.png)

## Neural Network Layer

在神经网络的每一层中，有一个或多个神经元，每一个神经元实现一个小的逻辑回归单元。
每一个神经元有参数$\vec{w}_i$和$b_i$，它的作用是输出一些激活值 $a_i = g(\vec{w}_i \cdot \vec{x} + b_i)$，该层的神经元的激活值 $a_1, a_2, \dots, a_k$（$k$为该层神经元的个数）组成输出值向量，传入下一层。

*以后统一用上标 (^\[n]) 表示第几层。输入层为第0层，中间的隐藏层从第1层开始进行标注。
例如，对于隐藏层第1层的第2个神经元，其参数为$\vec{w}_2^{[1]}$和$b_2^{[1]}$，其输出的激活值为 $a_2^{[1]} = g(\vec{w}_2^{[1]} \cdot \vec{a}^{[0]} + b_2^{[1]})$*

*注：对于隐藏层第1层，其输入值就是输入层的值，也就是样本的值$\vec{x}$，我们以后用$\vec{a}^{[0]}$表示$\vec{x}$*。

第$l$层的输入为第$l-1$层的输出$\vec{a}^{[l-1]}$，则第$l$层的第$j$个神经元的激活值为
$$a_j^{[l]} = g(\vec{w}_j^{[l]} \cdot \vec{a}^{[l-1]} + b_j^{[l]})$$

这里 $g()$ 是Sigmoid函数，$g(z) = 1 / (1 + e^{-z})$，深度学习中有时也称为**激活函数**(activation function)。

![|675](files/NeuralNetworks.png)

![|425](files/ComplexNeuralNetworks.png)

*我们说这个神经网络有四层。在计算神经网络的层数时，不算入输入层，只计算隐藏层和输出层。*

**激活函数**：输出激活值的函数。*目前为止只见到了Sigmoid函数*

## Inference: making predictions (forward propagation 前向传播)

```
layer 1 --> layer 2 --> layer 3 --> ......
```

以 手写数字识别 为例（暂时只识别0和1）

![|625](files/HandwrittenDigitRecognition.png)
![|625](files/HandwrittenDigitRecognition2.png)
![|625](files/HandwrittenDigitRecognition3.png)

$a_\text{output} = f(\vec{x}_\text{input})$，用 $f$ 来表示线性回归或逻辑回归的输出。

