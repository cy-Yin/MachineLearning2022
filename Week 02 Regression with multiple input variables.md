## Multiple features (variables)

$x_j$ 表示第$j$个特征/变量

$n$ 表示特征/变量的总数（注意与$m$区分，$m$表示单变量中的样本总数，如房子大小共有多少个数据，而$n$指的是例如房子大小、卧室个数、房子使用时间等等变量共有多少个）

$\vec{x}^{(i)}$ 表示第$i$个训练样本的所有变量组成的向量列表，*注意这是一个行向量*

$\vec{x}^{(i)}_j$ 表示第$i$个训练样本的第$j$个变量的值

例如：

|Size in feet^2|Number of bedrooms|Number of floors|Age of home in years|Price($) in $1000's|
|:---:|:---:|:---:|:---:|:---:|
|2104|5|1|45|460|
|1416|3|2|40|232|
|1534|3|2|30|315|
|852|2|1|36|178|
|2023|4|3|38|390|

则
$n=4$，$m=5$，前四列分别为$x_0$、$x_1$、$x_2$、$x_3$，$\vec{x}^{(1)} = [1416,3,2,40]$，$\vec{x}^{(1)}_2 = 2$

Model:
- previously: $f_{w,b}(x) = w * x + b$
- now: $f_{\vec{w},b}(\vec{x}) = w_1 * x_1 + w_2 * x_2 + \cdots + w_n * x_n + b = \vec{w} * \vec{x} + b$，其中$\vec{w} = [w_1, w_2, \cdots, w_n]$，$\vec{x} = [x_1, x_2, \cdots, x_n]$

**Multiple Linear Regression!!!**

### A very useful idea: **Vectorization**

*注意：*
- *linear algebra: count from 1: $x_1$, $x_2$, $x_3$ ...*
- *Python code: count from 0 (offset): `x[0]`, `x[1]`, `x[2]`*

```Python
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10, 20, 30])
```

Without vectorization: $f_{w,b}(x) = (\sum\limits_{i=1}\limits^{n} w_i * x_i) + b$
```Python
f = 0
for j in range(0, n):
	f = f + w[j] * x[j]
f = f + b
```

With vectorization: $f_{\vec{w},b}(\vec{x}) = \vec{w} * \vec{x} + b$
```Python
f = np.dot(w, x) + b
```

![|650](files/Pasted%20image%2020230505232003.png)

Vectorization will make a huge difference in the running time of your learning algorithm.

