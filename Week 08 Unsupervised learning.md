- Unsupervised Learning
	- Clustering 聚类算法
	- Anomaly detection 异常检测

## Clustering 聚类算法

在 supervised learning 中，我们有一个包含 input features $x$ 和 the labels $y$ 的数据集； 
然而，在 unsupervised learning 中，我们得到的数据集只有 $x$ ，而没有目标标签 $y$ 。

Clustering means looking at the dataset $\{x^{(1)}, x^{(2)}, x^{(3)}, \cdots, x^{(m)}\}$ and trying to see if it can be grouped into clusters. ^6819ae

Clustering 算法的相关应用
- Grouping similar news
- Market segmentation
- DNA analysis
- Astronomical analysis

### K-means K-均值算法

直观理解 K-means 做了什么（以数据集中有两个cluster为例）
1. 随机猜测两个簇的中心（又称为簇质心，cluster centroids）分别在哪里 
2. 猜测完质心的坐标后，遍历所有的样本数据点，检测每一个样本数据点更加接近哪个质心并进行标记
3. 将样本数据点分配给距离它最近的质心，最终形成两个簇
4. 分别查看两个簇中所有的数据点并对它们分别取均值操作，得到两个坐标，这就是两个簇的新的质心位置，将原来的簇质心移动到新的位置
5. 重复步骤 2-4，不断更新数据点的标记（也即更新簇），并不断调整簇质心的位置，直至簇和质心不再变化。

伪代码如下：
```
Randomly initialize K cluster centroids mu_1, mu_2, ... , mu_K
Repeat{
	# Assign points to cluster centroids
	for i = 1 to m: # m as the number of examples
		c_i := index (from 1 to K) of cluster centroid closest to x_i # x_i as the i_th example
	# Move cluster centroids
	for k = 1 to K:
		mu_k := average (mean) of points assigned to cluster k
}
```

如果在某一轮分给一个簇质心的点的个数为0，一个通常的做法是消除这个簇，将簇的总数从 $k$ 个减少为 $k-1$ 个。

#### K-means optimization objective

The K-means algorithm is also optimizing a special cost function.

- $c^{(i)}$ $=$ index of cluster ($1, 2, \cdots, K$) to which example $x^{(i)}$ is currently assigned 
- $\mu_k$ $=$ cluster centroid $k$

Combine two of these:
- $\mu_{c^{(i)}}$ $=$ cluster centroid of cluster to which example $x^{(i)}$ has been assigned 
$$
\begin{gather}
&& &x^{(i)} &\longrightarrow &c^{(i)} &\longrightarrow &\mu_{c^{(i)}} &&\\
&& &\text{example} &\longrightarrow &\text{cluster} &\longrightarrow &\text{cluster centroid} && \\
\end{gather}
$$

Cost function of K-means algorithm
$$
J(c^{(1)}, c^{(2)}, \cdots, c^{(m)}; \mu_1, \mu_2, \cdots, \mu_K) = \frac{1}{m}\sum_{i=1}^{m}||x^{(i)} - \mu_{c^{(i)}}||^2
$$ 
即每个数据点到其簇质心的平均距离。

K-means 算法的本质就是在寻找 $\mathop{\rm{min}}\limits_{c^{(1)}, c^{(2)}, \cdots, c^{(m)};\atop \mu_1, \mu_2, \cdots, \mu_K} J(c^{(1)}, c^{(2)}, \cdots, c^{(m)}; \mu_1, \mu_2, \cdots, \mu_K)$

Cost function $J(c^{(1)}, c^{(2)}, \cdots, c^{(m)}; \mu_1, \mu_2, \cdots, \mu_K)$ 有时也叫做**失真函数**(Distortion function)。

Cost function 是否处于下降是检验K-means算法是否收敛的一个好方法。

#### Initializing K-means

在初始时，如何猜测簇质心的位置十分重要，它决定了算法的计算时间和收敛速度。

- Choose $K < m$ 这样才能形成簇
- Randomly pick $K$ training examples
- Set $\mu_1, \mu_2, \cdots, \mu_K$ equal to these $K$ examples 即将$K$个随机选出的样本点作为初始化的簇质心

有时$K$个簇质心的初始化选择可能导致K-means算法收敛到局部最小值，而无法得到最终的结果。
这时一个解决办法就是多次初始化簇的质心并运行K-means算法，比较得出是否找到全局最小值还是仍然找的是局部最小值。

伪代码如下：
```
# Random Initialization

For i = 1 to T # T as number of times you want K-means algorithm to run; T = 50 ~ 1000
{
	Randomly initialize K-means.
	Run K-means. Get c_1, c_2, ... , c_m; mu_1, mu_2, ... , mu_K
	Compute cost function (distortion) J
}
Pick set of clusters that gave lowest cost J
```

#### Choosing the number of Clusters

