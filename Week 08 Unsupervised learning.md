- Unsupervised Learning
	- Clustering 聚类算法
	- Anomaly detection 异常检测

## Clustering 聚类算法

在 supervised learning 中，我们有一个包含 input features $x$ 和 the labels $y$ 的数据集； 
然而，在 unsupervised learning 中，我们得到的数据集只有 $x$ ，而没有目标标签 $y$ 。

Clustering means looking at the dataset $\{x^{(1)}, x^{(2)}, x^{(3)}, \cdots, x^{(m)}\}$ and trying to see if it can be grouped into clusters.

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