## 杂记

### SCANPY

一个基于python的生物数据分析工具库。[论文](https://doi.org/10.1186/s13059-017-1382-0)，[github](https://github.com/scverse/scanpy)

### Annotated Data (anndata, .h5ad文件)

一种用于存储高维矩阵数据及其注释信息的数据结构（[github](https://github.com/scverse/anndata)），常用于存储细胞信息。类定义：
```python
class anndata.AnnData(X=None, obs=None, var=None, uns=None, *, obsm=None, varm=None, layers=None, raw=None, dtype=None, shape=None, filename=None, filemode=None, asview=False, obsp=None, varp=None, oidx=None, vidx=None)
```
细胞数据中的主要属性：
* `X`为主矩阵，是一个稀疏矩阵，形状 $\mathbb{R}^{c\times g}$，其中 $c$ 代表细胞数，$g$ 代表基因数
* `obs`为细胞注释，$c\times n$ 的矩阵，其中 $n$ 为细胞属性个数
* `var`为基因注释，长度为 $g$ 的列表

读取anndata的样例代码：
```python
import scanpy as sc
import numpy as np

raw_data = sc.read("../data/Tosches_turtle.h5ad", index_col=0)

# 将空值转换为0
processed_data = np.nan_to_num(raw_data.X, nan=0.0)
raw_data.X = processed_data
label = raw_data.obs['celltype']
```

### 高变基因

在不同细胞中表达量具有显著差异的基因。通常采用基因表达水平的变异系数（coefficient of variation）来判断高变基因。

记 $y_i\in \mathbb{R}^m$ 是由基因 $i$ 在各个细胞中表达值构成的 $m$ 维向量，则基因 $i$ 的变异系数 $cv_i$ 为：
$$
cv_i = \frac{\text{std}(y_i)}{\text{mean}(y_i)}
$$

### 细胞聚类评价指标

#### NMI (归一化互信息)

衡量聚类结果与真是标签之间的信息一致性。

给定真实细胞划分 $U$，对于聚类结果 $V$，其互信息为
$$
MI(U, V) = \sum_{u\in U}\sum_{v\in V}p(u, v)\log\frac{p(u, v)}{p(u), p(v)}
$$

归一化
$$
NMI(U, V) = \frac{2\cdot MI(U, V)}{H(U)+H(V)}
$$
其中 $H(\cdot)$ 为信息熵。

可调用sickit-learn中的`sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method)`计算（见 [doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html)）

#### ARI (调整兰德指数)

衡量细胞对层面上的聚类一致性。

$$
ARI = \frac{RI - E(RI)}{\max(RI) - E(RI)}
$$
其中 $RI$ 为兰德指数，$E(\cdot)$ 表示期望。

可调用sickit-learn中的`sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)`计算（见 [doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)）

#### ACC (聚类准确率)

衡量聚类标签与真实标签在最佳匹配下的一致比例。

由于聚类后的类标签是无意义的，因此需要将聚类后的类标签映射到真实标签再进行计算。给定真实标签 $U$，对于聚类结果 $V$，定义准确率
$$
ACC = \frac{1}{N}\sum_{i=1}^N \mathbf{1}\{u_i=\text{map}(v_i)\}
$$
其中 $\text{map}(\cdot)$ 为将聚类簇映射到真实细胞类型的最佳映射函数，通常可基于匈牙利算法得到。

### 传统聚类方法

KNN建图+Leiden算法进行图聚类。参考[doc](https://www.sc-best-practices.org/cellular_structure/clustering.html?utm_source=chatgpt.com#)。