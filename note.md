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
