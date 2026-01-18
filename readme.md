# suda Machine Learning 2025 Fall Final

## 题目

单细胞转录组聚类分析

## 背景

单细胞转录组测序技术能够在单个细胞水平上解析基因表达模式，为理解细胞异质性、发育轨迹及疾病机制提供了前所未有的视角。细胞聚类是单细胞数据分析的核心步骤之一，其目标是将具有相似表达模式的细胞归为同一群体，从而识别潜在的细胞类型或状态。聚类分析不仅有助于发现新的细胞亚群，还能为后续的细胞注释、差异表达分析和功能研究奠定基础。

## 数据集

[Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris](./reference/s41586-018-0590-4.pdf)

[Evolution of pallium, hippocampus, and cortical cell types revealed by single-cell transcriptomics in reptiles](./reference/science.aar4237.pdf)

[百度网盘链接](https://pan.baidu.com/s/1n9cgGDf3gGvZ16pdgnr7ng) 提取码：ruca

## 要求

请基于上述数据集，使用Python编程语言，完成单细胞转录组数据的聚类分析。你需要对数据进行预处理（如低质量数据过滤、高变异基因选择、归一化等），随后设计一种聚类方法对细胞进行分组，并利用已知的细胞类型标签评估聚类效果（评估指标为：NMI、ARI及ACC）。最终需提交一份简要报告，包括方法描述、结果分析与可视化。

## 安装

项目运行环境：`Python==3.11.14, pytorch==2.5.0, pytorch-cuda=12.4`

将仓库克隆到本地

```bash
git clone https://github.com/BlueberryOreo/Machine-Learning-suda-2025-fall-Course-Report.git

cd Machine-Learning-suda-2025-fall-Course-Report
```

推荐创建虚拟环境运行本项目代码。

```bash
conda create -n scrna python=3.11
conda activate scrna
```

安装依赖

```bash
pip install -r requirements.txt
```

配置环境变量（Linux系统可选）

```bash
cd code
source setup.sh
```

## 运行

### 数据预处理

下载数据集（[百度网盘链接](https://pan.baidu.com/s/1n9cgGDf3gGvZ16pdgnr7ng) 提取码：ruca）到 `data/` 目录下。目录结构：

```
data/
├── download.txt
├── Quake_Diaphragm.h5ad
├── Quake_Lung.h5ad
└── Tosches_turtle.h5ad
```

运行数据预处理代码：

```bash
# linux
cd code

bash scripts/linux/preprocess_<turtle/lung/diaphragm>.sh
```

```bat
:: windows
cd code

scripts/windows/preprocess_<turtle/lung/diaphragm>.bat
```

数据将存储到 `data/` 目录下，以 `_processed.h5ad` 为后缀。

### 训练 & 测试

项目采用自编码器（Auto Encoder）进行细胞特征提取，并使用对比学习的方式辅助自编码器的学习。提取细胞特征后采用kNN建立图，并使用Leiden算法进行聚类。建议使用GPU进行训练。

预训练模型权重：[百度网盘链接](https://pan.baidu.com/s/1U6OrrGTXWxl--tsgvZkqZw) 提取码 gajz

```bash
cd code

# train
python train.py --config configs/<turtle/lung/diaphragm>_ae.yaml --out_dir output/<turtle/lung/diaphragm>

# eval
python train.py --config configs/<turtle/lung/diaphragm>_ae.yaml --out_dir output/<turtle/lung/diaphragm> --eval --resume path/to/pretrained/model
```

或直接运行脚本进行训练和测试：

```bash
# linux
cd code

bash scripts/linux/train_ae_<turtle/lung/diaphragm>.sh
```

```bat
:: windows
cd code

scripts/windows/train_ae_<turtle/lung/diaphragm>.bat
```

结果将输出到 `--out_dir` 指定的目录。
