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
