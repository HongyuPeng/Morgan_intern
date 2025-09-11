# 项目名称 JPMorgan chooser option pricing model

## 项目简介
本项目旨在参考，基于，并改进这篇paper：
https://pdfs.semanticscholar.org/3022/12a91cd5d30878ef132a8636b165b5973c8a.pdf
后续会引入真实股价数据，以及机器学习算法

## 文件结构
```
├── .gitignore                                              # Git 忽略文件
├── README.md                                               # 项目说明文件
├── Exploration of JPMorgan Chooser Option Pricing.pdf      # paper原文
├── fig/                                                    # 数据图集
└── BSM.py                                                  # 源代码
```

## 项目进度
论文都复现完成了，现在有两个问题：
一是变量分析的几张图有噪声没有论文中那么干净
二是Sensitivity analysis on the strike price这张图有很大的问题，如果将BSM.py的23行中的sp替换为150则可以跑出和论文中一致的图（保存在fig/Sensitivity Analysis on the Strike Price_WRONG.png中）而现有的BSM.py则会跑出fig/Sensitivity Analysis on the Strike Price.png，明显不一致
