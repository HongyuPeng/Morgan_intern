# 项目名称 JPMorgan chooser option pricing model

## 项目简介
本项目旨在参考，基于，并改进这篇paper：
https://pdfs.semanticscholar.org/3022/12a91cd5d30878ef132a8636b165b5973c8a.pdf
后续会引入真实股价数据，以及机器学习算法

## 文件结构
```
├── .gitignore                   # Git 忽略文件
├── README.md                    # 项目说明文件
├── Exploration of JPMorgan Chooser Option Pricing.pdf      # paper原文
├── data                         # 数据文件夹
│   ├── input                    # 存放输入数据，例如市场数据
│   └── output                   # 存放输出结果，例如分析图表和模型结果
├── models                       # 存放模型相关文件，如训练好的模型、参数等
├── scalers                      # 存放数据归一化或标准化的工具和参数
├── tests                        # 存放测试代码和相关资源
└── src                          # 项目代码
```

## 项目进度
模型在预测分布上表现出“过度集中”现象：它倾向于输出大量接近 0 的正值，而真实（蒙特卡洛）分布在这一区域的样本密度较低。这说明模型对低收益样本的概率或幅度估计存在系统性偏差。