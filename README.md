# JPMorgan Chooser Option Pricing Model

## 项目简介

本项目基于JPMorgan选择权期权定价研究论文，结合传统金融工程与深度学习技术，开发了创新的选择权期权定价模型。项目不仅实现了论文中的理论框架，还引入了以下关键改进：

- **历史股价数据集成**：使用真实市场数据增强模型泛化能力
- **机器学习算法应用**：采用多层感知机(MLP)实现高效定价
- **蒙特卡洛模拟优化**：基于布莱克-斯科尔斯模型的改进模拟方法

> 参考论文：[JPMorgan Chooser Option Pricing](https://pdfs.semanticscholar.org/3022/12a91cd5d30878ef132a8636b165b5973c8a.pdf)

## 项目特点

### 🚀 创新定价模型
- **传统与AI融合**：结合金融工程理论与深度学习技术
- **高精度定价方案**：相比传统方法显著提升定价准确性
- **实时推理能力**：生产级部署支持毫秒级响应

### 📊 数据多样性
- **合成数据生成**：大规模蒙特卡洛模拟生成训练数据
- **历史数据增强**：集成美国银行股历史股价数据
- **多市场适应性**：支持不同市场条件下的定价预测

### 🔧 技术先进性
- **深度特征工程**：提取路径依赖、波动率特征、技术指标等多维度特征
- **概率性输出框架**：提供定价不确定性量化
- **自动化工作流**：完整的训练、验证、部署流水线

## 核心功能

### 数据生成模块
- **随机参数空间**：大规模蒙特卡洛模拟
- **历史数据模式**：基于真实股价数据的数据生成
- **多资产支持**：扩展至美国银行板块股票数据

### 机器学习模型
- **多层感知机(MLP)**：深度神经网络架构
- **模型优化**：自动超参数调优
- **概率预测**：输出置信区间和不确定性估计

### 生产级部署
- **实时推理**：毫秒级定价响应
- **性能监控**：自动化模型评估与监控
- **可扩展架构**：支持模型版本管理和更新

## 性能指标

- **验证集精度**：R² ≥ 0.9900
- **推理速度**：毫秒级响应 (< 10ms)
- **效率提升**：相比传统蒙特卡洛加速1000+倍
- **泛化能力**：在历史数据测试集上保持高精度

## 技术栈

### 核心框架
- **TensorFlow/Keras** - 深度学习框架
- **Scikit-learn** - 数据预处理和模型评估
- **NumPy/SciPy** - 数值计算和随机模拟

### 数据处理
- **Pandas/Numpy** - 历史数据分析和处理
- **Matplotlib** - 数据可视化和分析

### 开发工具
- **Python 3.8+** - 主要编程语言
- **Git** - 版本控制

## 文件结构
```bash
MORGAN_INTERN/
├── data/                                # 数据存储目录
│   ├── input excel/                     # 输入Excel文件目录
│   ├── alt_bank/                        # 替代银行历史数据
│   │   ├── DGS1.csv                     # 美国1年期国债收益率数据
│   │   ├── JPM_Dividend.csv             # 摩根大通股息数据
│   │   └── JPM.csv                      # 摩根大通股价历史数据
│   └── output images/                   # 输出图像目录
│       ├── 2023-2024/                   # 特定年份分析图像
│       ├── ml_analysis/                 # 机器学习分析结果图像
│       └── paper_reproduction/          # 论文复现相关图像
├── models/                              # 预训练模型存储
│   ├── chooser_option_mlp_model_mixed.h5              # 混合数据训练的MLP模型
│   └── chooser_option_mlp_model_synthetic_only.h5     # 纯合成数据训练的MLP模型
├── scalers/                             # 数据标准化器存储
│   ├── X_scaler_mixed.pkl               # 混合数据特征标准化器
│   └── X_scaler_synthetic_only.pkl      # 纯合成数据特征标准化器
└── src/                                 # 源代码目录
    ├── __init__.py                      # Python包初始化文件
    ├── main.py                          # 主程序
    ├── BSM_ML.py                        # 布莱克-斯科尔斯机器学习主程序
    ├── BSM_non_path_dependent.py        # 非路径依赖期权定价实现
    ├── BSM_path_dependent.py            # 路径依赖期权定价实现
    ├── param_calculator.py              # 模型参数计算工具
    └── validation.py                    # 模型验证和测试模块
├── .gitignore                          # Git版本控制忽略规则
├── archive.7z                          # 项目归档文件
├── Exploration of JPMorgan Chooser Option Pricing.pdf  # 参考论文原文
├── README.md                           # 项目说明文档
├── requirements.txt                    # 项目依赖文件，列出所需的Python库及其版本
```

## 🛠 环境准备

### 1️⃣ 克隆项目
```bash
git clone <repository-url>
cd MORGAN_INTERN
```
### 2️⃣ 安装依赖
请使用 Python 3.8+（推荐 3.8–3.11），然后执行：

```bash
pip install -r requirements.txt
```
如使用 GPU：

```bash
pip install tensorflow-gpu
```
如遇依赖版本冲突，可使用：

```bash
pip install -r requirements.txt --no-cache-dir --upgrade
```

## 📁 数据准备

### 历史数据格式要求
- **股价数据** (如 JPM.csv): 包含 `Date`, `Open`, `High`, `Low`, `Close`, `Volume` 列
- **股息数据** (如 JPM_Dividend.csv): 包含 `Date`, `Dividend` 列  
- **利率数据** (如 DGS1.csv): 包含 `Date`, `Rate` 列

### 数据源
- [美联储经济数据 (FRED)](https://fred.stlouisfed.org/)
- [Investing.com](https://www.investing.com/)
- [macrotrends](https://www.macrotrends.net/)

## 🚀 快速开始

### 1️⃣ 训练模型
#### 使用混合数据训练模型
`python src/main.py --mode train --data_type mixed`
#### 使用增强历史数据训练
`python src/main.py --mode train --data_type historical --enhanced`
#### 自定义训练路径数量
`python src/main.py -m train -d synthetic -e -n 500000`
### 2️⃣ 模型评估
#### 默认评估混合数据模型
`python src/main.py --mode eval --data_type mixed`

#### 指定模型文件进行评估
`python src/main.py --mode eval --model_path models/chooser_option_mlp_model_mixed.h5`

#### 自定义评估路径数
`python src/main.py -m eval --model_path my_model.h5 --eval_paths 1000000`

### 3️⃣ 实时预测
#### 基础预测示例
`python src/main.py --mode predict --model_path models/chooser_option_mlp_model_mixed.h5 --s0 100 --K 95 --r 0.05 --q 0.02 --sigma 0.2 --T 1.0 --choice_date 0.5`

#### 使用真实市场参数预测
`python src/main.py --mode predict --model_path models/chooser_option_mlp_model_mixed.h5 --s0 156.7 --K 150 --r 0.0015 --q 0.0233 --sigma 0.282 --T 1.0 --choice_date 0.5`

#### 高精度预测（增加蒙特卡洛路径）
`python src/main.py --mode predict --model_path models/chooser_option_mlp_model_mixed.h5 --s0 100 --K 95 --r 0.05 --q 0.02 --sigma 0.2 --T 1.0 --choice_date 0.5 --n_paths_predict 500000`
    
## 🔧 参数说明

#### 训练模式参数
| 参数                | 简写   |  必需 | 说明                                   | 默认值    |
| ----------------- | ---- | :-: | ------------------------------------ | ------ |
| `--mode`          | `-m` |  ✅  | 运行模式（train）                          | -      |
| `--data_type`     | `-d` |  ✅  | 数据类型（synthetic / historical / mixed） | -      |
| `--enhanced`      | `-e` |  ❌  | 启用历史数据增强                             | False  |
| `--n_total_paths` | `-n` |  ❌  | 总训练路径数                               | 400000 |

#### 评估模式参数
| 参数             |  必需 | 说明         | 默认值    |
| -------------- | :-: | ---------- | ------ |
| `--mode`       |  ✅  | 运行模式（eval） | -      |
| `--model_path` |  ❌  | 模型文件路径     | 自动检测   |
| `--data_type`  |  ❌  | 模型训练数据类型   | mixed  |
| `--eval_paths` |  ❌  | 评估路径数      | 500000 |

#### 预测模式参数
| 参数                  |  必需 | 说明            | 默认值    |
| ------------------- | :-: | ------------- | ------ |
| `--mode`            |  ✅  | 运行模式（predict） | -      |
| `--model_path`      |  ✅  | 模型文件路径        | -      |
| `--s0`              |  ✅  | 初始股价          | -      |
| `--K`               |  ✅  | 执行价           | -      |
| `--r`               |  ✅  | 无风险利率         | -      |
| `--q`               |  ✅  | 股息率           | -      |
| `--sigma`           |  ✅  | 波动率           | -      |
| `--T`               |  ✅  | 到期时间（年）       | -      |
| `--choice_date`     |  ✅  | 选择日（年）        | -      |
| `--n_paths_predict` |  ❌  | 蒙特卡洛路径数       | 100000 |

## 📤 输出文件说明
| 模式     | 输出内容                       |
| ------ | ---------------------------- |
| **训练** | 模型文件（.h5）、标准化器（.pkl）、训练曲线图 |
| **评估** | 评估图像、MAE / RMSE / R² 性能指标  |
| **预测** | 定价对比曲线、蒙特卡洛 vs MLP预测结果     |

## 💡 使用建议

* 推荐训练模式：`--data_type mixed`
* 历史增强模式可提高模型 鲁棒性
* 最佳训练路径数：400k ~ 1M
* 预测时确保 `choice_date < T`

## 🧪 性能基准
| 指标         | 结果           |
| ---------- | ------------ |
| 验证集 R²     | ≥ **0.9900** |
| 单次预测用时     | **< 10 ms**  |
| 相比直接蒙特卡洛加速 | **1000x+**   |
| 历史数据泛化误差   | 低且稳定         |

## 🆘 故障排查
| 问题         | 解决方案                    |
| ---------- | ----------------------- |
| 模型加载失败     | 检查 TensorFlow 版本 & 模型路径 |
| Scaler 不匹配 | 请保持模型与对应 scaler 配套使用    |
| 选择时间大于到期时间 | 保证 `choice_date < T`    |

## 📄 许可证

本项目仅用于 **学术研究和个人学习**，禁止直接用于商业用途。  
如需商业授权，请联系作者获取许可。

## 🤝 贡献

欢迎提交 Issue / Pull Request 来改进模型、优化数据管线或扩展部署能力。

建议提交流程：
1. Fork 本仓库
2. 创建新分支： `git checkout -b feature-xyz`
3. 提交修改： `git commit -m "Add feature xyz"`
4. 推送分支并提交 PR： `git push origin feature-xyz`

## 📬 联系方式

如有讨论、合作或研究需求，欢迎联系：

- Email: *hongyustevenpeng@gmail.com*
- GitHub: [*HongyuPeng*](https://github.com/HongyuPeng/)
