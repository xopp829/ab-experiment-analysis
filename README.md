# AB Experiment Analysis

AB Test Analysis with Statistical & ML Methods

基于统计推断与机器学习增强的A/B实验全流程分析项目，包含完整的统计检验、辛普森悖论检查、功效分析、HTE异质性处理效应分析及CUPED方差缩减。

## 项目背景

对电商网站新页面设计进行A/B实验评估，数据集约29万条记录，目标为验证新页面是否显著提升用户转化率。

## 分析框架

```
数据校验 → 假设检验 → 辛普森悖论检查 → 决策耗时分析 → 功效分析 → ML增强分析
```

## 项目结构

```
├── notebooks/
│   ├── ab_statistical.py      # 统计分析全流程
│   └── ab_ml_enhanced.py      # ML增强分析（LR/RF/XGBoost/HTE/CUPED）
├── pyspark/
│   ├── ab_pyspark_analysis.py  # PySpark版AB实验分析
│   └── README.md               # PySpark教程说明
├── docs/
│   ├── methodology.md          # AB实验标准流程与方法论
│   └── interview_prep.md       # 面试项目串联话术
├── .gitignore
├── requirements.txt
└── README.md
```

## 技术栈

- **统计分析**：pandas, scipy, statsmodels
- **机器学习**：scikit-learn, xgboost
- **大数据**：PySpark (DataFrame API / Spark SQL / 分区优化)
- **可视化**：matplotlib, seaborn

## 核心结论

| 维度 | 结论 |
|------|------|
| 假设检验 | 新页面转化率无显著差异（p=0.8827），95%CI上界仅+0.08% |
| 辛普森悖论 | UK方向反转（实验组更高），US占70%样本主导整体结论 |
| 决策耗时 | 两组无显著差异，新页面未缩短用户决策时间 |
| 功效分析 | MDE=1%时需样本量差8倍，功效严重不足 |
| 逻辑回归 | 处理效应OR≈1.000，新页面不影响转化概率 |
| 随机森林 | is_treatment重要性极低，访问耗时是关键驱动 |
| XGBoost | 交互项增益低，无显著异质性处理效应 |
| HTE | 分群体CATE均不显著，新页面不对任何群体更有效 |
| CUPED | 用ML预测值缩减方差，展示工业界前沿方法 |

**最终决策**：不上线新页面，停止当前实验。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行统计分析
python notebooks/ab_statistical.py

# 运行ML增强分析
python notebooks/ab_ml_enhanced.py
```

> 注：原始数据未包含在仓库中，请自行准备AB实验数据集。

## 作者

谢书棋 - 中山大学数学学院 统计学专业
