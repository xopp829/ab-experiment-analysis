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

## 关键结论

| 分析模块 | 结论 |
|---------|------|
| 假设检验 | 新页面转化率无显著提升（p > 0.05） |
| 辛普森悖论 | 分层后结论与整体一致，无悖论 |
| ML特征重要性 | is_treatment重要性极低，访问耗时是主要驱动因素 |
| HTE分析 | 无显著异质性处理效应 |
| CUPED | 方差缩减有效但未改变结论 |

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
