# ML 结果报告（双口径）

## 1. 实验设置

### 口径 A：共享子数据集全量
- 输入：`data/shared/shared_subset.csv`
- 样本数：`7108`
- 切分：`train_fit=4548`，`val=1138`，`test=1422`
- 阈值：固定目标拒绝率 `RR=0.35`

### 口径 B：原始全量（实际违约率）
- 输入：`ml_standalone/data/processed/ml_full_processed.csv`
- 样本数：`123202`
- 切分：`train_fit=78848`，`val=19713`，`test=24641`
- 阈值：固定目标拒绝率 `RR=0.15327673252057597`（全量真实违约率）

候选模型（两套口径一致）：
- `logistic_tabular`
- `logistic_text_fusion`
- `xgboost_tabular`

## 2. 结果对比

### A. 共享子数据集全量（RR 目标=35%）

| 模型 | Precision@RR | Recall@RR | Approval Bad Rate | Lift@RR | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| logistic_tabular | 0.4154 | 0.4538 | 0.3098 | 1.1870 | 0.3826 |
| logistic_text_fusion | 0.4154 | 0.4337 | 0.3126 | 1.1868 | 0.3657 |
| xgboost_tabular | 0.4030 | 0.4378 | 0.3178 | 1.1513 | 0.3805 |

### B. 原始全量（RR 目标=15.33%）

| 模型 | Precision@RR | Recall@RR | Approval Bad Rate | Lift@RR | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| logistic_tabular | 0.3069 | 0.2992 | 0.1263 | 2.0023 | 0.1494 |
| logistic_text_fusion | 0.2994 | 0.3050 | 0.1262 | 1.9532 | 0.1562 |
| xgboost_tabular | 0.3153 | 0.3095 | 0.1246 | 2.0568 | 0.1505 |

## 3. 结论

1. 口径 A（7108）下，`logistic_tabular` 综合最优（同精度下召回更高）。
2. 口径 B（123202）下，`xgboost_tabular` 综合最优（Precision/Recall/Lift 均领先）。
3. 两套口径不可直接横向比较绝对值，应分别用于：
- A：跨模型家族（ML/BERT/DataAnalysis/LLM）统一对比
- B：更接近真实业务分布的全量表现评估

## 4. 关键文件

口径 A：
- `output/ml_standalone/run_report.json`
- `output/ml_standalone/model/model_summary.csv`
- `output/ml_standalone/model/model_predictions.csv`

口径 B：
- `output/ml_standalone_full_actual_rr/run_report.json`
- `output/ml_standalone_full_actual_rr/model/model_summary.csv`
- `output/ml_standalone_full_actual_rr/model/model_predictions.csv`
