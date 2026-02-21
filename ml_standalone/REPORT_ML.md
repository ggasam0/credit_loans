# ML 结果报告（2026-02-21 重跑）

## 1. 实验设置

- 输入：`data/shared/shared_subset.csv`
- 切分：共享分层切分（`train_fit=4548`，`val=1138`，`test=1422`）
- 阈值：验证集固定 `Reject Rate=35%`
- 候选模型：
  - `logistic_tabular`
  - `logistic_text_fusion`
  - `xgboost_tabular`

## 2. 测试集结果（固定 35% 口径）

| 模型 | Precision@35% | Recall@35% | Approval Bad Rate | Lift@35% | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| logistic_tabular | 0.4154 | 0.4538 | 0.3098 | 1.1870 | 0.3826 |
| logistic_text_fusion | 0.4154 | 0.4337 | 0.3126 | 1.1868 | 0.3657 |
| xgboost_tabular | 0.4030 | 0.4378 | 0.3178 | 1.1513 | 0.3805 |

## 3. 最优模型

- 本轮最优：`logistic_tabular`
- 选择依据：在相同口径下，`Recall@35%` 最高，且 `Precision@35%` 与第二名几乎持平。

## 4. 关键文件

- `output/ml_standalone/run_report.json`
- `output/ml_standalone/model/model_summary.csv`
- `output/ml_standalone/model/model_predictions.csv`
