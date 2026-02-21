# DataAnalysis 结果报告（双口径）

## 1. 实验设置

### 口径 A：共享子数据集全量
- 输入：`data/shared/shared_subset.csv`
- 样本量：`7108`
- 切分：`train_fit=4548`，`val=1138`，`test=1422`
- 阈值：验证集固定 `Reject Rate=35%`

### 口径 B：原始全量（实际违约率）
- 输入：`ml_standalone/data/processed/ml_full_processed.csv`
- 样本量：`123202`
- 切分：`train_fit=78848`，`val=19713`，`test=24641`
- 阈值：验证集固定 `Reject Rate=0.15327673252057597`（全量实际违约率）

## 2. 结果对比

### A. 共享子数据集全量（RR=35%）

| 模型 | Precision@35% | Recall@35% | Approval Bad Rate | Lift@35% | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| DataAnalysis | 0.4175 | 0.4317 | 0.3120 | 1.1928 | 0.3622 |

### B. 原始全量（RR=15.33%）

| 模型 | Precision@RR | Recall@RR | Approval Bad Rate | Lift@RR | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| DataAnalysis | 0.2849 | 0.2851 | 0.1294 | 1.8589 | 0.1534 |

## 3. 结论

1. 共享子数据集（用于与 ML/BERT/LLM 统一横向比较）下，DataAnalysis 的 `Precision@35%` 表现较强。
2. 原始全量分布下，DataAnalysis 的 `Lift@RR` 为 `1.8589`，可作为“规则可解释基线”。
3. 两套口径不能直接比较绝对值，应按场景分别使用：
- 口径 A：统一子集横向对比。
- 口径 B：贴近全量业务分布评估。

## 4. 关键文件

口径 A：
- `output/data_analysis_standalone/run_report.json`
- `output/data_analysis_standalone/model/analysis_predictions.csv`
- `output/data_analysis_standalone/model/analysis_gains_lift_curve.csv`

口径 B：
- `output/data_analysis_standalone_full_actual_rr/run_report.json`
- `output/data_analysis_standalone_full_actual_rr/model/analysis_predictions.csv`
- `output/data_analysis_standalone_full_actual_rr/model/analysis_gains_lift_curve.csv`
