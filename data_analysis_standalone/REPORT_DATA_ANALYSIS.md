# DataAnalysis 结果报告（2026-02-21 重跑）

## 1. 实验设置

- 输入：`data/shared/shared_subset.csv`
- 切分：共享分层切分（`train_fit=4548`，`val=1138`，`test=1422`）
- 阈值：验证集固定 `Reject Rate=35%`
- 特征：读取 ML 的 `feature_selection.json`，与其它模型维度一致

## 2. 测试集结果（固定 35% 口径）

| 模型 | Precision@35% | Recall@35% | Approval Bad Rate | Lift@35% | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| DataAnalysis | 0.4175 | 0.4317 | 0.3120 | 1.1928 | 0.3622 |

## 3. 结论

- 本轮 `Precision@35%` 在三类可比模型中最高。
- `Recall@35%` 低于 ML 最优模型，但高于 BERT-Embedding。
- 适合作为“规则解释性较强 + 适中拒绝精度”对照线。

## 4. 关键文件

- `output/data_analysis_standalone/run_report.json`
- `output/data_analysis_standalone/model/analysis_predictions.csv`
- `output/data_analysis_standalone/model/analysis_gains_lift_curve.csv`
