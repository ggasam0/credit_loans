# 项目总报告（2026-02-21 重跑）

## 1. 实验设置

- 数据集：`data/shared/shared_subset.csv`
- 总样本量：`7108`
- 切分方式：`stratified_random_split`
- 切分规模：`train_fit=4548`，`validation=1138`，`test=1422`
- 阈值策略：在验证集按 `Reject Rate=35%` 定阈值，固定阈值用于测试集
- 对比模型：
  - ML（`logistic_tabular` 最优）
  - DataAnalysis
  - BERT-Embedding
  - BERT-Finetune

## 2. 共享切分一致性校验

基于 `_shared_row_id` 对比三条流水线导出的 `train_fit/validation/test`：
- `ml_standalone`：一致
- `data_analysis_standalone`：一致
- `bert_standalone`：一致

结论：三者使用的是同一套样本切分，结果可直接横向比较。

## 3. 测试集结果（固定 Reject Rate=35% 口径）

| 模型 | Precision@35% | Recall@35% | Approval Bad Rate | Lift@35% | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| ML（logistic_tabular） | 0.4154 | 0.4538 | 0.3098 | 1.1870 | 0.3826 |
| DataAnalysis | 0.4175 | 0.4317 | 0.3120 | 1.1928 | 0.3622 |
| BERT-Embedding | 0.3859 | 0.4177 | 0.3284 | 1.1026 | 0.3790 |
| BERT-Finetune | 0.3895 | 0.4317 | 0.3253 | 1.1128 | 0.3882 |

## 4. 结论

- `Precision@35%` 最优：`DataAnalysis`（0.4175），略高于 ML。
- `Recall@35%` 最优：`ML`（0.4538）。
- BERT 两条线中：`BERT-Finetune` 的 `Recall@35%` 高于 `BERT-Embedding`，但 `Precision@35%` 仍低于 ML 与 DataAnalysis。
- 如果业务以“拒绝精度优先”为主，本轮最优是 `DataAnalysis`；若兼顾更多违约召回，本轮最优是 `ML`。

## 5. 产物路径

- `output/ml_standalone/run_report.json`
- `output/data_analysis_standalone/run_report.json`
- `output/bert_standalone/run_report.json`
