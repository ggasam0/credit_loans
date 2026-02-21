# BERT 结果报告（2026-02-21 重跑）

## 1. 实验设置

- 输入：`data/shared/shared_subset.csv`
- 切分：共享分层切分（`train_fit=4548`，`val=1138`，`test=1422`）
- 基座：`./models/bce-embedding-base_v1`
- 阈值：验证集固定 `Reject Rate=35%`

## 2. 测试集结果（固定 35% 口径）

| 模型 | Precision@35% | Recall@35% | Approval Bad Rate | Lift@35% | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| BERT-Embedding | 0.3859 | 0.4177 | 0.3284 | 1.1026 | 0.3790 |
| BERT-Finetune | 0.3895 | 0.4317 | 0.3253 | 1.1128 | 0.3882 |

## 3. 结论

- 本轮结果中，`BERT-Finetune` 在 `Precision@35%` 与 `Recall@35%` 均高于 `BERT-Embedding`。
- 但 BERT 两条线整体仍落后于 ML/DataAnalysis 的拒绝精度。
- BERT 可保留为“文本深层表征补充线”，而非当前最优主模型线。

## 4. 关键文件

- `output/bert_standalone/run_report.json`
- `output/bert_standalone/embedding_eval/bert_embedding_eval_report.json`
- `output/bert_standalone/finetune_eval/bert_finetune_eval_report.json`
