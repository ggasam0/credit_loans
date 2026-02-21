# BERT 单线说明（共享切分版本）

## 1. 输入与模型

- 输入数据：`data/shared/shared_subset.csv`
- 特征清单：`output/ml_standalone/analysis/feature_selection.json`
- 文本输入：结构化特征 + `desc_clean` 组装为 `bert_input`
- 基座模型：`./models/bce-embedding-base_v1`

## 2. 切分与评估口径

- 切分方式：`stratified_random_split`
- 共享切分目录：`data/shared/splits/stratified`
- 阈值策略：`fixed_reject_rate`
- 目标拒绝率：`Reject Rate=35%`
- 对比分支：
  - `BERT-Embedding`（冻结编码 + 逻辑回归）
  - `BERT-Finetune`（端到端分类头微调）

## 3. 重跑命令

```powershell
.\.venv\Scripts\python.exe bert_standalone\run_bert_pipeline.py `
  --input data/shared/shared_subset.csv `
  --output-dir output/bert_standalone `
  --feature-selection-json output/ml_standalone/analysis/feature_selection.json `
  --shared-split-dir data/shared/splits/stratified `
  --no-force-rebuild-shared-split `
  --test-size 0.2 `
  --val-size 0.2 `
  --threshold-policy fixed_reject_rate `
  --target-reject-rate 0.35 `
  --embedding-model-path .\models\bce-embedding-base_v1 `
  --model-path .\models\bce-embedding-base_v1 `
  --run-embedding `
  --run-finetune `
  --no-balance-test `
  --no-downsample-train-neg
```

## 4. 数据与结果产物

`output/bert_standalone/data`：
- `bert_subset_processed.csv`
- `bert_processed.csv`
- `train_fit.csv`
- `validation.csv`
- `test.csv`
- `split_summary.json`

`output/bert_standalone/embedding_eval`：
- `bert_embedding_eval_report.json`
- `bert_embedding_eval_predictions.csv`
- `bert_embedding_gains_lift_curve.csv`

`output/bert_standalone/finetune_eval`：
- `bert_finetune_eval_report.json`
- `bert_finetune_eval_predictions.csv`
- `bert_finetune_gains_lift_curve.csv`
- `finetuned_model/`
