# DataAnalysis 单线说明（共享切分版本）

## 1. 输入与特征

- 输入数据：`data/shared/shared_subset.csv`
- 特征清单：`output/ml_standalone/analysis/feature_selection.json`
- 与 ML/BERT 对齐：使用同一批结构化字段

## 2. 切分与评估口径

- 切分方式：`stratified_random_split`
- 共享切分目录：`data/shared/splits/stratified`
- 阈值策略：`fixed_reject_rate`
- 目标拒绝率：`Reject Rate=35%`
- 指标：`Precision@35%`、`Recall@35%`、`Approval Bad Rate`、`Lift@35%`

## 3. 重跑命令

```powershell
.\.venv\Scripts\python.exe data_analysis_standalone\run_data_analysis_classifier.py `
  --input data/shared/shared_subset.csv `
  --output-dir output/data_analysis_standalone `
  --feature-selection-json output/ml_standalone/analysis/feature_selection.json `
  --shared-split-dir data/shared/splits/stratified `
  --no-force-rebuild-shared-split `
  --test-size 0.2 `
  --val-size 0.2 `
  --threshold-policy fixed_reject_rate `
  --target-reject-rate 0.35
```

## 4. 数据与结果产物

`output/data_analysis_standalone/data`：
- `analysis_processed.csv`
- `train_fit.csv`
- `validation.csv`
- `test.csv`
- `split_summary.json`

`output/data_analysis_standalone/model`：
- `analysis_predictions.csv`
- `analysis_gains_lift_curve.csv`

`output/data_analysis_standalone/analysis`：
- `feature_profiles.csv`
- `feature_profiles.json`
