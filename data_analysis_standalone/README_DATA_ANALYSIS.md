# DataAnalysis 单线说明（双口径）

## 1. 评估口径

本目录同时维护两套口径：
1. 共享子数据集全量（`7108` 行）
- 输入：`data/shared/shared_subset.csv`
- 输出：`output/data_analysis_standalone`

2. 原始全量（实际违约率口径，`123202` 行）
- 输入：`ml_standalone/data/processed/ml_full_processed.csv`
- 输出：`output/data_analysis_standalone_full_actual_rr`

两套口径统一使用：
- 切分方式：`stratified_random_split`
- 阈值策略：`fixed_reject_rate`
- 指标：`Precision@RR`、`Recall@RR`、`Approval Bad Rate`、`Lift@RR`

## 2. 重跑命令

### A. 共享子数据集全量（RR=35%）

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

### B. 原始全量（RR=全量实际违约率）

```powershell
.\.venv\Scripts\python.exe data_analysis_standalone\run_data_analysis_classifier.py `
  --input ml_standalone/data/processed/ml_full_processed.csv `
  --output-dir output/data_analysis_standalone_full_actual_rr `
  --feature-selection-json output/ml_standalone_full_actual_rr/analysis/feature_selection.json `
  --shared-split-dir data/shared/splits/full_actual_rr `
  --no-force-rebuild-shared-split `
  --test-size 0.2 `
  --val-size 0.2 `
  --threshold-policy fixed_reject_rate `
  --target-reject-rate 0.15327673252057597
```

## 3. 数据与结果产物

A 口径：`output/data_analysis_standalone/data`
- `analysis_processed.csv`
- `train_fit.csv`
- `validation.csv`
- `test.csv`
- `split_summary.json`

A 口径：`output/data_analysis_standalone/model`
- `analysis_predictions.csv`
- `analysis_gains_lift_curve.csv`

A 口径：`output/data_analysis_standalone/analysis`
- `feature_profiles.csv`
- `feature_profiles.json`

B 口径：`output/data_analysis_standalone_full_actual_rr/data`
- `analysis_processed.csv`
- `train_fit.csv`
- `validation.csv`
- `test.csv`
- `split_summary.json`

B 口径：`output/data_analysis_standalone_full_actual_rr/model`
- `analysis_predictions.csv`
- `analysis_gains_lift_curve.csv`

B 口径：`output/data_analysis_standalone_full_actual_rr/analysis`
- `feature_profiles.csv`
- `feature_profiles.json`
