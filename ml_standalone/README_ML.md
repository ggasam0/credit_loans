# ML 单线说明（双口径）

## 1. 评估口径

本目录同时维护两套主口径：

1. 共享子数据集全量（`7108` 行）  
- 输入：`data/shared/shared_subset.csv`  
- 输出：`output/ml_standalone`

2. 原始全量（实际违约率口径，`123202` 行）  
- 输入：`ml_standalone/data/processed/ml_full_processed.csv`  
- 输出：`output/ml_standalone_full_actual_rr`

两套口径均使用：
- 分层切分（`stratified_random_split`）
- 阈值策略 `fixed_reject_rate`
- 指标 `Precision@RR`、`Recall@RR`、`Approval Bad Rate`、`Lift@RR`
- Gains/Lift 曲线（Reject 5% -> 50%）

## 2. 重跑命令

### A. 共享子数据集全量（7108）
```powershell
.\.venv\Scripts\python.exe ml_standalone\run_ml_pipeline.py `
  --input data/shared/shared_subset.csv `
  --output-dir output/ml_standalone `
  --shared-split-dir data/shared/splits/stratified `
  --no-force-rebuild-shared-split `
  --test-size 0.2 `
  --val-size 0.2 `
  --threshold-policy fixed_reject_rate `
  --target-reject-rate 0.35 `
  --no-enable-manual-review `
  --no-balance-train `
  --no-balance-test
```

### B. 原始全量（123202）
```powershell
.\.venv\Scripts\python.exe ml_standalone\run_ml_pipeline.py `
  --input ml_standalone/data/processed/ml_full_processed.csv `
  --output-dir output/ml_standalone_full_actual_rr `
  --shared-split-dir data/shared/splits/full_actual_rr `
  --no-force-rebuild-shared-split `
  --test-size 0.2 `
  --val-size 0.2 `
  --threshold-policy fixed_reject_rate `
  --target-reject-rate 0.15327673252057597 `
  --no-enable-manual-review `
  --no-balance-train `
  --no-balance-test
```

## 3. 数据产物

A 口径：`output/ml_standalone/data`
- `ml_processed.csv`
- `train_fit.csv`
- `validation.csv`
- `test.csv`
- `train_fit_for_training.csv`
- `test_eval_for_metrics.csv`
- `split_summary.json`

B 口径：`output/ml_standalone_full_actual_rr/data`
- `ml_processed.csv` 或 `ml_full_processed.csv`（视流程输入）
- `train_fit.csv`
- `validation.csv`
- `test.csv`
- `train_fit_for_training.csv`
- `test_eval_for_metrics.csv`
- `split_summary.json`

## 4. 结果产物

A 口径：
- `output/ml_standalone/run_report.json`
- `output/ml_standalone/model/model_summary.csv`
- `output/ml_standalone/model/model_predictions.csv`
- `output/ml_standalone/model/logistic_tabular_gains_lift_curve.csv`
- `output/ml_standalone/model/logistic_text_fusion_gains_lift_curve.csv`
- `output/ml_standalone/model/xgboost_tabular_gains_lift_curve.csv`

B 口径：
- `output/ml_standalone_full_actual_rr/run_report.json`
- `output/ml_standalone_full_actual_rr/model/model_summary.csv`
- `output/ml_standalone_full_actual_rr/model/model_predictions.csv`
- `output/ml_standalone_full_actual_rr/model/logistic_tabular_gains_lift_curve.csv`
- `output/ml_standalone_full_actual_rr/model/logistic_text_fusion_gains_lift_curve.csv`
- `output/ml_standalone_full_actual_rr/model/xgboost_tabular_gains_lift_curve.csv`
