# ML 单线说明（共享切分版本）

## 1. 输入与切分

- 输入数据：`data/shared/shared_subset.csv`
- 切分方式：`stratified_random_split`
- 共享切分目录：`data/shared/splits/stratified`
- 共享切分复用参数：
  - `--shared-split-dir data/shared/splits/stratified`
  - `--no-force-rebuild-shared-split`（复用）
  - `--force-rebuild-shared-split`（重建）

## 2. 评估口径

- 阈值策略：`fixed_reject_rate`
- 目标拒绝率：`Reject Rate=35%`
- 指标：`Precision@35%`、`Recall@35%`、`Approval Bad Rate`、`Lift@35%`
- 输出曲线：`Reject 5% -> 50%` 的 Gains/Lift 曲线

## 3. 重跑命令

```powershell
.\.venv\Scripts\python.exe ml_standalone\run_ml_pipeline.py `
  --input data/shared/shared_subset.csv `
  --output-dir output/ml_standalone `
  --shared-split-dir data/shared/splits/stratified `
  --force-rebuild-shared-split `
  --test-size 0.2 `
  --val-size 0.2 `
  --threshold-policy fixed_reject_rate `
  --target-reject-rate 0.35 `
  --threshold-objective precision_at_reject_rate_bounds `
  --no-enable-manual-review `
  --no-balance-train `
  --no-balance-test
```

## 4. 数据产物

`output/ml_standalone/data` 下保留：
- `ml_processed.csv`
- `train_fit.csv`
- `validation.csv`
- `test.csv`
- `train_fit_for_training.csv`
- `test_eval_for_metrics.csv`
- `split_summary.json`

## 5. 结果产物

- `output/ml_standalone/run_report.json`
- `output/ml_standalone/model/model_summary.csv`
- `output/ml_standalone/model/model_predictions.csv`
- `output/ml_standalone/model/logistic_tabular_gains_lift_curve.csv`
- `output/ml_standalone/model/logistic_text_fusion_gains_lift_curve.csv`
- `output/ml_standalone/model/xgboost_tabular_gains_lift_curve.csv`
