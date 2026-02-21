# 基于文本分析的信贷风控对比项目（主线）

本仓库当前保留四条独立模型线：
- `ml_standalone`
- `data_analysis_standalone`
- `bert_standalone`
- `llm_standalone`

本次重跑范围是三条可直接对比线：`ML + DataAnalysis + BERT`。

## 0. 环境与依赖（Windows 11 + RTX 3060）

- 操作系统与硬件：`Windows 11`，`NVIDIA RTX 3060`
- Python 版本：`Python 3.13.11`

```powershell
pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

modelscope download --model maidalun/bce-embedding-base_v1 --local_dir ./models/bce-embedding-base_v1
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B

git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e . --no-deps 
```

## 1. 统一数据与切分

- 原始数据入口：`data/accepted_2007_to_2018Q4.csv`
- 统一子数据集：`data/shared/shared_subset.csv`
- 共享切分目录：`data/shared/splits/stratified`
- 共享行标识：`_shared_row_id`

本轮切分统计（来自 `output/ml_standalone/run_report.json`）：
- 全量样本：`7108`
- `train_fit`：`4548`
- `validation`：`1138`
- `test`：`1422`
- 违约率（full/train_fit/val/test）：`0.3500 / 0.3500 / 0.3497 / 0.3502`

说明：三条流水线均复用同一份 `train_fit/validation/test`，并通过 `_shared_row_id` 校验为完全一致。

## 2. 本次统一评估口径

- 切分方式：`stratified_random_split`
- 阈值策略：`fixed_reject_rate`
- 目标拒绝率：`Reject Rate = 35%`（验证集定阈值）
- 测试集核心指标：
  - `Precision@35%`
  - `Recall@35%`
  - `Approval Bad Rate`
  - `Lift@35%`

## 3. 重跑命令

### 3.1 ML（先重建共享切分）

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

### 3.2 DataAnalysis（复用共享切分）

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

### 3.3 BERT（Embedding + Finetune，复用共享切分）

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

## 4. 本次结果总览（测试集）

| 模型 | Precision@35% | Recall@35% | Approval Bad Rate | Lift@35% | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| ML（logistic_tabular） | 0.4154 | 0.4538 | 0.3098 | 1.1870 | 0.3826 |
| DataAnalysis | 0.4175 | 0.4317 | 0.3120 | 1.1928 | 0.3622 |
| BERT-Embedding | 0.3859 | 0.4177 | 0.3284 | 1.1026 | 0.3790 |
| BERT-Finetune | 0.3895 | 0.4317 | 0.3253 | 1.1128 | 0.3882 |

### 4.1 全量数据结果（RR=全量数据实际违约率）

| 模型 | Precision@RR | Recall@RR | Approval Bad Rate | Lift@RR | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| ML (logistic_tabular) | 0.3069 | 0.2992 | 0.1263 | 2.0023 | 0.1494 |
| ML (logistic_text_fusion) | 0.2994 | 0.3050 | 0.1262 | 1.9532 | 0.1562 |
| ML (xgboost_tabular) | 0.3153 | 0.3095 | 0.1246 | 2.0568 | 0.1505 |
| DataAnalysis | 0.2849 | 0.2851 | 0.1294 | 1.8589 | 0.1534 |

## 5. 详细报告入口

- 总报告：`REPORT.md`
- ML：`ml_standalone/README_ML.md`，`ml_standalone/REPORT_ML.md`
- DataAnalysis：`data_analysis_standalone/README_DATA_ANALYSIS.md`，`data_analysis_standalone/REPORT_DATA_ANALYSIS.md`
- BERT：`bert_standalone/README_BERT.md`，`bert_standalone/REPORT_BERT.md`
