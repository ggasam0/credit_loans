# 基于文本分析的信贷风控对比项目（主目录汇总）

本仓库当前保留四条独立模型线：
- `ml_standalone`
- `data_analysis_standalone`
- `bert_standalone`
- `llm_standalone`

## 1. 环境与依赖

- 操作系统与硬件：`Windows 11`，`NVIDIA RTX 3060`
- Python：`3.13.11`

```powershell
pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

modelscope download --model maidalun/bce-embedding-base_v1 --local_dir ./models/bce-embedding-base_v1
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B

git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e . --no-deps
```

## 2. 数据与切分口径

### 2.1 共享子数据集（跨模型统一对比）
- 原始入口：`data/accepted_2007_to_2018Q4.csv`
- 统一子集：`data/shared/shared_subset.csv`
- 共享切分：`data/shared/splits/stratified`
- 样本：`7108`（`train_fit=4548`，`val=1138`，`test=1422`）
- 评估口径：验证集定阈值，目标 `Reject Rate=35%`

### 2.2 全量实际违约率口径（贴近真实分布）
- 输入：`ml_standalone/data/processed/ml_full_processed.csv`
- 切分：`data/shared/splits/full_actual_rr`
- 样本：`123202`
- 评估口径：目标 `Reject Rate=0.15327673252057597`（全量真实违约率）
- 当前仅 ML 与 DataAnalysis 运行此口径

## 3. 汇总结果

### 3.1 共享子集（7108，RR 目标=35%）

| 模型 | Precision(reject) | Recall(default) | Approval Bad Rate | Lift | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| ML (logistic_tabular) | 0.4154 | 0.4538 | 0.3098 | 1.1870 | 0.3826 |
| ML (logistic_text_fusion) | 0.4154 | 0.4337 | 0.3126 | 1.1868 | 0.3657 |
| ML (xgboost_tabular) | 0.4030 | 0.4378 | 0.3178 | 1.1513 | 0.3805 |
| DataAnalysis | 0.4175 | 0.4317 | 0.3120 | 1.1928 | 0.3622 |
| BERT-Embedding | 0.3859 | 0.4177 | 0.3284 | 1.1026 | 0.3790 |
| BERT-Finetune | 0.3895 | 0.4317 | 0.3253 | 1.1128 | 0.3882 |
| LLM 0.6B 两阶段微调 | 0.3653 | 0.3675 | 0.3420 | 1.0436 | 0.3523 |
| LLM 0.6B 单阶段微调 | 0.3587 | 0.6245 | 0.3369 | 1.0249 | 0.6097 |
| GPT-5.2 未微调 | 0.3462 | 0.8133 | 0.3690 | 0.9884 | 0.8228 |

说明：
- LLM 单阶段与 GPT-5.2 的实际拒绝率明显偏离 35%，因此与其他模型比较时需关注“同 Reject Rate 约束”公平性。

### 3.2 全量实际违约率口径（123202，RR 目标=15.33%）

| 模型 | Precision(reject) | Recall(default) | Approval Bad Rate | Lift | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| ML (logistic_tabular) | 0.3069 | 0.2992 | 0.1263 | 2.0023 | 0.1494 |
| ML (logistic_text_fusion) | 0.2994 | 0.3050 | 0.1262 | 1.9532 | 0.1562 |
| ML (xgboost_tabular) | 0.3153 | 0.3095 | 0.1246 | 2.0568 | 0.1505 |
| DataAnalysis | 0.2849 | 0.2851 | 0.1294 | 1.8589 | 0.1534 |

## 4. 一键/主线运行入口

- ML：`ml_standalone/README_ML.md`
- DataAnalysis：`data_analysis_standalone/README_DATA_ANALYSIS.md`
- BERT：`bert_standalone/README_BERT.md`
- LLM：`llm_standalone/README_LLM.md`

## 5. 对应报告入口

- 总报告：`report.md`
- ML 报告：`ml_standalone/REPORT_ML.md`
- DataAnalysis 报告：`data_analysis_standalone/REPORT_DATA_ANALYSIS.md`
- BERT 报告：`bert_standalone/REPORT_BERT.md`
- LLM 报告：`llm_standalone/REPORT_LLM.md`
