# 项目总报告（主目录汇总）

## 1. 评估口径与公平性说明

本项目当前使用两套口径：
1. 共享子数据集口径（`data/shared/shared_subset.csv`，7108 条）
- 目标：跨模型家族公平横向比较（ML / DataAnalysis / BERT / LLM）
- 阈值：验证集固定 `Reject Rate=35%`

2. 全量实际违约率口径（123202 条）
- 目标：评估在接近真实业务分布下的表现
- 阈值：固定 `Reject Rate=0.15327673252057597`
- 当前覆盖：ML + DataAnalysis

关键公平性约束：
- 若实际 Reject Rate 明显偏离目标值（例如单阶段 LLM 和 GPT-5.2），其结果不宜与“已对齐 RR=35%”模型直接做优劣结论。

## 2. 共享子集结果（7108，RR 目标=35%）

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

结论（共享子集）：
1. 在“Reject Rate 接近 35%”的模型里，`DataAnalysis` 取得最高 `Precision(reject)=0.4175`。
2. `ML (logistic_tabular)` 在召回上更强（`Recall(default)=0.4538`），综合更均衡。
3. BERT 两条线均低于 ML/DataAnalysis。
4. LLM 两阶段较稳健但精度仍偏低；LLM 单阶段与 GPT-5.2 因 Reject Rate 偏高，当前不满足同约束公平比较条件。

## 3. 全量实际违约率口径结果（123202，RR 目标=15.33%）

| 模型 | Precision(reject) | Recall(default) | Approval Bad Rate | Lift | 实际 Reject Rate |
|---|---:|---:|---:|---:|---:|
| ML (logistic_tabular) | 0.3069 | 0.2992 | 0.1263 | 2.0023 | 0.1494 |
| ML (logistic_text_fusion) | 0.2994 | 0.3050 | 0.1262 | 1.9532 | 0.1562 |
| ML (xgboost_tabular) | 0.3153 | 0.3095 | 0.1246 | 2.0568 | 0.1505 |
| DataAnalysis | 0.2849 | 0.2851 | 0.1294 | 1.8589 | 0.1534 |

结论（全量口径）：
1. `ML (xgboost_tabular)` 在 Precision/Recall/Lift 上领先。
2. DataAnalysis 作为可解释基线可用，但性能低于 ML 最优。

## 4. 当前推荐

1. 统一子集对比结论：主模型优先 `ML (logistic_tabular)` 与 `DataAnalysis` 双线并行。
2. 若追求最接近真实分布部署效果：优先 `ML (xgboost_tabular)`（全量口径结果最佳）。
3. LLM 若要纳入公平主对比，应先强制拒绝率对齐（例如 top-k RR 控制），再做最终横评。

## 5. 结果来源文件

- ML（共享）：`output/ml_standalone/run_report.json`
- DataAnalysis（共享）：`output/data_analysis_standalone/run_report.json`
- BERT（共享）：`output/bert_standalone/run_report.json`
- LLM 两阶段：`output/llm_two_stage_subset_full_06b/infer/stage2_action_predictions_full.json`
- LLM 单阶段：`output/llm_one_stage_subset_full_06b/infer/stage_action_direct_predictions_full.json`
- GPT-5.2：`output/llm_two_stage_subset_full_06b/infer/gpt52_external_metrics.json`
- ML（全量）：`output/ml_standalone_full_actual_rr/run_report.json`
- DataAnalysis（全量）：`output/data_analysis_standalone_full_actual_rr/run_report.json`
