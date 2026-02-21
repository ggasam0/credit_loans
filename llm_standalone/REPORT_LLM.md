# REPORT_LLM

## 实验范围
本报告汇总 3 条 LLM 主线结果：
- GPT-5.2 未微调推理（外部推理）
- Qwen3-0.6B 微调两阶段（Reason + Action）
- Qwen3-0.6B 微调单阶段（Direct Action）

## 数据与评估口径
- 数据来源：共享子数据集（分层切分，不使用时间切分）。
- 指标口径：`Precision(reject)`、`Recall(default)`、`Reject Rate`、`Approval Bad Rate`、`Lift`。
- 0.6B 两条主线使用 `fixed_reject_rate_eval`（目标 RR=0.35）口径字段；其中单阶段当前未能达到 0.35。
- GPT-5.2 使用 `data/loan_decisions.csv` 与共享测试集按 `_shared_row_id` 对齐，样本数与本地模型一致（1422）。

## 结果表

| 模型 | 样本数 | Precision(reject) | Recall(default) | Reject Rate | Approval Bad Rate | Lift |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.2 未微调 | 1422 | 0.3462 | 0.8133 | 0.8228 | 0.3690 | 0.9884 |
| Qwen3-0.6B 两阶段微调 | 1422 | 0.3653 | 0.3675 | 0.3523 | 0.3420 | 1.0436 |
| Qwen3-0.6B 单阶段微调 | 1422 | 0.3587 | 0.6245 | 0.6097 | 0.3369 | 1.0249 |

## 分析结论
1. 在 1422 同口径下，`Precision(reject)` 最高的是 0.6B 两阶段（0.3653），单阶段次之（0.3587），GPT-5.2 较低（0.3462）。
2. GPT-5.2 召回最高（0.8133），但代价是极高拒绝率（0.8228）和更高通过坏账率（0.3690），业务可用性较弱。
3. 0.6B 两阶段在“拒绝率接近 35%”条件下表现更均衡（Precision 0.3653，Recall 0.3675）。
4. 0.6B 单阶段仍然偏激进（Reject Rate 0.6097），当前阈值控制策略需继续优化。

## 主要问题
1. 单阶段 `target_reject_rate=0.35` 未达成（实际 0.6097），说明当前概率分数离散/集中，量化阈值不能稳定控制 RR。
2. GPT-5.2 当前策略过于激进（Reject Rate 0.8228），缺少对业务目标拒绝率的显式约束。

## 下一步建议
1. 对单阶段推理改为“严格 top-k 拒绝率控制”以保证 RR 精确对齐 35%。
2. 为 GPT-5.2 增加同样的拒绝率约束后再与本地模型做公平对比。
3. 保留两阶段与单阶段双方案：两阶段偏稳健，单阶段偏召回，根据业务策略选用。

## 指标来源文件
- GPT-5.2：`output/llm_two_stage_subset_full_06b/infer/gpt52_external_metrics.json`
- 0.6B 两阶段：`output/llm_two_stage_subset_full_06b/infer/stage2_action_predictions_full.json`
- 0.6B 单阶段：`output/llm_one_stage_subset_full_06b/infer/stage_action_direct_predictions_full.json`
