# LLM 两阶段模型报告

测试口径：`1173` 条（与其他模型一致）。

## 优化后主结果（不重训，仅阈值调优）

- 文件：`output/llm_two_stage_subset_full_06b/infer/stage2_action_predictions_full_thr1252.json`
- `reject_delta_threshold`: `1.251953`

指标：

- `Precision(reject)`: `0.3846`
- `Recall(default)`: `0.1862`
- `Reject Rate`: `0.1552`
- `Accuracy`: `0.6436`

## 对比说明

阈值调优后，相比默认阈值版本显著提高了 Precision，并降低了拒绝率，但 Recall 同时下降。  
该方案目前处于“Precision 与 Recall 平衡”区间，整体仍低于 ML 主线的 Precision。
