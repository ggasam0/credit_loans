# LLM 实验说明（3 条主线）

## 主线列表
- `GPT-5.2 未微调推理`：外部大模型直接在共享子数据集测试集上推理。
- `Qwen3-0.6B 微调两阶段`：阶段1 `Input->Reason`，阶段2 `Input+Reason->Action`。
- `Qwen3-0.6B 微调单阶段`：直接 `Input->Action`。

## 统一数据与切分
- 输入数据：`output/bert_standalone/data/bert_subset_processed.csv`（含 `bert_input`）。
- 切分方式：复用共享分层切分 `data/shared/splits/stratified`。
- 不使用时间切分。

## 运行脚本

### 0.6B 两阶段
- 脚本：`llm_standalone/run_llm_two_stage_full_06b.ps1`
- 命令：
```powershell
powershell -ExecutionPolicy Bypass -File llm_standalone\run_llm_two_stage_full_06b.ps1 `
  -ReasonEpochs 2 `
  -ActionEpochs 4 `
  -InferBatchSize 1 `
  -TargetRejectRate 0.35
```
- 结果：
  - `output/llm_two_stage_subset_full_06b/infer/stage2_action_predictions_full.csv`
  - `output/llm_two_stage_subset_full_06b/infer/stage2_action_predictions_full.json`

### 0.6B 单阶段
- 脚本：`llm_standalone/run_llm_one_stage_full_06b.ps1`
- 命令：
```powershell
powershell -ExecutionPolicy Bypass -File llm_standalone\run_llm_one_stage_full_06b.ps1 `
  -ActionEpochs 4 `
  -InferBatchSize 1 `
  -TargetRejectRate 0.35
```
- 结果：
  - `output/llm_one_stage_subset_full_06b/infer/stage_action_direct_predictions_full.csv`
  - `output/llm_one_stage_subset_full_06b/infer/stage_action_direct_predictions_full.json`

### GPT-5.2 未微调推理
- 指标文件：`output/llm_two_stage_subset_full_06b/infer/gpt52_external_metrics.json`
- 预测文件：`output/llm_two_stage_subset_full_06b/infer/gpt52_external_predictions_scored.csv`

## 当前结果汇总

| 模型 | 样本数 | Precision(reject) | Recall(default) | Reject Rate | Approval Bad Rate | Lift |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.2 未微调 | 1422 | 0.3462 | 0.8133 | 0.8228 | 0.3690 | 0.9884 |
| Qwen3-0.6B 两阶段微调 | 1422 | 0.3653 | 0.3675 | 0.3523 | 0.3420 | 1.0436 |
| Qwen3-0.6B 单阶段微调 | 1422 | 0.3587 | 0.6245 | 0.6097 | 0.3369 | 1.0249 |

说明：
- 两阶段/单阶段数据来自对应 JSON 中 `fixed_reject_rate_eval`（目标 RR=0.35）。
- GPT-5.2 数据来自 `data/loan_decisions.csv` 与 `data/shared/splits/stratified/test.csv` 按 `_shared_row_id` 对齐后的重算结果（1422 条）。
- 单阶段目前分数分布较集中，`target_reject_rate=0.35` 下实际拒绝率仍为 `0.6097`，后续若要严格对齐 RR，需改用 top-k 截断或离散分数去重策略。

## 运行注意事项
- 若 `output/.../data/*.csv` 被占用（如 Excel 打开），脚本会自动切到 `*_rerun_YYYYMMDD_HHMMSS` 目录继续执行。
- Windows 下建议用单行命令，避免反引号续行导致参数被识别成新命令。
- 脚本已统一设置 UTF-8 控制台和 Python 编码，避免 `UnicodeDecodeError`。
