# LLM 实验说明（两阶段 + 单阶段）

## 统一数据与切分
- 输入数据：`output/bert_standalone/data/bert_subset_processed.csv`（含 `bert_input`）。
- 切分方式：复用共享分层切分 `data/shared/splits/stratified`。
- 不使用时间切分。
- 默认全量：`sample_ratio=1.0`。

## 分支 A：两阶段（Input -> Reason -> Action）

### 核心脚本
- 主流程：`llm_standalone/run_llm_two_stage.py`
- 一键执行：`llm_standalone/run_llm_two_stage_full_06b.ps1`

### 微调数据
- 阶段1数据（Input -> Reason）：`output/llm_two_stage_subset_full_06b/data/sft/credit_teacher_reason_full.jsonl`
- 阶段2数据（Input + Reason -> Action）：`output/llm_two_stage_subset_full_06b/data/sft/credit_teacher_action_full.jsonl`
- 审计文件：`output/llm_two_stage_subset_full_06b/data/sft/credit_teacher_two_stage_full_audit.csv`

### 一键运行
```powershell
powershell -ExecutionPolicy Bypass -File llm_standalone\run_llm_two_stage_full_06b.ps1
```

### 当前默认参数
- 学生模型：`./models/Qwen3-0.6B`
- 阶段1 epoch：`2`
- 阶段2 epoch：`4`
- 推理目标拒绝率：`0.35`

### 可选参数示例
```powershell
powershell -ExecutionPolicy Bypass -File llm_standalone\run_llm_two_stage_full_06b.ps1 `
  -ReasonEpochs 2 `
  -ActionEpochs 4 `
  -InferBatchSize 1 `
  -TargetRejectRate 0.35
```

### 输出目录
- `output/llm_two_stage_subset_full_06b`

关键文件：
- `output/llm_two_stage_subset_full_06b/subset_report.json`
- `output/llm_two_stage_subset_full_06b/infer/stage2_action_predictions_full.csv`
- `output/llm_two_stage_subset_full_06b/infer/stage2_action_predictions_full.json`

## 分支 B：单阶段（Input -> Action）

### 核心脚本
- 主流程：`llm_standalone/run_llm_one_stage.py`
- 一键执行：`llm_standalone/run_llm_one_stage_full_06b.ps1`

### 微调数据
- 单阶段动作数据（Input -> Action）：`output/llm_one_stage_subset_full_06b/data/sft/credit_teacher_action_direct_full.jsonl`
- 审计文件：`output/llm_one_stage_subset_full_06b/data/sft/credit_teacher_action_direct_full_audit.csv`

### 一键运行
```powershell
powershell -ExecutionPolicy Bypass -File llm_standalone\run_llm_one_stage_full_06b.ps1
```

### 当前默认参数
- 学生模型：`./models/Qwen3-0.6B`
- 单阶段 epoch：`4`
- 推理目标拒绝率：`0.35`

### 可选参数示例
```powershell
powershell -ExecutionPolicy Bypass -File llm_standalone\run_llm_one_stage_full_06b.ps1 `
  -ActionEpochs 4 `
  -InferBatchSize 1 `
  -TargetRejectRate 0.35
```

### 输出目录
- `output/llm_one_stage_subset_full_06b`

关键文件：
- `output/llm_one_stage_subset_full_06b/subset_report.json`
- `output/llm_one_stage_subset_full_06b/infer/stage_action_direct_predictions_full.csv`
- `output/llm_one_stage_subset_full_06b/infer/stage_action_direct_predictions_full.json`

## 运行注意事项
- 若 `output/.../data/*.csv` 被占用（如 Excel 打开），一键脚本会自动切到 `*_rerun_YYYYMMDD_HHMMSS` 目录继续执行。
- 脚本已统一设置 UTF-8 控制台和 Python 编码，避免 Windows 下 `UnicodeDecodeError`。
