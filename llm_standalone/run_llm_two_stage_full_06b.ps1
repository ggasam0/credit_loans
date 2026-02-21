param(
  [string]$PythonExe = ".\.venv\Scripts\python.exe",
  [string]$OutputDir = "output/llm_two_stage_subset_full_06b",
  [double]$ReasonEpochs = 2.0,
  [double]$ActionEpochs = 4.0,
  [int]$PerDeviceTrainBatchSize = 1,
  [int]$GradientAccumulationSteps = 8,
  [double]$LearningRate = 1e-4,
  [int]$CutoffLen = 2048,
  [string]$DeviceMap = "auto",
  [int]$InferBatchSize = 1,
  [double]$TargetRejectRate = 0.35
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = $Utf8NoBom
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Invoke-Step {
  param(
    [string]$Name,
    [scriptblock]$Script
  )
  Write-Host ""
  Write-Host "========== $Name ==========" -ForegroundColor Cyan
  & $Script
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Name (exit code: $LASTEXITCODE)"
  }
}

function Test-FileLocked {
  param(
    [string]$Path
  )
  if (-not (Test-Path $Path)) {
    return $false
  }
  try {
    $stream = [System.IO.File]::Open($Path, "Open", "ReadWrite", "None")
    $stream.Close()
    return $false
  }
  catch {
    return $true
  }
}

function New-OutputPaths {
  param(
    [string]$BaseOutputDir
  )
  return @{
    TrainCsv         = (Join-Path $BaseOutputDir "data/train_full.csv")
    TestCsv          = (Join-Path $BaseOutputDir "data/test_full.csv")
    AllCsv           = (Join-Path $BaseOutputDir "data/all_full.csv")
    ReasonJsonl      = (Join-Path $BaseOutputDir "data/sft/credit_teacher_reason_full.jsonl")
    ActionJsonl      = (Join-Path $BaseOutputDir "data/sft/credit_teacher_action_full.jsonl")
    AuditCsv         = (Join-Path $BaseOutputDir "data/sft/credit_teacher_two_stage_full_audit.csv")
    ReasonAdapter    = (Join-Path $BaseOutputDir "qwen3_06b_lora_reason")
    ActionAdapter    = (Join-Path $BaseOutputDir "qwen3_06b_lora_action")
    ReasonPredCsv    = (Join-Path $BaseOutputDir "infer/stage1_reason_predictions_full.csv")
    ActionPredCsv    = (Join-Path $BaseOutputDir "infer/stage2_action_predictions_full.csv")
    ReasonTrainScript= (Join-Path $BaseOutputDir "lf_data/run_qwen3_06b_lora_reason.ps1")
    ActionTrainScript= (Join-Path $BaseOutputDir "lf_data/run_qwen3_06b_lora_action.ps1")
  }
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $ProjectRoot

try {
  if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
  }

  if (-not (Test-Path "./models/Qwen3-0.6B")) {
    throw "Student model not found: ./models/Qwen3-0.6B"
  }
  $StudentModelPath = (Resolve-Path "./models/Qwen3-0.6B").Path

  $paths = New-OutputPaths -BaseOutputDir $OutputDir
  $lockedFiles = @()
  foreach ($candidate in @($paths.TrainCsv, $paths.TestCsv, $paths.AllCsv)) {
    if (Test-FileLocked -Path $candidate) {
      $lockedFiles += $candidate
    }
  }
  if ($lockedFiles.Count -gt 0) {
    $suffix = (Get-Date -Format "yyyyMMdd_HHmmss")
    $newOutputDir = "${OutputDir}_rerun_${suffix}"
    Write-Host "Detected locked output files:" -ForegroundColor Yellow
    $lockedFiles | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    Write-Host "Switching OutputDir to: $newOutputDir" -ForegroundColor Yellow
    $OutputDir = $newOutputDir
    $paths = New-OutputPaths -BaseOutputDir $OutputDir
  }

  $TrainCsv = $paths.TrainCsv
  $TestCsv = $paths.TestCsv
  $ReasonJsonl = $paths.ReasonJsonl
  $ActionJsonl = $paths.ActionJsonl
  $AuditCsv = $paths.AuditCsv
  $ReasonAdapter = $paths.ReasonAdapter
  $ActionAdapter = $paths.ActionAdapter
  $ReasonPredCsv = $paths.ReasonPredCsv
  $ActionPredCsv = $paths.ActionPredCsv
  $ReasonTrainScript = $paths.ReasonTrainScript
  $ActionTrainScript = $paths.ActionTrainScript

  Invoke-Step -Name "Prepare Full Subset Data" -Script {
    & $PythonExe llm_standalone/run_llm_two_stage.py prepare-data `
      --input-csv output/bert_standalone/data/bert_subset_processed.csv `
      --output-dir $OutputDir `
      --sample-ratio 1.0 `
      --shared-split-dir data/shared/splits/stratified `
      --no-force-rebuild-shared-split `
      --test-size 0.2 `
      --val-size 0.2
  }

  Invoke-Step -Name "Generate Two-Stage Teacher SFT Data" -Script {
    & $PythonExe llm_standalone/run_llm_two_stage.py generate-two-stage-sft `
      --train-csv $TrainCsv `
      --reason-jsonl $ReasonJsonl `
      --action-jsonl $ActionJsonl `
      --audit-csv $AuditCsv
  }

  Invoke-Step -Name "Write LlamaFactory Configs" -Script {
    & $PythonExe llm_standalone/run_llm_two_stage.py write-two-stage-lf-config `
      --reason-jsonl $ReasonJsonl `
      --action-jsonl $ActionJsonl `
      --output-dir $OutputDir `
      --student-model-key qwen3_0.6b `
      --student-model-path $StudentModelPath `
      --reason-num-train-epochs $ReasonEpochs `
      --action-num-train-epochs $ActionEpochs `
      --per-device-train-batch-size $PerDeviceTrainBatchSize `
      --gradient-accumulation-steps $GradientAccumulationSteps `
      --learning-rate $LearningRate `
      --cutoff-len $CutoffLen
  }

  if (-not (Test-Path $ReasonTrainScript)) {
    throw "Missing generated train script: $ReasonTrainScript"
  }
  if (-not (Test-Path $ActionTrainScript)) {
    throw "Missing generated train script: $ActionTrainScript"
  }

  Invoke-Step -Name "Train Stage-1 Adapter (Reason)" -Script {
    powershell -NoProfile -ExecutionPolicy Bypass -File $ReasonTrainScript
  }

  Invoke-Step -Name "Train Stage-2 Adapter (Action)" -Script {
    powershell -NoProfile -ExecutionPolicy Bypass -File $ActionTrainScript
  }

  Invoke-Step -Name "Infer Stage-1 Reason" -Script {
    & $PythonExe llm_standalone/run_llm_two_stage.py infer-reason `
      --test-csv $TestCsv `
      --output-csv $ReasonPredCsv `
      --model-key qwen3_0.6b `
      --model-path $StudentModelPath `
      --adapter-path $ReasonAdapter `
      --device-map $DeviceMap `
      --batch-size $InferBatchSize `
      --max-new-tokens 192 `
      --temperature 0.0
  }

  Invoke-Step -Name "Infer Stage-2 Action" -Script {
    & $PythonExe llm_standalone/run_llm_two_stage.py infer-action `
      --reason-pred-csv $ReasonPredCsv `
      --output-csv $ActionPredCsv `
      --model-key qwen3_0.6b `
      --model-path $StudentModelPath `
      --adapter-path $ActionAdapter `
      --device-map $DeviceMap `
      --batch-size $InferBatchSize `
      --max-new-tokens 16 `
      --temperature 0.0 `
      --target-reject-rate $TargetRejectRate
  }

  Write-Host ""
  Write-Host "All done." -ForegroundColor Green
  Write-Host "Final prediction file: $ActionPredCsv"
  Write-Host "Final metrics file: $($ActionPredCsv).json"
}
finally {
  Pop-Location
}
