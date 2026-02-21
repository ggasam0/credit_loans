param(
  [string]$PythonExe = ".\.venv\Scripts\python.exe",
  [string]$OutputDir = "output/llm_one_stage_subset_full_06b",
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
    TrainCsv          = (Join-Path $BaseOutputDir "data/train_full.csv")
    TestCsv           = (Join-Path $BaseOutputDir "data/test_full.csv")
    AllCsv            = (Join-Path $BaseOutputDir "data/all_full.csv")
    ActionJsonl       = (Join-Path $BaseOutputDir "data/sft/credit_teacher_action_direct_full.jsonl")
    AuditCsv          = (Join-Path $BaseOutputDir "data/sft/credit_teacher_action_direct_full_audit.csv")
    ActionAdapter     = (Join-Path $BaseOutputDir "qwen3_06b_lora_action_direct")
    ActionPredCsv     = (Join-Path $BaseOutputDir "infer/stage_action_direct_predictions_full.csv")
    ActionTrainScript = (Join-Path $BaseOutputDir "lf_data/run_qwen3_06b_lora_action_direct.ps1")
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
  $ActionJsonl = $paths.ActionJsonl
  $AuditCsv = $paths.AuditCsv
  $ActionAdapter = $paths.ActionAdapter
  $ActionPredCsv = $paths.ActionPredCsv
  $ActionTrainScript = $paths.ActionTrainScript

  Invoke-Step -Name "Prepare Full Subset Data" -Script {
    & $PythonExe llm_standalone/run_llm_one_stage.py prepare-data `
      --input-csv output/bert_standalone/data/bert_subset_processed.csv `
      --output-dir $OutputDir `
      --sample-ratio 1.0 `
      --shared-split-dir data/shared/splits/stratified `
      --no-force-rebuild-shared-split `
      --test-size 0.2 `
      --val-size 0.2
  }

  Invoke-Step -Name "Generate One-Stage Teacher SFT Data" -Script {
    & $PythonExe llm_standalone/run_llm_one_stage.py generate-one-stage-sft `
      --train-csv $TrainCsv `
      --action-jsonl $ActionJsonl `
      --audit-csv $AuditCsv
  }

  Invoke-Step -Name "Write LlamaFactory Config" -Script {
    & $PythonExe llm_standalone/run_llm_one_stage.py write-one-stage-lf-config `
      --action-jsonl $ActionJsonl `
      --output-dir $OutputDir `
      --student-model-key qwen3_0.6b `
      --student-model-path $StudentModelPath `
      --num-train-epochs $ActionEpochs `
      --per-device-train-batch-size $PerDeviceTrainBatchSize `
      --gradient-accumulation-steps $GradientAccumulationSteps `
      --learning-rate $LearningRate `
      --cutoff-len $CutoffLen
  }

  if (-not (Test-Path $ActionTrainScript)) {
    throw "Missing generated train script: $ActionTrainScript"
  }

  Invoke-Step -Name "Train One-Stage Adapter (Action)" -Script {
    powershell -NoProfile -ExecutionPolicy Bypass -File $ActionTrainScript
  }

  Invoke-Step -Name "Infer One-Stage Action" -Script {
    & $PythonExe llm_standalone/run_llm_one_stage.py infer-one-stage-action `
      --test-csv $TestCsv `
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
