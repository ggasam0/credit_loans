# Run this manually. Do not run while other heavy training is in progress.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
$LlamaFactoryDir = Join-Path $ProjectRoot "LlamaFactory"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = $Utf8NoBom
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$YamlPath = Join-Path $ScriptDir "qwen3_06b_lora_reason.yaml"
Set-Location $LlamaFactoryDir
if (Test-Path $PythonExe) {
  & $PythonExe -X utf8 -m llamafactory.cli train $YamlPath
} else {
  python -X utf8 -m llamafactory.cli train $YamlPath
}
