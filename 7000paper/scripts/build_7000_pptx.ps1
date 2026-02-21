$ErrorActionPreference = "Stop"

$DenoBin = "C:\Users\Administrator\.deno\bin"
$env:Path = "$DenoBin;$env:Path"

$SkillScript = "C:\Users\Administrator\.codex\skills\pptx-generator\scripts\generate-scratch.ts"
$Spec = "7000paper/ppt/7000_presentation_spec_v4.json"
$Out = "7000paper/7000_presentation_v4.pptx"

deno --version
deno run --allow-read --allow-write $SkillScript $Spec $Out -v

Write-Host "Generated: $Out"
