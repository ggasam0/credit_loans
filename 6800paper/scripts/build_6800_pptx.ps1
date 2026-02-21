$ErrorActionPreference = "Stop"

$DenoBin = "C:\Users\Administrator\.deno\bin"
$env:Path = "$DenoBin;$env:Path"

$SkillScript = "C:\Users\Administrator\.codex\skills\pptx-generator\scripts\generate-scratch.ts"
$Spec = "6800paper/ppt/6800_presentation_spec_v5.json"
$Out = "6800paper/6800_final_presentation_v5.pptx"

deno --version
deno run --allow-read --allow-write $SkillScript $Spec $Out -v

Write-Host "Generated: $Out"
