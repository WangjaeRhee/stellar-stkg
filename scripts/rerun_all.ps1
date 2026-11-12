# Clean re-run pipeline (run from the project root).
# Every step is resumable; re-running a step skips ids that are already done.
param(
    [string]$Py = "python",
    [string]$Model = "qwen2.5:7b",
    [switch]$SkipOpenAI
)
$ErrorActionPreference = "Stop"
$slug = ($Model -replace '[^A-Za-z0-9.\-]+', '-')
$R = "results\rerun"

Write-Host "== 0. leakage check"
& $Py scripts\check_leakage.py --json "$R\leakage_report.json"
if ($LASTEXITCODE -ne 0) { throw "leakage check failed" }

Write-Host "== 1. extraction (450)"
& $Py evaluation\run_extract.py --model $Model --no-show

Write-Host "== 2. judge + summary"
& $Py evaluation\run_judge.py --extract "$R\extract_$slug.jsonl"
& $Py evaluation\summarize.py --judge "$R\judge_$slug.jsonl=$Model" --title "Full 450 ($Model)"

Write-Host "== 3. sample90"
& $Py evaluation\make_sample90.py

Write-Host "== 4. component analysis"
& $Py evaluation\run_ablation.py --model $Model
$cfgs = @("direct=Direct", "atomic_direct=+Atomic", "multistage_nocot=+Multi-stage", "multistage_cot=+CoT", "full=Full")
$judgeArgs = @()
foreach ($c in $cfgs) {
    $name, $label = $c.Split("=")
    & $Py evaluation\run_judge.py --extract "$R\extract_ablation_${name}_$slug.jsonl"
    $judgeArgs += "--judge"; $judgeArgs += "$R\judge_ablation_${name}_$slug.jsonl=$label"
}
& $Py evaluation\summarize.py @judgeArgs --out "$R\summary_ablation_$slug" --title "Component analysis (sample90)"

Write-Host "== 5. model comparison"
& $Py evaluation\run_extract.py --model $Model --ids "$R\sample90_ids.json" --tag sample90 --no-show
& $Py evaluation\run_judge.py --extract "$R\extract_${slug}_sample90.jsonl"
$cmpArgs = @("--judge", "$R\judge_${slug}_sample90.jsonl=Qwen2.5-7B")
if (-not $SkipOpenAI -and $env:OPENAI_API_KEY) {
    foreach ($m in @("gpt-4o-mini", "gpt-4o")) {
        & $Py evaluation\run_extract.py --provider openai --model $m --ids "$R\sample90_ids.json" --tag sample90 --no-show
        & $Py evaluation\run_judge.py --extract "$R\extract_${m}_sample90.jsonl"
        $cmpArgs += "--judge"; $cmpArgs += "$R\judge_${m}_sample90.jsonl=$m"
    }
} else {
    Write-Host "OPENAI_API_KEY not set (or -SkipOpenAI): model comparison limited to $Model"
}
& $Py evaluation\summarize.py @cmpArgs --out "$R\summary_model_compare" --title "Model comparison (sample90, judge=Qwen)"

Write-Host "== 6. human check sheet"
& $Py evaluation\make_human_check.py --extract "$R\extract_$slug.jsonl"

Write-Host "== 7. entity2 rule impact"
& $Py evaluation\corrections_report.py --extract "$R\extract_$slug.jsonl"
& $Py evaluation\run_judge.py --extract "$R\extract_$slug.jsonl" --no-corrections
& $Py evaluation\summarize.py --judge "$R\judge_$slug.jsonl=rules ON" --judge "$R\judge_${slug}_norules.jsonl=rules OFF" --out "$R\summary_rules_onoff_$slug" --title "ENTITY2_CORRECTIONS on/off (450)"

Write-Host "done"
