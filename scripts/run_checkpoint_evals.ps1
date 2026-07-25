$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = '.\.venv\Scripts\python.exe'
$evals = @(
    @{ Name = 'sft'; Checkpoint = 'checkpoints\sft'; Output = 'reports\eval_sft_test_seen.jsonl' },
    @{ Name = 'grpo_terminal'; Checkpoint = 'checkpoints\grpo_terminal'; Output = 'reports\eval_grpo_terminal_test_seen.jsonl' },
    @{ Name = 'grpo_shaped'; Checkpoint = 'checkpoints\grpo_shaped'; Output = 'reports\eval_grpo_shaped_test_seen.jsonl' }
)
foreach ($item in $evals) {
    $csv = "reports\$($item.Name)_model_metrics.csv"
    $log = "logs\$($item.Name)_model_eval.log"
    Write-Host "START $($item.Name)"
    & $py scripts\evaluate_checkpoint.py --checkpoint $item.Checkpoint --input data\processed\test_seen.jsonl --output $item.Output --csv-output $csv *>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $($item.Name) with exit code $LASTEXITCODE" }
    Write-Host "DONE $($item.Name)"
}
