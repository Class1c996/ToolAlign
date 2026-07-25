$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = '.\.venv\Scripts\python.exe'
$evals = @(
    @{ Name = 'sft_chat'; Checkpoint = 'checkpoints\sft_chat'; Output = 'reports\eval_sft_chat_test_seen.jsonl' },
    @{ Name = 'grpo_terminal_chat'; Checkpoint = 'checkpoints\grpo_terminal_chat'; Output = 'reports\eval_grpo_terminal_chat_test_seen.jsonl' },
    @{ Name = 'grpo_shaped_chat'; Checkpoint = 'checkpoints\grpo_shaped_chat'; Output = 'reports\eval_grpo_shaped_chat_test_seen.jsonl' }
)
foreach ($item in $evals) {
    $csv = "reports\$($item.Name)_model_metrics.csv"
    $log = "logs\$($item.Name)_model_eval.log"
    Write-Host "START $($item.Name)"
    & $py scripts\evaluate_checkpoint.py --checkpoint $item.Checkpoint --input data\processed\test_seen.jsonl --output $item.Output --csv-output $csv --max-turns 3 --max-new-tokens 256 *>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $($item.Name) with exit code $LASTEXITCODE" }
    Write-Host "DONE $($item.Name)"
}
