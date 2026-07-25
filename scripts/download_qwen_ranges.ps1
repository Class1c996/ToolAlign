param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [int]$Concurrency = 8
)

$downloadRoot = Join-Path $Root '.downloads\qwen-ranges-20260720'
$modelRoot = Join-Path $Root 'models\Qwen3-1.7B'
$chunkSize = 67108864
$files = @(
    @{ Name = 'model-00001-of-00002.safetensors'; Size = 3441185608 },
    @{ Name = 'model-00002-of-00002.safetensors'; Size = 622329984 }
)
New-Item -ItemType Directory -Force -Path $downloadRoot,$modelRoot | Out-Null

foreach ($file in $files) {
    $base = "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/$($file.Name)"
    $fileDir = Join-Path $downloadRoot $file.Name
    New-Item -ItemType Directory -Force -Path $fileDir | Out-Null
    $pending = @()
    for ($start = 0; $start -lt $file.Size; $start += $chunkSize) {
        $end = [Math]::Min($file.Size - 1, $start + $chunkSize - 1)
        $index = [int]($start / $chunkSize)
        $part = Join-Path $fileDir ("part-{0:D3}.bin" -f $index)
        if (-not (Test-Path -LiteralPath $part) -or (Get-Item -LiteralPath $part).Length -ne ($end - $start + 1)) {
            $pending += [pscustomobject]@{ Start = $start; End = $end; Part = $part }
        }
    }
    for ($offset = 0; $offset -lt $pending.Count; $offset += $Concurrency) {
        $batchEnd = [Math]::Min($pending.Count - 1, $offset + $Concurrency - 1)
        $batch = @($pending[$offset..$batchEnd])
        $processes = @()
        foreach ($item in $batch) {
            $arguments = @('--ssl-no-revoke','-L','--fail','--range',"$($item.Start)-$($item.End)",'--retry','5','--retry-delay','2','--speed-limit','50000','--speed-time','30','--connect-timeout','15','--max-time','240','--silent','--show-error',$base,'-o',$item.Part)
            $processes += Start-Process -FilePath 'curl.exe' -ArgumentList $arguments -PassThru -WindowStyle Hidden
        }
        Wait-Process -Id @($processes | ForEach-Object Id)
        foreach ($process in $processes) { if ($process.ExitCode -ne 0) { throw "download failed: $($file.Name) process $($process.Id)" } }
        Write-Output ("$($file.Name): completed {0}/{1} chunks" -f [Math]::Min($offset + $Concurrency, $pending.Count), $pending.Count)
    }

    $output = Join-Path $modelRoot $file.Name
    $stream = [System.IO.File]::Open($output, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        for ($start = 0; $start -lt $file.Size; $start += $chunkSize) {
            $index = [int]($start / $chunkSize)
            $part = Join-Path $fileDir ("part-{0:D3}.bin" -f $index)
            $input = [System.IO.File]::OpenRead($part)
            try { $input.CopyTo($stream) } finally { $input.Dispose() }
        }
    } finally { $stream.Dispose() }
    if ((Get-Item -LiteralPath $output).Length -ne $file.Size) { throw "combined size mismatch: $output" }
    Write-Output ("PASS $($file.Name): $($file.Size) bytes")
}
