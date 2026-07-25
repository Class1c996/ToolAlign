param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [int]$StartPart = 36,
    [int]$Concurrency = 8
)

$url = 'https://download-r2.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp312-cp312-win_amd64.whl'
$total = 3273024349
$partSize = 67108864
$subSize = 16777216
$downloadRoot = Join-Path $Root '.downloads\torch-ranges-20260719'
New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null

$pending = @()
for ($partIndex = $StartPart; $partIndex * $partSize -lt $total; $partIndex++) {
    $partStart = $partIndex * $partSize
    $partEnd = [Math]::Min($total - 1, $partStart + $partSize - 1)
    $subIndex = 0
    for ($start = $partStart; $start -le $partEnd; $start += $subSize) {
        $end = [Math]::Min($partEnd, $start + $subSize - 1)
        $sub = Join-Path $downloadRoot ("tail-{0:D2}-{1:D2}.bin" -f $partIndex, $subIndex)
        if (-not (Test-Path -LiteralPath $sub) -or (Get-Item -LiteralPath $sub).Length -ne ($end - $start + 1)) {
            $pending += [pscustomobject]@{ Start = $start; End = $end; Part = $sub }
        }
        $subIndex++
    }
}

for ($offset = 0; $offset -lt $pending.Count; $offset += $Concurrency) {
    $batchEnd = [Math]::Min($pending.Count - 1, $offset + $Concurrency - 1)
    $batch = @($pending[$offset..$batchEnd])
    $processes = @()
    foreach ($item in $batch) {
        $arguments = @('--ssl-no-revoke','-L','--fail','--range',"$($item.Start)-$($item.End)",'--retry','5','--retry-delay','2','--speed-limit','50000','--speed-time','30','--connect-timeout','15','--max-time','180','--silent','--show-error',$url,'-o',$item.Part)
        $processes += Start-Process -FilePath 'curl.exe' -ArgumentList $arguments -PassThru -WindowStyle Hidden
    }
    Wait-Process -Id @($processes | ForEach-Object Id)
    foreach ($process in $processes) { if ($process.ExitCode -ne 0) { throw "tail range process failed: $($process.Id)" } }
}

for ($partIndex = $StartPart; $partIndex * $partSize -lt $total; $partIndex++) {
    $partStart = $partIndex * $partSize
    $partEnd = [Math]::Min($total - 1, $partStart + $partSize - 1)
    $part = Join-Path $downloadRoot ("part-{0:D2}.bin" -f $partIndex)
    $outputStream = [System.IO.File]::Open($part, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try {
        $subIndex = 0
        for ($start = $partStart; $start -le $partEnd; $start += $subSize) {
            $sub = Join-Path $downloadRoot ("tail-{0:D2}-{1:D2}.bin" -f $partIndex, $subIndex)
            $expected = [Math]::Min($partEnd, $start + $subSize - 1) - $start + 1
            if ((Get-Item -LiteralPath $sub).Length -ne $expected) { throw "tail size mismatch: $sub" }
            $inputStream = [System.IO.File]::OpenRead($sub)
            try { $inputStream.CopyTo($outputStream) } finally { $inputStream.Dispose() }
            $subIndex++
        }
    } finally { $outputStream.Dispose() }
    if ((Get-Item -LiteralPath $part).Length -ne ($partEnd - $partStart + 1)) { throw "part size mismatch: $part" }
}
Write-Output 'PASS tail parts'
