param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [int]$Concurrency = 6
)

$url = 'https://download-r2.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp312-cp312-win_amd64.whl'
$total = 3273024349
$chunkSize = 67108864
$downloadRoot = Join-Path $Root '.downloads\torch-ranges-20260719'
$output = Join-Path $downloadRoot 'torch-2.7.1+cu128-cp312-cp312-win_amd64-ranged.whl'
New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null

$pending = @()
for ($start = 0; $start -lt $total; $start += $chunkSize) {
    $end = [Math]::Min($total - 1, $start + $chunkSize - 1)
    $index = [int]($start / $chunkSize)
    $part = Join-Path $downloadRoot ("part-{0:D2}.bin" -f $index)
    $expected = $end - $start + 1
    if (-not (Test-Path -LiteralPath $part) -or (Get-Item -LiteralPath $part).Length -ne $expected) {
        $pending += [pscustomobject]@{ Start = $start; End = $end; Part = $part }
    }
}
for ($offset = 0; $offset -lt $pending.Count; $offset += $Concurrency) {
    $batch = @($pending[$offset..([Math]::Min($pending.Count - 1, $offset + $Concurrency - 1))])
    $processes = @()
    foreach ($item in $batch) {
        Write-Output ("Starting bytes {0}-{1}" -f $item.Start, $item.End)
        $arguments = @('--ssl-no-revoke','-L','--fail','--range',"$($item.Start)-$($item.End)",'--retry','5','--retry-delay','2','--speed-limit','100000','--speed-time','30','--connect-timeout','15','--max-time','300','--silent','--show-error',$url,'-o',$item.Part)
        $processes += Start-Process -FilePath 'curl.exe' -ArgumentList $arguments -PassThru -WindowStyle Hidden
    }
    Wait-Process -Id @($processes | ForEach-Object Id)
    foreach ($process in $processes) { if ($process.ExitCode -ne 0) { throw "range process failed: $($process.Id)" } }
}

$parts = @()
for ($start = 0; $start -lt $total; $start += $chunkSize) {
    $end = [Math]::Min($total - 1, $start + $chunkSize - 1)
    $index = [int]($start / $chunkSize)
    $part = Join-Path $downloadRoot ("part-{0:D2}.bin" -f $index)
    if (-not (Test-Path -LiteralPath $part) -or (Get-Item -LiteralPath $part).Length -ne ($end - $start + 1)) { throw "range size mismatch: $part" }
    $parts += $part
}

$outputStream = [System.IO.File]::Open($output, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
try { foreach ($part in $parts) { $inputStream = [System.IO.File]::OpenRead($part); try { $inputStream.CopyTo($outputStream) } finally { $inputStream.Dispose() } } } finally { $outputStream.Dispose() }
$actual = (Get-Item -LiteralPath $output).Length
if ($actual -ne $total) { throw "combined wheel size mismatch: $actual vs $total" }
Write-Output ("PASS {0} bytes: {1}" -f $actual, $output)
