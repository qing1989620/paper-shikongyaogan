# Start or resume the finite supervised experiment.
param(
    [switch]$Resume,
    [int]$StopAfterSeed,
    [ValidateRange(1, 24)][int]$HealthCheckHours = 1
)

$ErrorActionPreference = 'Stop'
$researchRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$researchPython = Join-Path $researchRoot 'work/envs/research_v1/Scripts/python.exe'
$researchScript = Join-Path $researchRoot 'scripts/supervise_research_2026.py'
$researchLogDir = Join-Path $researchRoot 'work/logs/research_2026'
if (-not (Test-Path -LiteralPath $researchPython)) { throw 'Research Python environment is missing' }
$monitor = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -and $_.CommandLine.Contains($researchScript)
})
if ($monitor.Count) {
    Write-Output 'The experiment supervisor is already active.'
    return
}

$pausePath = Join-Path $researchRoot 'work/runs/research_2026/PAUSE_STATUS.json'
if (Test-Path -LiteralPath $pausePath) {
    $pause = Get-Content -LiteralPath $pausePath -Raw | ConvertFrom-Json
    $due = [DateTimeOffset]::Parse($pause.scheduled_for_cst)
    if ([DateTimeOffset]::Now -ge $due) {
        if (!$Resume) {
            Write-Output ('Research pause is in effect. Resume explicitly with -Resume. Status: ' + $pausePath)
            return
        }
        if ($pause.state -in @('scheduled','waiting_for_checkpoint','waiting_for_run_completion','pausing') -and $pause.pid) {
            $guard = Get-CimInstance Win32_Process -Filter "ProcessId=$($pause.pid)"
            if ($guard.CommandLine -and $guard.CommandLine.Contains((Join-Path $researchRoot 'scripts/pause_research_2026.py'))) {
                throw 'The pause is still saving the current epoch. Wait for PAUSE_STATUS.json to show the result.'
            }
        }
        $archived = 'PAUSE_STATUS_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.json'
        Move-Item -LiteralPath $pausePath -Destination (Join-Path (Split-Path -Parent $pausePath) $archived)
    }
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdoutPath = Join-Path $researchLogDir ('supervisor_' + $stamp + '.log')
$stderrPath = Join-Path $researchLogDir ('supervisor_' + $stamp + '.err')
$arguments = @('-B', '-u', '-X', 'utf8', ('"' + $researchScript + '"'), '--health-check-hours', [string]$HealthCheckHours)
if ($PSBoundParameters.ContainsKey('StopAfterSeed')) {
    $arguments += @('--stop-after-seed', [string]$StopAfterSeed)
}
$proc = Start-Process -FilePath $researchPython -ArgumentList $arguments -WorkingDirectory $researchRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -WindowStyle Hidden -PassThru
Write-Output ('Supervisor launched. PID: ' + $proc.Id + '    Log: ' + $stdoutPath)
