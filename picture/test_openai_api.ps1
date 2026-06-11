param(
    [string]$BaseUrl = "",
    [int]$WaitSeconds = 180,
    [int]$PollIntervalSeconds = 3,
    [string]$DocumentImagePath = "",
    [string]$UnsafeImagePath = ""
)

$ErrorActionPreference = "Stop"

function Get-DotEnvValue {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $parts = $trimmed -split "=", 2
        if ($parts.Count -ne 2) {
            continue
        }
        if ($parts[0].Trim() -eq $Name) {
            return $parts[1].Trim().Trim('"').Trim("'")
        }
    }
    return $null
}

function Write-Log {
    param([string]$Message)
    Write-Host "[picture-openai-test] $Message"
}

function Fail {
    param([string]$Message)
    throw "[picture-openai-test] $Message"
}

function Require-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        Fail "Missing required file: $Path"
    }
}

function New-DocumentProbeImage {
    param([string]$Path)

    Add-Type -AssemblyName System.Drawing

    $bitmap = New-Object System.Drawing.Bitmap 1200, 800
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.Clear([System.Drawing.Color]::White)

    $titleFont = New-Object System.Drawing.Font("Microsoft YaHei", 28, [System.Drawing.FontStyle]::Bold)
    $bodyFont = New-Object System.Drawing.Font("Microsoft YaHei", 24, [System.Drawing.FontStyle]::Regular)
    $blackBrush = [System.Drawing.Brushes]::Black
    $grayPen = New-Object System.Drawing.Pen([System.Drawing.Color]::LightGray, 2)

    $graphics.DrawString("Picture Compliance Test Document", $titleFont, $blackBrush, 60, 60)
    $graphics.DrawLine($grayPen, 60, 120, 1100, 120)
    $graphics.DrawString("Name: Zhang San", $bodyFont, $blackBrush, 80, 180)
    $graphics.DrawString("Phone: 13800138000", $bodyFont, $blackBrush, 80, 250)
    $graphics.DrawString("Email: student@example.com", $bodyFont, $blackBrush, 80, 320)
    $graphics.DrawString("Address: No. 1 Xueyuan Road Haidian Beijing", $bodyFont, $blackBrush, 80, 390)
    $graphics.DrawString("Note: This image is generated for picture compliance smoke testing.", $bodyFont, $blackBrush, 80, 500)

    $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)

    $graphics.Dispose()
    $grayPen.Dispose()
    $titleFont.Dispose()
    $bodyFont.Dispose()
    $bitmap.Dispose()
}

function Wait-JobTerminalStatus {
    param(
        [string]$Url,
        [int]$TimeoutSeconds,
        [int]$IntervalSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds $IntervalSeconds
        $statusPayload = Invoke-RestMethod -Method Get -Uri $Url -TimeoutSec 20
        $status = [string]$statusPayload.status
        Write-Log "Task status: $status"
        if ($status -in @("DONE", "DROPPED", "FAILED")) {
            return $statusPayload
        }
    }
    Fail "Timed out after $TimeoutSeconds seconds waiting for task completion."
}

function Submit-And-Assert {
    param(
        [string]$CaseName,
        [string]$ImagePath,
        [string]$RouteHint,
        [string]$ExpectedStatus,
        [string]$ExpectedDecision
    )

    $body = @{
        tenant_id = "picture-openai-smoke"
        source = @{
            type = "file"
            uri = [System.IO.Path]::GetFullPath($ImagePath)
            mime_type = "image/png"
        }
        profile = "default_cn_enterprise"
        options = @{
            route_hint = $RouteHint
        }
    } | ConvertTo-Json -Depth 8

    Write-Log "Submitting case '$CaseName'"
    $submit = Invoke-RestMethod -Method Post -Uri "$BaseUrl/v1/picture/jobs" -ContentType "application/json" -Body $body -TimeoutSec 30
    $jobId = [string]$submit.job_id
    if (-not $jobId) {
        Fail "Submit response did not contain job_id: $($submit | ConvertTo-Json -Depth 8)"
    }
    Write-Log "Case '$CaseName' job id: $jobId"

    $statusPayload = Wait-JobTerminalStatus -Url "$BaseUrl/v1/picture/jobs/$jobId" -TimeoutSeconds $WaitSeconds -IntervalSeconds $PollIntervalSeconds
    if ([string]$statusPayload.status -eq "FAILED") {
        $errorMessage = if ($statusPayload.error) { [string]$statusPayload.error } else { "unknown error" }
        Fail "Case '$CaseName' failed on server side: $errorMessage"
    }
    if ([string]$statusPayload.status -ne $ExpectedStatus) {
        Fail "Case '$CaseName' expected status '$ExpectedStatus' but got '$($statusPayload.status)'"
    }

    $result = Invoke-RestMethod -Method Get -Uri "$BaseUrl/v1/picture/jobs/$jobId/result" -TimeoutSec 30
    $findings = Invoke-RestMethod -Method Get -Uri "$BaseUrl/v1/picture/jobs/$jobId/findings" -TimeoutSec 30

    if ([string]$result.decision -ne $ExpectedDecision) {
        Fail "Case '$CaseName' expected decision '$ExpectedDecision' but got '$($result.decision)'"
    }

    Write-Log "Case '$CaseName' passed: status=$($statusPayload.status) decision=$($result.decision) findings=$($findings.total)"
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$EnvPath = Join-Path $ProjectRoot ".env"
$TempRoot = Join-Path $ProjectRoot "temp\picture_openai_smoke"
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

if (-not $BaseUrl) {
    $BaseUrl = if ($env:PICTURE_API_BASE_URL) {
        $env:PICTURE_API_BASE_URL
    } else {
        $value = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_API_BASE_URL"
        if ($value) {
            $value
        } else {
            $serverHost = if ($env:PICTURE_SERVER_HOST) {
                $env:PICTURE_SERVER_HOST
            } else {
                $hostFromEnv = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_SERVER_HOST"
                if ($hostFromEnv) { $hostFromEnv } else { "127.0.0.1" }
            }
            $serverPort = if ($env:PICTURE_SERVER_PORT) {
                [int]$env:PICTURE_SERVER_PORT
            } else {
                $portFromEnv = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_SERVER_PORT"
                if ($portFromEnv) { [int]$portFromEnv } else { 19012 }
            }
            "http://${serverHost}:$serverPort"
        }
    }
}
$BaseUrl = $BaseUrl.TrimEnd("/")

if (-not $DocumentImagePath) {
    $DocumentImagePath = Join-Path $TempRoot "document_probe.png"
    New-DocumentProbeImage -Path $DocumentImagePath
}
if (-not $UnsafeImagePath) {
    $UnsafeImagePath = Join-Path $ProjectRoot "picture\tests\fixtures\sample_unsafe_explicit.png"
}

Require-File -Path $DocumentImagePath
Require-File -Path $UnsafeImagePath

Write-Log "Checking service health: $BaseUrl/api/v1/health"
try {
    $health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/v1/health" -TimeoutSec 10
} catch {
    Fail "Cannot reach picture service. Start it first with picture\start_openai_api.ps1. Detail: $($_.Exception.Message)"
}
if ($health.status -ne "healthy") {
    Fail "Picture service health check returned unexpected payload: $($health | ConvertTo-Json -Depth 8)"
}

Submit-And-Assert -CaseName "document" -ImagePath $DocumentImagePath -RouteHint "document" -ExpectedStatus "DONE" -ExpectedDecision "pass_redacted"
Submit-And-Assert -CaseName "unsafe" -ImagePath $UnsafeImagePath -RouteHint "natural" -ExpectedStatus "DROPPED" -ExpectedDecision "drop"

Write-Log "All picture OpenAI smoke tests passed."
