param(
    [string]$BindHost = "",
    [int]$Port = 0,
    [switch]$Reload
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
    Write-Host "[picture-openai] $Message"
}

function Fail {
    param([string]$Message)
    throw "[picture-openai] $Message"
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$EnvPath = Join-Path $ProjectRoot ".env"

if (-not $BindHost) {
    $BindHost = if ($env:PICTURE_SERVER_HOST) {
        $env:PICTURE_SERVER_HOST
    } else {
        $hostFromEnv = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_SERVER_HOST"
        if ($hostFromEnv) { $hostFromEnv } else { "127.0.0.1" }
    }
}
if ($Port -le 0) {
    $Port = if ($env:PICTURE_SERVER_PORT) {
        [int]$env:PICTURE_SERVER_PORT
    } else {
        $portFromEnv = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_SERVER_PORT"
        if ($portFromEnv) { [int]$portFromEnv } else { 19012 }
    }
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    Fail "Python executable not found: $Python"
}

$ApiKey = if ($env:PICTURE_OPENAI_API_KEY) {
    $env:PICTURE_OPENAI_API_KEY
} elseif ($env:OPENAI_API_KEY) {
    $env:OPENAI_API_KEY
} elseif (Get-DotEnvValue -Path $EnvPath -Name "PICTURE_OPENAI_API_KEY") {
    Get-DotEnvValue -Path $EnvPath -Name "PICTURE_OPENAI_API_KEY"
} elseif (Get-DotEnvValue -Path $EnvPath -Name "OPENAI_API_KEY") {
    Get-DotEnvValue -Path $EnvPath -Name "OPENAI_API_KEY"
} else {
    ""
}
if (-not $ApiKey) {
    Fail "Missing OPENAI_API_KEY or PICTURE_OPENAI_API_KEY."
}

$env:PICTURE_OPENAI_API_KEY = $ApiKey
$env:PICTURE_OPENAI_BASE_URL = if ($env:PICTURE_OPENAI_BASE_URL) {
    $env:PICTURE_OPENAI_BASE_URL
} else {
    $value = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_OPENAI_BASE_URL"
    if ($value) { $value } else { "https://api.openai.com/v1" }
}
$env:PICTURE_OPENAI_MODEL = if ($env:PICTURE_OPENAI_MODEL) {
    $env:PICTURE_OPENAI_MODEL
} else {
    $value = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_OPENAI_MODEL"
    if ($value) { $value } else { "gpt-5.2" }
}
$env:PICTURE_OPENAI_TIMEOUT_SECONDS = if ($env:PICTURE_OPENAI_TIMEOUT_SECONDS) {
    $env:PICTURE_OPENAI_TIMEOUT_SECONDS
} else {
    $value = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_OPENAI_TIMEOUT_SECONDS"
    if ($value) { $value } else { "90" }
}
$env:PICTURE_OPENAI_IMAGE_DETAIL = if ($env:PICTURE_OPENAI_IMAGE_DETAIL) {
    $env:PICTURE_OPENAI_IMAGE_DETAIL
} else {
    $value = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_OPENAI_IMAGE_DETAIL"
    if ($value) { $value } else { "high" }
}

$env:PICTURE_OCR_PROVIDER = "openai_gpt52"
$env:PICTURE_PII_PROVIDER = "openai_gpt52"
$env:PICTURE_SAFETY_PROVIDER = "openai_gpt52"
$env:PICTURE_VISION_PROVIDER = "openai_gpt52"
$env:PICTURE_SEGMENTATION_PROVIDER = if ($env:PICTURE_SEGMENTATION_PROVIDER) {
    $env:PICTURE_SEGMENTATION_PROVIDER
} else {
    $value = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_SEGMENTATION_PROVIDER"
    if ($value) { $value } else { "mock" }
}

$env:PICTURE_STORAGE_BACKEND = if ($env:PICTURE_STORAGE_BACKEND) {
    $env:PICTURE_STORAGE_BACKEND
} else {
    $value = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_STORAGE_BACKEND"
    if ($value) { $value } else { "local" }
}
$env:PICTURE_STORAGE_BASE_PATH = if ($env:PICTURE_STORAGE_BASE_PATH) {
    $env:PICTURE_STORAGE_BASE_PATH
} else {
    $value = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_STORAGE_BASE_PATH"
    if ($value) { $value } else { Join-Path $ProjectRoot "compliance_output_picture\storage" }
}
$env:PICTURE_WORK_DIR = if ($env:PICTURE_WORK_DIR) {
    $env:PICTURE_WORK_DIR
} else {
    $value = Get-DotEnvValue -Path $EnvPath -Name "PICTURE_WORK_DIR"
    if ($value) { $value } else { Join-Path $ProjectRoot "compliance_output_picture" }
}

New-Item -ItemType Directory -Force -Path $env:PICTURE_WORK_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:PICTURE_STORAGE_BASE_PATH | Out-Null

Write-Log "Project root: $ProjectRoot"
Write-Log "Python: $Python"
Write-Log "Model: $($env:PICTURE_OPENAI_MODEL)"
Write-Log "Bind: http://${BindHost}:$Port"
Write-Log "Providers: OCR/PII/Safety/Vision = openai_gpt52"
Write-Log "Segmentation: $($env:PICTURE_SEGMENTATION_PROVIDER)"
Write-Log "Press Ctrl+C to stop."

$Args = @(
    "-m", "uvicorn",
    "picture.api.app:app",
    "--host", $BindHost,
    "--port", "$Port"
)

if ($Reload) {
    $Args += "--reload"
}

& $Python @Args
