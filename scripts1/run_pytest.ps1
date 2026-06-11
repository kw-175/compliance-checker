$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

New-Item -ItemType Directory -Force ".tmp\runtime-temp", ".tmp\pytest-runs", ".pytest-cache-local", ".uv-cache" | Out-Null
$env:TMP = (Resolve-Path ".tmp\runtime-temp").Path
$env:TEMP = (Resolve-Path ".tmp\runtime-temp").Path
$env:UV_CACHE_DIR = (Resolve-Path ".uv-cache").Path

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$runner = if (Test-Path $venvPython) { $venvPython } else { "python" }

function Test-RequiredImports {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe
    )

    & $PythonExe -c "import importlib.util, sys; mods=['fastapi','pydantic','pydantic_settings','yaml','httpx','uvicorn','pytest']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; sys.exit(0 if not missing else 1)" 2>$null
    return $LASTEXITCODE -eq 0
}

if (-not (Test-RequiredImports -PythonExe $runner) -and (Test-Path $venvPython)) {
    $sharedPathsJson = & python -c "import json, site; print(json.dumps(site.getsitepackages()))"
    if ($LASTEXITCODE -eq 0 -and $sharedPathsJson) {
        $sharedPaths = $sharedPathsJson | ConvertFrom-Json
        $env:PYTHONPATH = (($sharedPaths + @($env:PYTHONPATH)) | Where-Object { $_ }) -join ";"
    }
}

& $runner -m pytest @args
exit $LASTEXITCODE
