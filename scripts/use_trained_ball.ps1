# Run this after training finishes. Copies best.pt to backend and shows the env var to set.
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$best = Join-Path $repoRoot "runs\ball\train\weights\best.pt"
$dest = Join-Path $repoRoot "backend\ball_best.pt"

if (-not (Test-Path $best)) {
    Write-Host "Trained weights not found at: $best"
    Write-Host "Wait for training to finish (script: evaluate_ball_model.py --train-epochs 50), then run this again."
    exit 1
}

Copy-Item -Path $best -Destination $dest -Force
Write-Host "Copied best.pt to backend\ball_best.pt"
Write-Host ""
Write-Host "Set this before starting the backend (PowerShell):"
Write-Host ('$env:BALL_DETECT_MODEL = "' + (Resolve-Path $dest).Path + '"') -ForegroundColor Green
