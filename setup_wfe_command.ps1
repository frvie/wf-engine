# Workflow Engine CLI Setup Script
# Run this script to make 'wfe' command available system-wide

Write-Host "Setting up Workflow Engine CLI..." -ForegroundColor Green

$workflowDir = "C:\dev\workflow_engine"
$batchFile = Join-Path $workflowDir "wfe.bat"
$psModule = Join-Path $workflowDir "wfe.psm1"

# Method 1: Add to PATH (for batch file)
Write-Host "`n1. Adding to system PATH..." -ForegroundColor Yellow
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$workflowDir*") {
    $newPath = $currentPath + ";" + $workflowDir
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    Write-Host "   ✅ Added $workflowDir to user PATH" -ForegroundColor Green
} else {
    Write-Host "   ✅ Directory already in PATH" -ForegroundColor Green
}

# Method 2: PowerShell Profile Function
Write-Host "`n2. Setting up PowerShell function..." -ForegroundColor Yellow
$profilePath = $PROFILE.CurrentUserAllHosts
$profileDir = Split-Path $profilePath -Parent

if (!(Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
}

$functionCode = @"

# Workflow Engine CLI Function
function wfe {
    param(
        [Parameter(ValueFromRemainingArguments)]
        [string[]]`$Arguments
    )
    
    `$originalPath = Get-Location
    try {
        Set-Location "$workflowDir"
        uv run python wfe.py @Arguments
    }
    finally {
        Set-Location `$originalPath
    }
}
"@

if (Test-Path $profilePath) {
    $profileContent = Get-Content $profilePath -Raw
    if ($profileContent -notlike "*function wfe*") {
        Add-Content $profilePath $functionCode
        Write-Host "   ✅ Added wfe function to PowerShell profile" -ForegroundColor Green
    } else {
        Write-Host "   ✅ wfe function already exists in profile" -ForegroundColor Green
    }
} else {
    Set-Content $profilePath $functionCode
    Write-Host "   ✅ Created PowerShell profile with wfe function" -ForegroundColor Green
}

Write-Host "`n🎉 Setup Complete!" -ForegroundColor Green
Write-Host "`nUsage options:" -ForegroundColor Cyan
Write-Host "  1. Restart your terminal and use: wfe --help" -ForegroundColor White
Write-Host "  2. Or reload your profile: . `$PROFILE" -ForegroundColor White
Write-Host "  3. Or use directly: uv run python wfe.py --help" -ForegroundColor White

Write-Host "`nExample commands:" -ForegroundColor Cyan
Write-Host "  wfe templates" -ForegroundColor White
Write-Host "  wfe devices" -ForegroundColor White
Write-Host "  wfe run workflows/granular_parallel_inference.json" -ForegroundColor White
Write-Host "  wfe create `"real-time NPU detection`"" -ForegroundColor White