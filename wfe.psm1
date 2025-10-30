# PowerShell function for Workflow Engine CLI
function wfe {
    param(
        [Parameter(ValueFromRemainingArguments)]
        [string[]]$Arguments
    )
    
    $originalPath = Get-Location
    try {
        Set-Location "C:\dev\workflow_engine"
        uv run python wfe.py @Arguments
    }
    finally {
        Set-Location $originalPath
    }
}

# Export the function
Export-ModuleMember -Function wfe