param([Parameter(Mandatory=$true)][string]$InputPath,[Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference='Stop'
$studyRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')+'\'
$inputResolved=[IO.Path]::GetFullPath($InputPath)
$outputResolved=[IO.Path]::GetFullPath($OutputPath)
if (-not $inputResolved.StartsWith($studyRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Input must remain inside contribution workspace.' }
if (-not $outputResolved.StartsWith($studyRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must remain inside contribution workspace.' }
$word=$null
$document=$null
try {
    $word=New-Object -ComObject Word.Application
    $word.Visible=$false
    $word.DisplayAlerts=0
    $missing=[Type]::Missing
    # Read-only, no recent-file entry, hidden document. No template/settings changes.
    $document=$word.Documents.Open($inputResolved,$false,$true,$false,$missing,$missing,$false,$missing,$missing,$missing,$missing,$false)
    $document.ExportAsFixedFormat($outputResolved,17,$false)
    $document.Close(0)
    [Runtime.InteropServices.Marshal]::FinalReleaseComObject($document) | Out-Null
    $document=$null
    $word.Quit(0)
    [Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
    $word=$null
    if (-not (Test-Path -LiteralPath $outputResolved)) { throw 'Word export did not produce a PDF.' }
    Get-Item -LiteralPath $outputResolved | Select-Object FullName,Length
}
finally {
    if ($null -ne $document) { $document.Close(0);[Runtime.InteropServices.Marshal]::FinalReleaseComObject($document) | Out-Null }
    if ($null -ne $word) { $word.Quit(0);[Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
