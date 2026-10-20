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
    [object]$missing=[Type]::Missing
    [object]$fileNameArgument=$inputResolved
    [object]$confirm=$false
    [object]$readOnly=$true
    [object]$recent=$false
    [object]$revert=$false
    [object]$visible=$false
    [object]$saveChanges=0
    # Read-only, no recent-file entry, hidden document. No template/settings changes.
    $documents=$word.Documents
    [object[]]$openArguments=@($inputResolved,$false,$true,$false,[Type]::Missing,[Type]::Missing,$false,[Type]::Missing,[Type]::Missing,[Type]::Missing,[Type]::Missing,$false)
    $document=$documents.GetType().InvokeMember('Open',[Reflection.BindingFlags]::InvokeMethod,$null,$documents,$openArguments)
    $document.ExportAsFixedFormat($outputResolved,17,$false)
    $document.Close([ref]$saveChanges)
    [Runtime.InteropServices.Marshal]::FinalReleaseComObject($document) | Out-Null
    $document=$null
    $word.Quit([ref]$saveChanges)
    [Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
    $word=$null
    if (-not (Test-Path -LiteralPath $outputResolved)) { throw 'Word export did not produce a PDF.' }
    Get-Item -LiteralPath $outputResolved | Select-Object FullName,Length
}
finally {
    if ($null -ne $document) { $document.Close([ref]$saveChanges);[Runtime.InteropServices.Marshal]::FinalReleaseComObject($document) | Out-Null }
    if ($null -ne $word) { $word.Quit([ref]$saveChanges);[Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
