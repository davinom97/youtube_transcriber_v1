param(
    [Parameter(Mandatory = $true)]
    [string]$PlaylistUrl,

    [string]$OutputDir = "transcripts",
    [string]$Device = "cuda",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

& $Python -m pip install -r (Join-Path $ScriptDir "requirements-transcribe.txt")
& $Python (Join-Path $ScriptDir "transcribe_social_video.py") `
    $PlaylistUrl `
    --playlist `
    --quality high `
    --device $Device `
    --output-dir $OutputDir
