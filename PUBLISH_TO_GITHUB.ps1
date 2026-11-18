# Publish the current directory to a NEW public repository under the connected user's account.
# Requires Git and GitHub CLI: https://cli.github.com/
param([string]$Repository = "zeenall/KAN-HMRTA-Reproducibility")
$ErrorActionPreference = "Stop"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Please install Git for Windows first." }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw "Please install GitHub CLI (gh) first." }
& gh auth status
if ($LASTEXITCODE -ne 0) { throw "Run: gh auth login" }
Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
try {
    if (Test-Path ".git") { throw "Repository already initialized; use git status/push manually to avoid overwriting existing history." }
    & git init -b main
    if ($LASTEXITCODE -ne 0) { throw "git init failed" }
    & git add .
    if ($LASTEXITCODE -ne 0) { throw "git add failed" }
    & git commit -m "Public reproducibility release: Stage A KAN, H-MRTA benchmarks and checkpoints"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed; set git user.name and user.email." }
    & gh repo create $Repository --public --source . --remote origin --push
    if ($LASTEXITCODE -ne 0) { throw "GitHub repository creation/push failed. Check whether the repo already exists and whether gh has permissions." }
    Write-Host "Published to https://github.com/$Repository"
} finally { Pop-Location }
