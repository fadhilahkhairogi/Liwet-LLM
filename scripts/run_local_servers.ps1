<#
.SYNOPSIS
    Jalankan liwet-space-cpp secara lokal (in-process GGUF atau lewat llama-server).

.DESCRIPTION
    - Backend `llamacpp` (default): UI memuat GGUF langsung via llama-cpp-python.
    - Backend `server`: skrip menyalakan llama-server.exe lebih dulu, menunggu
      /health, lalu menjalankan UI dengan LIWET_SERVER_URL.
    - Backend `torch`: jalur PyTorch asli (butuh bobot safetensors).

    Tidak mengubah UI dan tidak mengunggah apa pun.

.EXAMPLE
    .\scripts\run_local_servers.ps1
    .\scripts\run_local_servers.ps1 -Backend server -Quant Q8_0 -Gpu
#>
[CmdletBinding()]
param(
    [ValidateSet("llamacpp", "server", "torch")]
    [string]$Backend = "llamacpp",

    [ValidateSet("BF16", "F16", "Q8_0")]
    [string]$Quant = "Q8_0",

    [string]$GgufDir = "E:\liwet_gguf_build\gguf",
    [string]$Model = "Liwet-400M-A188M-id-chat",
    [string]$LlamaServer = "E:\liwet_gguf_build\llama.cpp\build\bin\llama-server.exe",
    [int]$Port = 8080,
    [switch]$Gpu,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

$env:LIWET_GGUF_DIR = $GgufDir
$env:LIWET_GGUF_QUANT = $Quant
$env:LIWET_BACKEND = $Backend
if (-not $Gpu) { $env:LIWET_GGUF_NGL = "0" }
else { $env:LIWET_GGUF_NGL = "-1" }

$serverProc = $null
try {
    if ($Backend -eq "server") {
        $modelPath = Join-Path $GgufDir "$Model-$Quant.gguf"
        if (-not (Test-Path $modelPath)) {
            throw "GGUF tidak ditemukan: $modelPath"
        }
        if (-not (Test-Path $LlamaServer)) {
            throw "llama-server.exe tidak ditemukan: $LlamaServer"
        }

        $ngl = if ($Gpu) { "99" } else { "0" }
        Write-Host "Menyalakan llama-server di port $Port (ngl=$ngl)..."
        $serverProc = Start-Process -FilePath $LlamaServer -PassThru -NoNewWindow -ArgumentList @(
            "-m", $modelPath,
            "--port", "$Port",
            "-c", "2048",
            "-b", "512",
            "-ub", "512",
            "-ngl", $ngl,
            "--no-webui"
        )

        $url = "http://127.0.0.1:$Port"
        $ready = $false
        for ($i = 0; $i -lt 120; $i++) {
            Start-Sleep -Milliseconds 500
            try {
                $resp = Invoke-WebRequest -Uri "$url/health" -UseBasicParsing -TimeoutSec 2
                if ($resp.StatusCode -eq 200) { $ready = $true; break }
            } catch { }
            if ($serverProc.HasExited) {
                throw "llama-server berhenti dengan kode $($serverProc.ExitCode)."
            }
        }
        if (-not $ready) { throw "llama-server tidak siap setelah 60 detik." }
        $env:LIWET_SERVER_URL = $url
        Write-Host "llama-server siap: $url"
    }

    Write-Host "Menjalankan UI (backend=$Backend, quant=$Quant)..."
    Push-Location (Join-Path $repoRoot "liwet-space-cpp")
    try {
        & $Python app.py
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($serverProc -and -not $serverProc.HasExited) {
        Write-Host "Menghentikan llama-server..."
        Stop-Process -Id $serverProc.Id -Force
    }
}
