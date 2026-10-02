<#
.SYNOPSIS
  Starts the full FootyVision stack: Postgres, LM Studio (chat + embedding models),
  the FastAPI backend, and the Next.js dashboard. Then opens the dashboard in a browser.

.USAGE
  From the project root:  .\scripts\start.ps1
  (If script execution is blocked, run once: Set-ExecutionPolicy -Scope Process Bypass)
#>

# The script lives in scripts/, so the project root is one level up.
$root = Split-Path $PSScriptRoot -Parent
$lms = "$env:USERPROFILE\.lmstudio\bin\lms.exe"
$chatModel = "google/gemma-4-e4b"
$embedModel = "text-embedding-nomic-embed-text-v1.5"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

# --- 1. Postgres ---------------------------------------------------------
Step "Postgres (Docker)"
$dbStatus = docker ps --filter name=footyvision-db --format "{{.Status}}" 2>$null
if ($dbStatus -match "Up") {
    Ok "already running ($dbStatus)"
} else {
    docker compose -f "$root\docker-compose.yml" up -d db *>$null
    Write-Host "    waiting for healthy..." -NoNewline
    $h = $null
    for ($i = 0; $i -lt 40; $i++) {
        $h = docker inspect --format '{{.State.Health.Status}}' footyvision-db 2>$null
        if ($h -eq "healthy") { Write-Host ""; Ok "healthy"; break }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 3
    }
    if ($h -ne "healthy") { Warn "Postgres did not become healthy in time - check Docker Desktop / free RAM." }
}

# --- 2. LM Studio (chat + embedding models) -------------------------------
Step "LM Studio (local LLM)"
if (-not (Test-Path $lms)) {
    Warn "lms CLI not found at $lms - skipping. Reports / NL search / assistant need it running manually."
} else {
    $psOutput = & $lms ps 2>$null
    $serverDown = ($psOutput -join "") -match "server is not running|No models are currently"

    if ($serverDown -and (($psOutput -join "") -match "not running")) {
        & $lms server start 2>$null *>$null
        Ok "server started on port 1234"
    } else {
        Ok "server already running on port 1234"
    }

    if (($psOutput -join "`n") -notmatch [regex]::Escape($chatModel)) {
        Write-Host "    loading $chatModel (this can take ~20-60s)..." -NoNewline
        & $lms load $chatModel -y --context-length 4096 2>$null *>$null
        Write-Host " done"
    } else { Ok "$chatModel already loaded" }

    if (($psOutput -join "`n") -notmatch [regex]::Escape($embedModel)) {
        Write-Host "    loading $embedModel ..." -NoNewline
        & $lms load $embedModel -y 2>$null *>$null
        Write-Host " done"
    } else { Ok "$embedModel already loaded" }
}

# --- 3. API (FastAPI / uvicorn) ------------------------------------------
Step "API (FastAPI)"
$apiUp = $false
try { Invoke-RestMethod http://127.0.0.1:8000/health -TimeoutSec 2 -ErrorAction Stop | Out-Null; $apiUp = $true } catch {}
if ($apiUp) {
    Ok "already running on :8000"
} else {
    $venvUvicorn = Join-Path $root ".venv\Scripts\uvicorn.exe"
    if (-not (Test-Path $venvUvicorn)) {
        Warn "venv not found at $venvUvicorn - create it first (see README): python -m venv .venv; pip install -e `".[dev]`""
    } else {
        Start-Process -FilePath $venvUvicorn `
            -ArgumentList "footyvision.api.main:app", "--host", "127.0.0.1", "--port", "8000" `
            -WorkingDirectory $root `
            -WindowStyle Hidden
        Write-Host "    waiting for API..." -NoNewline
        $up = $false
        for ($i = 0; $i -lt 30; $i++) {
            try { Invoke-RestMethod http://127.0.0.1:8000/health -TimeoutSec 2 -ErrorAction Stop | Out-Null; $up = $true; break }
            catch { Write-Host "." -NoNewline; Start-Sleep -Seconds 2 }
        }
        Write-Host ""
        if ($up) { Ok "up on :8000" } else { Warn "API did not respond in time - check .venv and Postgres." }
    }
}

# --- 4. Frontend (Next.js dashboard) -------------------------------------
Step "Frontend (Next.js)"
$webUp = $false
try { Invoke-WebRequest http://127.0.0.1:3000 -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop | Out-Null; $webUp = $true } catch {}
if ($webUp) {
    Ok "already running on :3000"
} else {
    $webDir = Join-Path $root "frontend\web"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "npm run dev" `
        -WorkingDirectory $webDir -WindowStyle Hidden
    Write-Host "    waiting for dashboard..." -NoNewline
    $up = $false
    for ($i = 0; $i -lt 30; $i++) {
        try { Invoke-WebRequest http://127.0.0.1:3000 -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop | Out-Null; $up = $true; break }
        catch { Write-Host "." -NoNewline; Start-Sleep -Seconds 2 }
    }
    Write-Host ""
    if ($up) { Ok "up on :3000" } else { Warn "Dashboard did not respond in time - check `frontend\web\node_modules` (npm install)." }
}

# --- Done ------------------------------------------------------------------
Step "All set"
Ok "Dashboard:  http://localhost:3000"
Ok "API docs:   http://localhost:8000/docs"
Start-Process "http://localhost:3000"
