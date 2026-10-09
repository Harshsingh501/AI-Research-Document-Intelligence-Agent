<#
.SYNOPSIS
    Manage the portable PostgreSQL instance used by this project (no Docker, no admin).

.DESCRIPTION
    We run PostgreSQL from a plain zip extracted into tools\pgsql. Data lives in
    tools\pgdata. This script wraps the three binaries you'd otherwise call by hand:

      initdb  - create an empty database cluster (run ONCE)
      pg_ctl  - start / stop / check the server process
      psql    - interactive SQL shell

    Equivalent to docker-compose.yml, which we'll switch to later.

.EXAMPLE
    .\tools\pg.ps1 init      # first time only: create cluster + docintel database + pgvector
    .\tools\pg.ps1 start     # start the server (listens on localhost:5432)
    .\tools\pg.ps1 status
    .\tools\pg.ps1 psql      # open a SQL shell as the docintel user
    .\tools\pg.ps1 stop
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet("init", "start", "stop", "status", "psql", "restart")]
    [string]$Command = "status"
)

$ErrorActionPreference = "Stop"

# --- Paths -------------------------------------------------------------------
$ToolsDir = $PSScriptRoot
$PgBin    = Join-Path $ToolsDir "pgsql\bin"
$PgData   = Join-Path $ToolsDir "pgdata"
$PgLog    = Join-Path $ToolsDir "pgdata\server.log"

# Must match backend\.env  (DATABASE_URL=postgresql+asyncpg://docintel:docintel@localhost:5433/docintel)
$DbUser = "docintel"
$DbPass = "docintel"
$DbName = "docintel"
$DbPort = 5433   # 5432 is taken by a pre-existing Postgres service on this machine

$env:PGPASSWORD = $DbPass   # lets psql connect without prompting

function Assert-Binaries {
    if (-not (Test-Path (Join-Path $PgBin "pg_ctl.exe"))) {
        throw "PostgreSQL binaries not found at $PgBin. Extract the EDB zip into tools\pgsql first."
    }
}

function Invoke-Init {
    Assert-Binaries
    if (Test-Path (Join-Path $PgData "PG_VERSION")) {
        Write-Host "Cluster already exists at $PgData - skipping initdb." -ForegroundColor Yellow
    } else {
        Write-Host "Creating database cluster in $PgData ..." -ForegroundColor Cyan
        # initdb wants the superuser password in a file, not on the command line
        $pwfile = Join-Path $env:TEMP "pg_pw.txt"
        Set-Content -Path $pwfile -Value $DbPass -Encoding ascii -NoNewline
        & (Join-Path $PgBin "initdb.exe") `
            --pgdata=$PgData `
            --username=$DbUser `
            --pwfile=$pwfile `
            --auth=scram-sha-256 `
            --encoding=UTF8 `
            --locale=C
        Remove-Item $pwfile -Force
        if ($LASTEXITCODE -ne 0) { throw "initdb failed" }
    }

    Invoke-Start

    Write-Host "Creating database '$DbName' and enabling pgvector ..." -ForegroundColor Cyan
    # CREATE DATABASE has no IF NOT EXISTS, so ignore an 'already exists' error
    & (Join-Path $PgBin "psql.exe") -U $DbUser -p $DbPort -d postgres -v ON_ERROR_STOP=0 `
        -c "CREATE DATABASE $DbName;" 2>$null | Out-Null
    & (Join-Path $PgBin "psql.exe") -U $DbUser -p $DbPort -d $DbName -v ON_ERROR_STOP=1 `
        -c "CREATE EXTENSION IF NOT EXISTS vector;" `
        -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
    if ($LASTEXITCODE -ne 0) { throw "Failed to enable pgvector. Is vector.dll in tools\pgsql\lib?" }

    Write-Host "`nDone. Connection string:" -ForegroundColor Green
    Write-Host "  postgresql+asyncpg://${DbUser}:${DbPass}@localhost:${DbPort}/${DbName}"
}

function Invoke-Start {
    Assert-Binaries
    & (Join-Path $PgBin "pg_ctl.exe") status -D $PgData *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PostgreSQL is already running." -ForegroundColor Yellow
        return
    }
    Write-Host "Starting PostgreSQL on port $DbPort ..." -ForegroundColor Cyan
    & (Join-Path $PgBin "pg_ctl.exe") start -D $PgData -l $PgLog -o "-p $DbPort" -w
    if ($LASTEXITCODE -ne 0) { throw "pg_ctl start failed - see $PgLog" }
}

function Invoke-Stop {
    Assert-Binaries
    Write-Host "Stopping PostgreSQL ..." -ForegroundColor Cyan
    & (Join-Path $PgBin "pg_ctl.exe") stop -D $PgData -m fast
}

function Invoke-Status {
    Assert-Binaries
    & (Join-Path $PgBin "pg_ctl.exe") status -D $PgData
}

function Invoke-Psql {
    Assert-Binaries
    & (Join-Path $PgBin "psql.exe") -U $DbUser -p $DbPort -d $DbName
}

switch ($Command) {
    "init"    { Invoke-Init }
    "start"   { Invoke-Start }
    "stop"    { Invoke-Stop }
    "restart" { Invoke-Stop; Invoke-Start }
    "status"  { Invoke-Status }
    "psql"    { Invoke-Psql }
}

