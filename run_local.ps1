$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$envFile = Join-Path $PSScriptRoot ".env"

if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creating Python virtual environment..."
    & python -m venv (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Install Python 3.11 or newer and add it to PATH." }
    & $python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Run pip install -r requirements.txt again." }
}

if (-not (Test-Path -LiteralPath $envFile)) {
    $telegramToken = Read-Host "Telegram bot token from @BotFather" -AsSecureString
    $telegramPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($telegramToken)
    try {
        $plainTelegramToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($telegramPointer)
        if ($plainTelegramToken -notmatch '^\d+:[A-Za-z0-9_-]{30,}$') { throw "Invalid token format." }
        @("TELEGRAM_BOT_TOKEN=$plainTelegramToken", "WHISPER_MODEL_SIZE=small") |
            Set-Content -LiteralPath $envFile -Encoding utf8
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($telegramPointer)
        $plainTelegramToken = $null
    }
}

& $python (Join-Path $PSScriptRoot "bot.py")
exit $LASTEXITCODE
