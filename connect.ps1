param(
    [Parameter(Mandatory=$false)]
    [string]$Hostname
)

if (-not $Hostname) {
    $Hostname = Read-Host "Paste the Cloudflare Hostname from Colab (e.g., example.trycloudflare.com)"
}

$Hostname = $Hostname.Trim()
if ($Hostname -match "([a-zA-Z0-9\-]+\.trycloudflare\.com)") {
    $Hostname = $matches[1]
}

$sshConfigFile = "$env:USERPROFILE\.ssh\config"
$configEntry = @"

Host colab
    HostName $Hostname
    User root
    Port 22
    ProxyCommand cloudflared access ssh --hostname %h
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
"@

# Update ~/.ssh/config cleanly
if (Test-Path $sshConfigFile) {
    $current = Get-Content $sshConfigFile -Raw
    if ($current -match "Host colab\s") {
        # Replace existing entry
        $updated = [System.Text.RegularExpressions.Regex]::Replace($current, "(?ms)Host colab\s.*?(?=(\r?\nHost |\Z))", $configEntry.Trim())
        Set-Content -Path $sshConfigFile -Value $updated
    } else {
        Add-Content -Path $sshConfigFile -Value $configEntry
    }
} else {
    Set-Content -Path $sshConfigFile -Value $configEntry
}

Write-Host "Configured ~/.ssh/config for Host 'colab' -> $Hostname" -ForegroundColor Green
Write-Host "Connecting to Google Colab... (Default password: alpha_sovereign_2026)" -ForegroundColor Cyan
ssh colab
