Param (
    [Parameter(Mandatory=$true)][ValidateSet("HF", "HF-Mirror", "ModelScope")][string]$Source,
    [Parameter(Mandatory=$true)][ValidateSet("v1", "v2", "v2Pro", "v2ProPlus", "v3", "v4", "v5", "all")][string]$Version,
    [Alias("PipIndexUrl")][string]$UvIndexUrl = ""
)

$global:ErrorActionPreference = 'Stop'

trap {
    Write-ErrorLog $_
}

function Write-ErrorLog {
    param (
        [System.Management.Automation.ErrorRecord]$ErrorRecord
    )

    Write-Host "`n[ERROR] Command failed:" -ForegroundColor Red
    if (-not $ErrorRecord.Exception.Message){
    } else {
        Write-Host "Message:" -ForegroundColor Red 
        $ErrorRecord.Exception.Message -split "`n" | ForEach-Object {
            Write-Host "    $_"
        }
    }

    Write-Host "Command:" -ForegroundColor Red  -NoNewline
    Write-Host " $($ErrorRecord.InvocationInfo.Line)".Replace("`r", "").Replace("`n", "")
    Write-Host "Location:" -ForegroundColor Red -NoNewline
    Write-Host " $($ErrorRecord.InvocationInfo.ScriptName):$($ErrorRecord.InvocationInfo.ScriptLineNumber)"
    Write-Host "Call Stack:" -ForegroundColor DarkRed
    $ErrorRecord.ScriptStackTrace -split "`n" | ForEach-Object {
        Write-Host "    $_" -ForegroundColor DarkRed
    }

    exit 1
}

function Write-Info($msg) {
    Write-Host "[INFO]:" -ForegroundColor Green -NoNewline
    Write-Host " $msg"
}
function Write-Success($msg) {
    Write-Host "[SUCCESS]:" -ForegroundColor Blue -NoNewline
    Write-Host " $msg"
}



function Invoke-Download {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Uri,

        [Parameter()]
        [string]$OutFile
    )

    try {
        $params = @{
            Uri = $Uri
        }

        if ($OutFile) {
            $params["OutFile"] = $OutFile
        }

        $null = Invoke-WebRequest @params -ErrorAction Stop

    } catch {
        Write-Host "Failed to download:" -ForegroundColor Red
        Write-Host "  $Uri"
        throw
    }
}

function Invoke-Unzip {
    param($ZipPath, $DestPath)
    Expand-Archive -Path $ZipPath -DestinationPath $DestPath -Force
    Remove-Item $ZipPath -Force
}

function Download-RepoFileIfMissing {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,

        [Parameter()]
        [string]$UrlPrefix = $RepoFileUrlPrefix
    )

    $localPath = Join-Path "GPT_SoVITS" $RelativePath
    $remoteUrl = "$UrlPrefix/$RelativePath"

    if (Test-Path $localPath -PathType Leaf) {
        Write-Info "File Exists: $localPath"
        return
    }

    $parent = Split-Path -Parent $localPath
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    Write-Info "Downloading $RelativePath..."
    Invoke-Download -Uri $remoteUrl -OutFile $localPath
    Write-Success "Downloaded $RelativePath"
}

function Download-G2PWFileIfMissing {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FileName,

        [Parameter()]
        [string]$UrlPrefix = $G2PWFileUrlPrefix
    )

    $localPath = Join-Path "GPT_SoVITS/text/G2PWModel" $FileName
    $remoteUrl = "$UrlPrefix/$FileName"

    if (Test-Path $localPath -PathType Leaf) {
        Write-Info "File Exists: $localPath"
        return
    }

    $parent = Split-Path -Parent $localPath
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    Write-Info "Downloading G2PWModel/$FileName..."
    Invoke-Download -Uri $remoteUrl -OutFile $localPath
    Write-Success "Downloaded G2PWModel/$FileName"
}

function Download-G2PWFiles {
    Download-G2PWFileIfMissing "MONOPHONIC_CHARS.txt"
    Download-G2PWFileIfMissing "POLYPHONIC_CHARS.txt"
    Download-G2PWFileIfMissing "config.py"
    Download-G2PWFileIfMissing "g2pw.pth"
    Download-G2PWFileIfMissing "record.log"
}

function Download-SharedInferenceFiles {
    Download-RepoFileIfMissing "pretrained_models/chinese-hubert-base/config.json"
    Download-RepoFileIfMissing "pretrained_models/chinese-hubert-base/preprocessor_config.json"
    Download-RepoFileIfMissing "pretrained_models/chinese-hubert-base/pytorch_model.bin"

    Download-RepoFileIfMissing "pretrained_models/chinese-roberta-wwm-ext-large/config.json"
    Download-RepoFileIfMissing "pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin"
    Download-RepoFileIfMissing "pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json"

    Download-RepoFileIfMissing "pretrained_models/fast_langdetect/lid.176.bin"
    Download-RepoFileIfMissing "pretrained_models/fast_langdetect/lid.176.ftz"
}

function Download-VersionFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SelectedVersion
    )

    switch ($SelectedVersion) {
        "v1" {
            Download-RepoFileIfMissing "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
            Download-RepoFileIfMissing "pretrained_models/s2G488k.pth"
        }
        "v2" {
            Download-RepoFileIfMissing "pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
            Download-RepoFileIfMissing "pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        }
        "v2Pro" {
            Download-RepoFileIfMissing "pretrained_models/s1v3.ckpt"
            Download-RepoFileIfMissing "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
            Download-RepoFileIfMissing "pretrained_models/v2Pro/s2Gv2Pro.pth"
        }
        "v2ProPlus" {
            Download-RepoFileIfMissing "pretrained_models/s1v3.ckpt"
            Download-RepoFileIfMissing "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
            Download-RepoFileIfMissing "pretrained_models/v2Pro/s2Gv2ProPlus.pth"
        }
        "v3" {
            Download-RepoFileIfMissing "pretrained_models/s1v3.ckpt"
            Download-RepoFileIfMissing "pretrained_models/s2Gv3.pth"
            Download-RepoFileIfMissing "pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x/config.json"
            Download-RepoFileIfMissing "pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt"
        }
        "v4" {
            Download-RepoFileIfMissing "pretrained_models/s1v3.ckpt"
            Download-RepoFileIfMissing "pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
            Download-RepoFileIfMissing "pretrained_models/gsv-v4-pretrained/vocoder.pth"
        }
        "v5" {
            Download-RepoFileIfMissing "pretrained_models/s1v3.ckpt"
            Download-RepoFileIfMissing "gsv-v5-pretrained/s2Gv5dev.pth" $V5FileUrlPrefix
            Download-RepoFileIfMissing "gsv-v5-pretrained/s2Gv5turbo.pth" $V5FileUrlPrefix
            Download-RepoFileIfMissing "gsv-v5-pretrained/vocoder.pth" $V5FileUrlPrefix
        }
        "all" {
            Download-VersionFiles "v1"
            Download-VersionFiles "v2"
            Download-VersionFiles "v2Pro"
            Download-VersionFiles "v2ProPlus"
            Download-VersionFiles "v3"
            Download-VersionFiles "v4"
            Download-VersionFiles "v5"
        }
        default {
            throw "Unknown version: $SelectedVersion"
        }
    }
}

chcp 65001
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found. Install uv from https://docs.astral.sh/uv/getting-started/installation/"
}
if (-not (Get-Command cl -ErrorAction SilentlyContinue) -and
    -not (Get-Command c++ -ErrorAction SilentlyContinue) -and
    -not (Get-Command gcc -ErrorAction SilentlyContinue) -and
    -not (Get-Command clang -ErrorAction SilentlyContinue)) {
    throw "C/C++ compiler not found. Install Visual Studio Build Tools with the C++ workload before continuing."
}

Write-Info "Syncing locked Python dependencies with uv..."
$uvArgs = @('sync', '--locked')
if ($UvIndexUrl) {
    $uvArgs += @('--default-index', $UvIndexUrl)
}
& uv @uvArgs
if ($LASTEXITCODE -ne 0) {
    throw "uv sync failed with exit code $LASTEXITCODE"
}
Write-Success "Python Dependencies Installed"

$DataDir = if ($env:GPT_SOVITS_DATA_DIR) {
    [System.IO.Path]::GetFullPath($env:GPT_SOVITS_DATA_DIR)
} else {
    $PSScriptRoot
}
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
$env:GPT_SOVITS_DATA_DIR = $DataDir
Set-Location $DataDir

$RepoFileUrlPrefix = ""
$V5FileUrlPrefix = ""
$G2PWFileUrlPrefix = ""
$NLTKURL          = ""
$OpenJTalkURL     = ""

switch ($Source) {
    "HF" {
        Write-Info "Download Model From HuggingFace"
        $RepoFileUrlPrefix = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
        $V5FileUrlPrefix   = "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main"
        $G2PWFileUrlPrefix = "https://huggingface.co/baicai1145/g2pw/resolve/main"
        $NLTKURL           = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
        $OpenJTalkURL      = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
    }
    "HF-Mirror" {
        Write-Info "Download Model From HuggingFace-Mirror"
        $RepoFileUrlPrefix = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
        $V5FileUrlPrefix   = "https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main"
        $G2PWFileUrlPrefix = "https://hf-mirror.com/baicai1145/g2pw/resolve/main"
        $NLTKURL           = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
        $OpenJTalkURL      = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
    }
    "ModelScope" {
        Write-Info "Download Model From ModelScope"
        $RepoFileUrlPrefix = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master"
        $V5FileUrlPrefix   = "https://www.modelscope.cn/models/lj1995/GPT-SoVITS/resolve/master"
        $G2PWFileUrlPrefix = "https://www.modelscope.cn/models/baicai1145/g2pw/resolve/master"
        $NLTKURL           = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip"
        $OpenJTalkURL      = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz"
    }
}

Write-Info "Downloading Shared Inference Resources For Version $Version..."
Download-SharedInferenceFiles
Write-Info "Downloading Version-Specific Inference Weights For $Version..."
Download-VersionFiles $Version
Write-Success "Inference Pretrained Files Downloaded"

Write-Info "Downloading G2PWModel Files..."
Download-G2PWFiles
Write-Success "G2PWModel Files Downloaded"

$python = Join-Path (Join-Path $PSScriptRoot '.venv') (Join-Path 'Scripts' 'python.exe')

Write-Info "Downloading NLTK Data..."
Invoke-Download -Uri $NLTKURL -OutFile "nltk_data.zip"
Invoke-Unzip "nltk_data.zip" (& $python -c "import sys; print(sys.prefix)").Trim()

Write-Info "Downloading Open JTalk Dict..."
Invoke-Download -Uri $OpenJTalkURL -OutFile "open_jtalk_dic_utf_8-1.11.tar.gz"
$target = (& $python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))").Trim()
tar -xzf open_jtalk_dic_utf_8-1.11.tar.gz -C $target
Remove-Item "open_jtalk_dic_utf_8-1.11.tar.gz" -Force
Write-Success "Open JTalk Dic Downloaded"

Write-Success "Installation Completed"
