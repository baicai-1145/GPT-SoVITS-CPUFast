Param (
    [Parameter(Mandatory=$true)][ValidateSet("HF", "HF-Mirror", "ModelScope")][string]$Source,
    [Parameter(Mandatory=$true)][ValidateSet("v1", "v2", "v2Pro", "v2ProPlus", "all")][string]$Version
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


function Invoke-Conda {
    param (
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )

    $output = & conda install -y -q -c conda-forge @Args 2>&1
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "Conda Install $Args Failed" -ForegroundColor Red
        $errorMessages = @()
        foreach ($item in $output) {
            if ($item -is [System.Management.Automation.ErrorRecord]) {
                $msg = $item.Exception.Message
                Write-Host "$msg" -ForegroundColor Red
                $errorMessages += $msg
            }
            else {
                Write-Host $item
                $errorMessages += $item
            }
        }
        throw [System.Exception]::new(($errorMessages -join "`n"))
    }
}

function Invoke-Pip {
    param (
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )
    
    $output = & pip install @Args 2>&1
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -ne 0) {
        $errorMessages = @()
        Write-Host "Pip Install $Args Failed" -ForegroundColor Red
        foreach ($item in $output) {
            if ($item -is [System.Management.Automation.ErrorRecord]) {
                $msg = $item.Exception.Message
                Write-Host "$msg" -ForegroundColor Red
                $errorMessages += $msg
            }
            else {
                Write-Host $item
                $errorMessages += $item
            }
        }
        throw [System.Exception]::new(($errorMessages -join "`n"))
    }
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
        [string]$RelativePath
    )

    $localPath = Join-Path "GPT_SoVITS" $RelativePath
    $remoteUrl = "$RepoFileUrlPrefix/$RelativePath"

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
        "all" {
            Download-VersionFiles "v1"
            Download-VersionFiles "v2"
            Download-VersionFiles "v2Pro"
            Download-VersionFiles "v2ProPlus"
        }
        default {
            throw "Unknown version: $SelectedVersion"
        }
    }
}

chcp 65001
Set-Location $PSScriptRoot

Write-Info "Installing FFmpeg & CMake..."
Invoke-Conda  ffmpeg cmake
Write-Success "FFmpeg & CMake Installed"

$RepoFileUrlPrefix = ""
$G2PWURL          = ""
$NLTKURL          = ""
$OpenJTalkURL     = ""

switch ($Source) {
    "HF" {
        Write-Info "Download Model From HuggingFace"
        $RepoFileUrlPrefix = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
        $G2PWURL           = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
        $NLTKURL           = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
        $OpenJTalkURL      = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
    }
    "HF-Mirror" {
        Write-Info "Download Model From HuggingFace-Mirror"
        $RepoFileUrlPrefix = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
        $G2PWURL           = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
        $NLTKURL           = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
        $OpenJTalkURL      = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
    }
    "ModelScope" {
        Write-Info "Download Model From ModelScope"
        $RepoFileUrlPrefix = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master"
        $G2PWURL           = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
        $NLTKURL           = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip"
        $OpenJTalkURL      = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz"
    }
}

Write-Info "Downloading Shared Inference Resources For Version $Version..."
Download-SharedInferenceFiles
Write-Info "Downloading Version-Specific Inference Weights For $Version..."
Download-VersionFiles $Version
Write-Success "Inference Pretrained Files Downloaded"

if (-not (Test-Path "GPT_SoVITS/text/G2PWModel")) {
    Write-Info "Downloading G2PWModel..."
    Invoke-Download -Uri $G2PWURL -OutFile "G2PWModel.zip"
    Invoke-Unzip "G2PWModel.zip" "GPT_SoVITS/text"
    Write-Success "G2PWModel Downloaded"
} else {
    Write-Info "G2PWModel Exists"
    Write-Info "Skip Downloading G2PWModel"
}

Write-Info "Installing PyTorch For CPU..."
Invoke-Pip torch --index-url "https://download.pytorch.org/whl/cpu"
Write-Success "PyTorch Installed"

Write-Info "Installing Python Dependencies From requirements.txt..."
Invoke-Pip -r requirements.txt
Write-Success "Python Dependencies Installed"

Write-Info "Downloading NLTK Data..."
Invoke-Download -Uri $NLTKURL -OutFile "nltk_data.zip"
Invoke-Unzip "nltk_data.zip" (python -c "import sys; print(sys.prefix)").Trim()

Write-Info "Downloading Open JTalk Dict..."
Invoke-Download -Uri $OpenJTalkURL -OutFile "open_jtalk_dic_utf_8-1.11.tar.gz"
$target = (python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))").Trim()
tar -xzf open_jtalk_dic_utf_8-1.11.tar.gz -C $target
Remove-Item "open_jtalk_dic_utf_8-1.11.tar.gz" -Force
Write-Success "Open JTalk Dic Downloaded"

Write-Success "Installation Completed"
