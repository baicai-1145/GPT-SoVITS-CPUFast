$ErrorActionPreference = "Stop"

$date = $env:DATE_SUFFIX
if ([string]::IsNullOrWhiteSpace($date)) {
    $date = Get-Date -Format "MMdd"
}

$suffix = $env:PKG_SUFFIX
$artifactLabel = "gpt-sovits-cpufast-$date"
if (-not [string]::IsNullOrWhiteSpace($suffix)) {
    $artifactLabel = "$artifactLabel$suffix"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required to build this project."
}

Write-Host "[INFO] Building the locked Python wheel..."
if (Test-Path "dist") {
    Remove-Item "dist" -Recurse -Force
}
& uv build --wheel
if ($LASTEXITCODE -ne 0) {
    throw "uv build failed with exit code $LASTEXITCODE"
}

$wheel = Get-ChildItem "dist\*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $wheel) {
    throw "uv build did not produce a wheel."
}

$artifact = $wheel.FullName
Write-Host "[SUCCESS] Built $artifact (release label: $artifactLabel)"

$msUser = $env:MODELSCOPE_USERNAME
$msToken = $env:MODELSCOPE_TOKEN
if ($msUser -and $msToken) {
    Write-Host "[INFO] Uploading wheel to ModelScope..."
    & uvx --from modelscope modelscope upload `
        "$msUser/GPT-SoVITS-Packages" `
        $artifact `
        "$(Split-Path $artifact -Leaf)" `
        --repo-type model `
        --token $msToken
    if ($LASTEXITCODE -ne 0) {
        throw "ModelScope upload failed with exit code $LASTEXITCODE"
    }
}

$hfUser = $env:HUGGINGFACE_USERNAME
$hfToken = $env:HUGGINGFACE_TOKEN
if ($hfUser -and $hfToken) {
    Write-Host "[INFO] Uploading wheel to Hugging Face..."
    & uvx --from "huggingface_hub[hf_transfer]" hf upload `
        "$hfUser/GPT-SoVITS-Packages" `
        $artifact `
        "$(Split-Path $artifact -Leaf)" `
        --repo-type model `
        --token $hfToken
    if ($LASTEXITCODE -ne 0) {
        throw "Hugging Face upload failed with exit code $LASTEXITCODE"
    }
}
