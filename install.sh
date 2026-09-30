#!/bin/bash

# cd into GPT-SoVITS Base Path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

RESET="\033[0m"
BOLD="\033[1m"
ERROR="\033[1;31m[ERROR]: $RESET"
WARNING="\033[1;33m[WARNING]: $RESET"
INFO="\033[1;32m[INFO]: $RESET"
SUCCESS="\033[1;34m[SUCCESS]: $RESET"

set -eE
set -o errtrace

trap 'on_error $LINENO "$BASH_COMMAND" $?' ERR

# shellcheck disable=SC2317
on_error() {
    local lineno="$1"
    local cmd="$2"
    local code="$3"

    echo -e "${ERROR}${BOLD}Command \"${cmd}\" Failed${RESET} at ${BOLD}Line ${lineno}${RESET} with Exit Code ${BOLD}${code}${RESET}"
    echo -e "${ERROR}${BOLD}Call Stack:${RESET}"
    for ((i = ${#FUNCNAME[@]} - 1; i >= 1; i--)); do
        echo -e "  in ${BOLD}${FUNCNAME[i]}()${RESET} at ${BASH_SOURCE[i]}:${BOLD}${BASH_LINENO[i - 1]}${RESET}"
    done
    exit "$code"
}

run_wget_quiet() {
    if wget --tries=25 --wait=5 --read-timeout=40 -q --show-progress "$@" 2>&1; then
        if [ -t 1 ]; then
            tput cuu1 && tput el
        fi
    else
        echo -e "${ERROR} Wget failed"
        exit 1
    fi
}

MODEL_VERSION=""
UV_INDEX_URL=""

USE_HF=false
USE_HF_MIRROR=false
USE_MODELSCOPE=false

print_help() {
    echo "Usage: bash install.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --source   HF|HF-Mirror|ModelScope     Specify the model source (REQUIRED)"
    echo "  --version  v1|v2|v2Pro|v2ProPlus|all"
    echo "                                            Specify which inference pretrained files to download (REQUIRED)"
    echo "  --uv-index-url URL                     Optional default package index mirror for uv"
    echo "  -h, --help                             Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  bash install.sh --source HF --version v2Pro"
    echo "  bash install.sh --source ModelScope --version all"
    echo "  bash install.sh --source HF-Mirror --version v2ProPlus --uv-index-url https://pypi.tuna.tsinghua.edu.cn/simple"
}

# Show help if no arguments provided
if [[ $# -eq 0 ]]; then
    print_help
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
    --source)
        case "$2" in
        HF)
            USE_HF=true
            ;;
        HF-Mirror)
            USE_HF_MIRROR=true
            ;;
        ModelScope)
            USE_MODELSCOPE=true
            ;;
        *)
            echo -e "${ERROR}Error: Invalid Download Source: $2"
            echo -e "${ERROR}Choose From: [HF, HF-Mirror, ModelScope]"
            exit 1
            ;;
        esac
        shift 2
        ;;
    --version)
        case "$2" in
        v1 | v2 | v2Pro | v2ProPlus | all)
            MODEL_VERSION="$2"
            ;;
        *)
            echo -e "${ERROR}Error: Invalid Version: $2"
            echo -e "${ERROR}Choose From: [v1, v2, v2Pro, v2ProPlus, all]"
            exit 1
            ;;
        esac
        shift 2
        ;;
    --uv-index-url | --pip-index-url)
        if [ -z "$2" ]; then
            echo -e "${ERROR}Error: --uv-index-url requires a value"
            exit 1
        fi
        UV_INDEX_URL="$2"
        shift 2
        ;;
    -h | --help)
        print_help
        exit 0
        ;;
    *)
        if [[ "$1" == "--device" ]]; then
            echo -e "${ERROR}Error: --device is no longer supported. This installer now installs CPU-only dependencies."
        else
            echo -e "${ERROR}Unknown Argument: $1"
        fi
        echo ""
        print_help
        exit 1
        ;;
    esac
done

if ! $USE_HF && ! $USE_HF_MIRROR && ! $USE_MODELSCOPE; then
    echo -e "${ERROR}Error: Download Source is REQUIRED"
    echo ""
    print_help
    exit 1
fi

if [ -z "$MODEL_VERSION" ]; then
    echo -e "${ERROR}Error: Version is REQUIRED"
    echo ""
    print_help
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo -e "${ERROR}uv Not Found. Install uv from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if ! command -v cc &>/dev/null && ! command -v c++ &>/dev/null && ! command -v gcc &>/dev/null && ! command -v clang &>/dev/null; then
    echo -e "${ERROR}C/C++ compiler not found. Install Xcode Command Line Tools, GCC, or Clang before continuing."
    exit 1
fi

echo -e "${INFO}Detected system: $(uname -s) $(uname -r) $(uname -m)"
echo -e "${INFO}Syncing locked Python dependencies with uv..."
if [ -n "$UV_INDEX_URL" ]; then
    uv sync --locked --default-index "$UV_INDEX_URL"
else
    uv sync --locked
fi
echo -e "${SUCCESS}Python Dependencies Installed"

DATA_DIR="${GPT_SOVITS_DATA_DIR:-$SCRIPT_DIR}"
mkdir -p "$DATA_DIR"
export GPT_SOVITS_DATA_DIR="$DATA_DIR"
cd "$DATA_DIR" || exit 1

if [ "$USE_HF" = "true" ]; then
    echo -e "${INFO}Download Model From HuggingFace"
    REPO_FILE_URL_PREFIX="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
    G2PW_FILE_URL_PREFIX="https://huggingface.co/baicai1145/g2pw/resolve/main"
    NLTK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    PYOPENJTALK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
elif [ "$USE_HF_MIRROR" = "true" ]; then
    echo -e "${INFO}Download Model From HuggingFace-Mirror"
    REPO_FILE_URL_PREFIX="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
    G2PW_FILE_URL_PREFIX="https://hf-mirror.com/baicai1145/g2pw/resolve/main"
    NLTK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    PYOPENJTALK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
elif [ "$USE_MODELSCOPE" = "true" ]; then
    echo -e "${INFO}Download Model From ModelScope"
    REPO_FILE_URL_PREFIX="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master"
    G2PW_FILE_URL_PREFIX="https://www.modelscope.cn/models/baicai1145/g2pw/resolve/master"
    NLTK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip"
    PYOPENJTALK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz"
fi

download_repo_file_if_missing() {
    local relative_path="$1"
    local local_path="GPT_SoVITS/${relative_path}"
    local remote_url="${REPO_FILE_URL_PREFIX}/${relative_path}"

    if [ -f "$local_path" ]; then
        echo -e "${INFO}File Exists: ${local_path}"
        return
    fi

    mkdir -p "$(dirname "$local_path")"
    echo -e "${INFO}Downloading ${relative_path}..."
    run_wget_quiet "$remote_url" -O "$local_path"
    echo -e "${SUCCESS}Downloaded ${relative_path}"
}

download_g2pw_file_if_missing() {
    local filename="$1"
    local url_prefix="${2:-$G2PW_FILE_URL_PREFIX}"
    local local_path="GPT_SoVITS/text/G2PWModel/${filename}"
    local remote_url="${url_prefix}/${filename}"

    if [ -f "$local_path" ]; then
        echo -e "${INFO}File Exists: ${local_path}"
        return
    fi

    mkdir -p "$(dirname "$local_path")"
    echo -e "${INFO}Downloading G2PWModel/${filename}..."
    run_wget_quiet "$remote_url" -O "$local_path"
    echo -e "${SUCCESS}Downloaded G2PWModel/${filename}"
}

download_g2pw_files() {
    download_g2pw_file_if_missing "MONOPHONIC_CHARS.txt"
    download_g2pw_file_if_missing "POLYPHONIC_CHARS.txt"
    download_g2pw_file_if_missing "config.py"
    download_g2pw_file_if_missing "g2pw.pth"
    download_g2pw_file_if_missing "record.log"
}

download_shared_inference_files() {
    download_repo_file_if_missing "pretrained_models/chinese-hubert-base/config.json"
    download_repo_file_if_missing "pretrained_models/chinese-hubert-base/preprocessor_config.json"
    download_repo_file_if_missing "pretrained_models/chinese-hubert-base/pytorch_model.bin"

    download_repo_file_if_missing "pretrained_models/chinese-roberta-wwm-ext-large/config.json"
    download_repo_file_if_missing "pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin"
    download_repo_file_if_missing "pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json"

    download_repo_file_if_missing "pretrained_models/fast_langdetect/lid.176.bin"
    download_repo_file_if_missing "pretrained_models/fast_langdetect/lid.176.ftz"
}

download_version_files() {
    case "$1" in
    v1)
        download_repo_file_if_missing "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
        download_repo_file_if_missing "pretrained_models/s2G488k.pth"
        ;;
    v2)
        download_repo_file_if_missing "pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        download_repo_file_if_missing "pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        ;;
    v2Pro)
        download_repo_file_if_missing "pretrained_models/s1v3.ckpt"
        download_repo_file_if_missing "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
        download_repo_file_if_missing "pretrained_models/v2Pro/s2Gv2Pro.pth"
        ;;
    v2ProPlus)
        download_repo_file_if_missing "pretrained_models/s1v3.ckpt"
        download_repo_file_if_missing "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
        download_repo_file_if_missing "pretrained_models/v2Pro/s2Gv2ProPlus.pth"
        ;;
    all)
        download_version_files v1
        download_version_files v2
        download_version_files v2Pro
        download_version_files v2ProPlus
        ;;
    *)
        echo -e "${ERROR}Unknown version: $1"
        exit 1
        ;;
    esac
}

echo -e "${INFO}Downloading Shared Inference Resources For Version ${MODEL_VERSION}..."
download_shared_inference_files
echo -e "${INFO}Downloading Version-Specific Inference Weights For ${MODEL_VERSION}..."
download_version_files "$MODEL_VERSION"
echo -e "${SUCCESS}Inference Pretrained Files Downloaded"

echo -e "${INFO}Downloading G2PWModel Files..."
download_g2pw_files
echo -e "${SUCCESS}G2PWModel Files Downloaded"

PYTHON="$SCRIPT_DIR/.venv/bin/python"
PY_PREFIX=$("$PYTHON" -c "import sys; print(sys.prefix)")
PYOPENJTALK_PREFIX=$("$PYTHON" -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))")

echo -e "${INFO}Downloading NLTK Data..."
rm -rf nltk_data.zip
run_wget_quiet "$NLTK_URL" -O nltk_data.zip
"$PYTHON" -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' nltk_data.zip "$PY_PREFIX"
rm -rf nltk_data.zip
echo -e "${SUCCESS}NLTK Data Downloaded"

echo -e "${INFO}Downloading Open JTalk Dict..."
rm -rf open_jtalk_dic_utf_8-1.11.tar.gz
run_wget_quiet "$PYOPENJTALK_URL" -O open_jtalk_dic_utf_8-1.11.tar.gz
tar -xzf open_jtalk_dic_utf_8-1.11.tar.gz -C "$PYOPENJTALK_PREFIX"
rm -rf open_jtalk_dic_utf_8-1.11.tar.gz
echo -e "${SUCCESS}Open JTalk Dic Downloaded"

echo -e "${SUCCESS}Installation Completed"
