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

run_conda_quiet() {
    local output
    output=$(conda install --yes --quiet -c conda-forge "$@" 2>&1) || {
        echo -e "${ERROR} Conda install failed:\n$output"
        exit 1
    }
}

run_pip_quiet() {
    local output
    output=$(pip install "$@" 2>&1) || {
        echo -e "${ERROR} Pip install failed:\n$output"
        exit 1
    }
}

run_wget_quiet() {
    if wget --tries=25 --wait=5 --read-timeout=40 -q --show-progress "$@" 2>&1; then
        if [ "$WORKFLOW" = "false" ]; then
            tput cuu1 && tput el
        fi
    else
        echo -e "${ERROR} Wget failed"
        exit 1
    fi
}

WORKFLOW=${WORKFLOW:-"false"}
MODEL_VERSION=""
USE_INT8=false

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
    echo "  --int8                                  Download INT8 quantized models (g2pw, bert, cnhubert)"
    echo "                                            Reduces memory ~3GB, requires ~534MB instead of ~2.2GB"
    echo "  -h, --help                             Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  bash install.sh --source HF --version v2Pro"
    echo "  bash install.sh --source ModelScope --version v2ProPlus --int8"
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
    --int8)
        USE_INT8=true
        shift
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

if ! command -v conda &>/dev/null; then
    echo -e "${ERROR}Conda Not Found"
    exit 1
fi

case "$(uname -m)" in
x86_64 | amd64) SYSROOT_PKG="sysroot_linux-64>=2.28" ;;
aarch64 | arm64) SYSROOT_PKG="sysroot_linux-aarch64>=2.28" ;;
ppc64le) SYSROOT_PKG="sysroot_linux-ppc64le>=2.28" ;;
*)
    echo "Unsupported architecture: $(uname -m)"
    exit 1
    ;;
esac

# Install build tools
echo -e "${INFO}Detected system: $(uname -s) $(uname -r) $(uname -m)"
if [ "$(uname)" != "Darwin" ]; then
    gcc_major_version=$(command -v gcc >/dev/null 2>&1 && gcc -dumpversion | cut -d. -f1 || echo 0)
    if [ "$gcc_major_version" -lt 11 ]; then
        echo -e "${INFO}Installing GCC & G++..."
        run_conda_quiet gcc=11 gxx=11
        run_conda_quiet "$SYSROOT_PKG"
        echo -e "${SUCCESS}GCC & G++ Installed..."
    else
        echo -e "${INFO}Detected GCC Version: $gcc_major_version"
        echo -e "${INFO}Skip Installing GCC & G++ From Conda-Forge"
        echo -e "${INFO}Installing libstdcxx-ng From Conda-Forge"
        run_conda_quiet "libstdcxx-ng>=$gcc_major_version"
        echo -e "${SUCCESS}libstdcxx-ng=$gcc_major_version Installed..."
    fi
else
    if ! xcode-select -p &>/dev/null; then
        echo -e "${INFO}Installing Xcode Command Line Tools..."
        xcode-select --install
        echo -e "${INFO}Waiting For Xcode Command Line Tools Installation Complete..."
        while true; do
            sleep 20

            if xcode-select -p &>/dev/null; then
                echo -e "${SUCCESS}Xcode Command Line Tools Installed"
                break
            else
                echo -e "${INFO}Installing，Please Wait..."
            fi
        done
    else
        XCODE_PATH=$(xcode-select -p)
        if [[ "$XCODE_PATH" == *"Xcode.app"* ]]; then
            echo -e "${WARNING} Detected Xcode path: $XCODE_PATH"
            echo -e "${WARNING} If your Xcode version does not match your macOS version, it may cause unexpected issues during compilation or package builds."
        fi
    fi
fi

echo -e "${INFO}Installing FFmpeg & CMake..."
run_conda_quiet ffmpeg cmake make
echo -e "${SUCCESS}FFmpeg & CMake Installed"

echo -e "${INFO}Installing unzip..."
run_conda_quiet unzip
echo -e "${SUCCESS}unzip Installed"

if [ "$USE_HF" = "true" ]; then
    echo -e "${INFO}Download Model From HuggingFace"
    REPO_FILE_URL_PREFIX="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
    G2PW_URL="https://huggingface.co/baicai1145/g2pw/resolve/main/g2pw.pth"
    G2PW_INT8_URL="https://huggingface.co/baicai1145/g2pw-int8/resolve/main/g2pw_int8.pth"
    BERT_INT8_URL="https://huggingface.co/baicai1145/chinese-roberta-wwm-ext-large-22L-int8/resolve/main/bert_large_int8.pth"
    CNHUBERT_INT8_URL="https://huggingface.co/baicai1145/chinese-hubert-base-int8/resolve/main/cnhubert_int8.pth"
    NLTK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    PYOPENJTALK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
elif [ "$USE_HF_MIRROR" = "true" ]; then
    echo -e "${INFO}Download Model From HuggingFace-Mirror"
    REPO_FILE_URL_PREFIX="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
    G2PW_URL="https://hf-mirror.com/baicai1145/g2pw/resolve/main/g2pw.pth"
    G2PW_INT8_URL="https://hf-mirror.com/baicai1145/g2pw-int8/resolve/main/g2pw_int8.pth"
    BERT_INT8_URL="https://hf-mirror.com/baicai1145/chinese-roberta-wwm-ext-large-22L-int8/resolve/main/bert_large_int8.pth"
    CNHUBERT_INT8_URL="https://hf-mirror.com/baicai1145/chinese-hubert-base-int8/resolve/main/cnhubert_int8.pth"
    NLTK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    PYOPENJTALK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
elif [ "$USE_MODELSCOPE" = "true" ]; then
    echo -e "${INFO}Download Model From ModelScope"
    REPO_FILE_URL_PREFIX="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master"
    G2PW_URL="https://www.modelscope.cn/models/baicai1145/g2pw/resolve/master/g2pw.pth"
    G2PW_INT8_URL="https://www.modelscope.cn/models/baicai1145/g2pw-int8/resolve/master/g2pw_int8.pth"
    BERT_INT8_URL="https://www.modelscope.cn/models/baicai1145/chinese-roberta-wwm-ext-large-22L-int8/resolve/master/bert_large_int8.pth"
    CNHUBERT_INT8_URL="https://www.modelscope.cn/models/baicai1145/chinese-hubert-base-int8/resolve/master/cnhubert_int8.pth"
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

download_file_if_missing() {
    local local_path="$1"
    local remote_url="$2"

    if [ -f "$local_path" ]; then
        echo -e "${INFO}File Exists: ${local_path}"
        return
    fi

    mkdir -p "$(dirname "$local_path")"
    echo -e "${INFO}Downloading $(basename "$local_path")..."
    run_wget_quiet "$remote_url" -O "$local_path"
    echo -e "${SUCCESS}Downloaded $(basename "$local_path")"
}

download_shared_inference_files() {
    if [ "$USE_INT8" = "true" ]; then
        # INT8 模式: 下载量化模型, 不需要 FP32 的 bert/cnhubert 权重
        echo -e "${INFO}INT8 mode: Downloading quantized models..."

        # cnhubert: 只需要 config (tokenizer 不需要), INT8 权重
        download_repo_file_if_missing "pretrained_models/chinese-hubert-base/config.json"
        download_repo_file_if_missing "pretrained_models/chinese-hubert-base/preprocessor_config.json"
        download_file_if_missing "GPT_SoVITS/pretrained_models/chinese-hubert-base/cnhubert_int8.pth" "$CNHUBERT_INT8_URL"

        # bert: 只需要 tokenizer.json 和 config, INT8 权重
        download_repo_file_if_missing "pretrained_models/chinese-roberta-wwm-ext-large/config.json"
        download_repo_file_if_missing "pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json"
        mkdir -p "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large-22L"
        download_file_if_missing "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large-22L/bert_large_int8.pth" "$BERT_INT8_URL"
    else
        # FP32 模式: 下载完整模型
        download_repo_file_if_missing "pretrained_models/chinese-hubert-base/config.json"
        download_repo_file_if_missing "pretrained_models/chinese-hubert-base/preprocessor_config.json"
        download_repo_file_if_missing "pretrained_models/chinese-hubert-base/pytorch_model.bin"

        download_repo_file_if_missing "pretrained_models/chinese-roberta-wwm-ext-large/config.json"
        download_repo_file_if_missing "pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin"
        download_repo_file_if_missing "pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json"
    fi

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

# G2PW: 下载 torch 权重 (从 baicai1145/g2pw)
if [ "$USE_INT8" = "true" ]; then
    # INT8 模式: 下载 g2pw INT8 量化权重
    if [ ! -f "GPT_SoVITS/text/G2PWModel/g2pW_int8.pth" ] && [ ! -f "GPT_SoVITS/text/G2PWModel/g2pw_int8.pth" ]; then
        echo -e "${INFO}Downloading G2PW INT8 Model..."
        mkdir -p GPT_SoVITS/text/G2PWModel
        download_file_if_missing "GPT_SoVITS/text/G2PWModel/g2pw_int8.pth" "$G2PW_INT8_URL"
        echo -e "${SUCCESS}G2PW INT8 Model Downloaded"
    else
        echo -e "${INFO}G2PW INT8 Model Exists"
    fi
fi

# G2PW torch 权重 (FP32, 量化和推理都需要作为 fallback)
if [ ! -f "GPT_SoVITS/text/G2PWModel/g2pw.pth" ] && [ ! -f "GPT_SoVITS/text/G2PWModel/g2pW.pth" ]; then
    echo -e "${INFO}Downloading G2PW PyTorch Model..."
    mkdir -p GPT_SoVITS/text/G2PWModel
    download_file_if_missing "GPT_SoVITS/text/G2PWModel/g2pw.pth" "$G2PW_URL"
    echo -e "${SUCCESS}G2PW PyTorch Model Downloaded"
else
    echo -e "${INFO}G2PW Model Exists"
fi

# G2PW 静态资源 (POLYPHONIC_CHARS.txt 等)
if [ ! -f "GPT_SoVITS/text/G2PWModel/POLYPHONIC_CHARS.txt" ]; then
    echo -e "${INFO}Downloading G2PW Static Assets..."
    if [ "$USE_HF" = "true" ]; then
        G2PW_ASSETS_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    elif [ "$USE_HF_MIRROR" = "true" ]; then
        G2PW_ASSETS_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    elif [ "$USE_MODELSCOPE" = "true" ]; then
        G2PW_ASSETS_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
    fi
    rm -rf G2PWModel.zip
    run_wget_quiet "$G2PW_ASSETS_URL"
    unzip -q -o G2PWModel.zip -d GPT_SoVITS/text
    rm -rf G2PWModel.zip
    echo -e "${SUCCESS}G2PW Static Assets Downloaded"
else
    echo -e "${INFO}G2PW Static Assets Exist"
fi

if [ "$WORKFLOW" = false ]; then
    echo -e "${INFO}Installing PyTorch For CPU..."
    run_pip_quiet torch --index-url "https://download.pytorch.org/whl/cpu"
fi
echo -e "${SUCCESS}PyTorch Installed"

echo -e "${INFO}Installing Python Dependencies From requirements.txt..."

hash -r

run_pip_quiet -r requirements.txt

echo -e "${SUCCESS}Python Dependencies Installed"

PY_PREFIX=$(python -c "import sys; print(sys.prefix)")
PYOPENJTALK_PREFIX=$(python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))")

echo -e "${INFO}Downloading NLTK Data..."
rm -rf nltk_data.zip
run_wget_quiet "$NLTK_URL" -O nltk_data.zip
unzip -q -o nltk_data.zip -d "$PY_PREFIX"
rm -rf nltk_data.zip
echo -e "${SUCCESS}NLTK Data Downloaded"

echo -e "${INFO}Downloading Open JTalk Dict..."
rm -rf open_jtalk_dic_utf_8-1.11.tar.gz
run_wget_quiet "$PYOPENJTALK_URL" -O open_jtalk_dic_utf_8-1.11.tar.gz
tar -xzf open_jtalk_dic_utf_8-1.11.tar.gz -C "$PYOPENJTALK_PREFIX"
rm -rf open_jtalk_dic_utf_8-1.11.tar.gz
echo -e "${SUCCESS}Open JTalk Dic Downloaded"

if [ "$USE_INT8" = "true" ]; then
    echo -e "${SUCCESS}Installation Completed (INT8 quantized mode)"
    echo -e "${INFO}INT8 models will be auto-detected at runtime. Memory savings: ~3GB"
else
    echo -e "${SUCCESS}Installation Completed"
fi
