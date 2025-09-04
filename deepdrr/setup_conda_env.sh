#!/bin/bash
# DeepDRR CUDA 11.8 conda環境セットアップスクリプト  
# md_files/answer1.mdの解決策に基づく

set -e

echo "🐍 DeepDRR CUDA 11.8 conda環境セットアップ"
echo "=============================================="

# 必要なツールの確認
if ! command -v conda &> /dev/null && ! command -v mamba &> /dev/null; then
    echo "❌ condaまたはmambaが見つかりません。"
    echo "📥 Miniconda/Mambaをインストールしてから再実行してください。"
    echo ""
    echo "インストール方法:"
    echo "  wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh"
    echo "  bash Mambaforge-Linux-x86_64.sh"
    exit 1
fi

# mambaが使えるかチェック
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✅ mambaを使用します（高速）"
else
    CONDA_CMD="conda"
    echo "✅ condaを使用します"
fi

ENV_NAME="deepdrr-cuda11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📁 環境設定ファイル: $SCRIPT_DIR/environment-deepdrr-cuda11.yml"
echo "🏷️  環境名: $ENV_NAME"

# 既存の環境があるかチェック
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "⚠️  既存の環境 '$ENV_NAME' が見つかりました。"
    read -p "🤔 削除して再作成しますか？ [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  環境を削除中..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "❌ セットアップを中止しました。"
        exit 1
    fi
fi

echo "🔨 conda環境を作成中..."
cd "$SCRIPT_DIR"
$CONDA_CMD env create -f environment-deepdrr-cuda11.yml

echo "✅ conda環境が作成されました！"
echo ""
echo "📋 使用方法:"
echo "  conda activate $ENV_NAME"
echo "  cd $(pwd)"
echo "  python test_deepdrr_cuda11.py"
echo ""
echo "🔍 環境確認:"
echo "  conda activate $ENV_NAME"
echo "  nvcc --version  # CUDA 11.8であることを確認"
echo "  python -c \"import pycuda.driver as drv; drv.init(); print(f'GPU: {drv.Device(0).name()}')\""
echo ""
echo "⚡ テスト実行例:"
echo "  conda activate $ENV_NAME"
echo "  python test_deepdrr_cuda11.py"