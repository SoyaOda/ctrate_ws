#!/bin/bash
# DeepDRR CUDA 11.8 Docker環境実行スクリプト
# md_files/answer1.mdの解決策に基づく

set -e

echo "🐳 DeepDRR CUDA 11.8 Docker環境セットアップ"
echo "================================================"

# Docker imageの名前
IMAGE_NAME="deepdrr-cuda11"
CONTAINER_NAME="deepdrr-test"

# 現在のディレクトリ
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 プロジェクトルート: $PROJECT_ROOT"
echo "📁 DeepDRRディレクトリ: $SCRIPT_DIR"

# Dockerイメージをビルド
echo "🔨 Dockerイメージをビルド中..."
cd "$SCRIPT_DIR"
docker build -f Dockerfile.cuda11 -t "$IMAGE_NAME" .

# 既存のコンテナがあれば削除
if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🗑️  既存のコンテナを削除中..."
    docker rm -f "$CONTAINER_NAME"
fi

echo "🚀 Dockerコンテナを実行中..."

# CTデータとスクリプトをマウントしてコンテナ実行
docker run -it --gpus all \
    --name "$CONTAINER_NAME" \
    -v "$PROJECT_ROOT/data:/workspace/data:ro" \
    -v "$PROJECT_ROOT/outputs:/workspace/outputs:ro" \
    -v "$SCRIPT_DIR:/workspace/deepdrr" \
    -w /workspace/deepdrr \
    "$IMAGE_NAME" bash -c "
        echo '🔍 CUDA環境確認'
        ./check_cuda.sh
        echo ''
        echo '🧪 DeepDRRテスト実行'
        python3 test_deepdrr_cuda11.py
        echo ''
        echo '💾 出力ファイル確認'
        ls -la outputs/
        echo ''
        echo '✅ テスト完了。結果を確認してください。'
        echo '📁 出力: deepdrr/outputs/'
        exec bash
    "

echo ""
echo "📋 使用方法:"
echo "  - コンテナが起動したら、bash内で追加のテストが可能です"
echo "  - 出力ファイルは deepdrr/outputs/ に保存されます"
echo "  - 終了するには 'exit' を入力してください"