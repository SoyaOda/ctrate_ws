#!/bin/bash
# PA左右反転修正版テスト実行スクリプト
# 元のrun_deepdrr_docker.shをベースに修正

set -e

echo "🐳 PA左右反転修正版 DeepDRR テスト"
echo "================================================"

# Docker imageの名前
IMAGE_NAME="deepdrr-cuda11"
CONTAINER_NAME="deepdrr-pa-test"

# 現在のディレクトリ
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 プロジェクトルート: $PROJECT_ROOT"
echo "📁 DeepDRRディレクトリ: $SCRIPT_DIR"

# Dockerイメージの存在確認
if ! docker images | grep -q "$IMAGE_NAME"; then
    echo "❌ Dockerイメージ '$IMAGE_NAME' が見つかりません"
    echo "   run_deepdrr_docker.sh を先に実行してイメージをビルドしてください"
    exit 1
fi

# 既存のコンテナがあれば削除
if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🗑️  既存のコンテナを削除中..."
    docker rm -f "$CONTAINER_NAME"
fi

echo "🚀 PA左右反転修正版テストを実行中..."

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
        echo '🧪 PA左右反転修正版テスト実行'
        python3 test_deepdrr_final_correct.py
        echo ''
        echo '💾 出力ファイル確認'
        ls -la outputs/
        echo ''
        echo '✅ PA修正テスト完了。結果を確認してください。'
        echo '📁 出力: deepdrr/outputs/'
        echo '📊 確認項目:'
        echo '  - PA画像の心陰影が右側に配置されているか'
        echo '  - LAT画像は変更されていないか'
        echo '  - γ=90°による縦向きが維持されているか'
        exec bash
    "

echo ""
echo "📋 使用方法:"
echo "  - コンテナが起動したら、bash内で追加のテストが可能です"
echo "  - 出力ファイルは deepdrr/outputs/ に保存されます"
echo "  - 終了するには 'exit' を入力してください"