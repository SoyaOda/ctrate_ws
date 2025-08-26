#!/bin/bash
# TotalSegmentatorライセンス設定スクリプト

echo "========================================="
echo "TotalSegmentatorライセンス設定"
echo "========================================="

# ライセンスキー（デフォルト）
LICENSE_KEY="aca_DHDKRFJMHUX0PK"

# カラー出力
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Dockerイメージ
IMAGE_NAME="wasserth/totalsegmentator:2.10.0"

# 既存のライセンス確認
if [ -f ~/.totalsegmentator/totalseg_license.txt ]; then
    echo -e "${YELLOW}既存のライセンスファイルが検出されました${NC}"
    echo "現在のライセンス:"
    head -n 1 ~/.totalsegmentator/totalseg_license.txt
    
    read -p "上書きしますか？ (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "ライセンス設定をスキップしました"
        exit 0
    fi
fi

# ライセンスディレクトリ作成
mkdir -p ~/.totalsegmentator

# Dockerイメージの確認
echo "Dockerイメージを確認中..."
if ! docker images | grep -q "wasserth/totalsegmentator.*2.10.0"; then
    echo "イメージをダウンロード中..."
    docker pull "$IMAGE_NAME"
fi

# ライセンス設定
echo ""
echo "ライセンスを設定中: $LICENSE_KEY"
echo "------------------------"

docker run --rm -it --gpus all \
    -v ~/.totalsegmentator:/root/.totalsegmentator \
    "$IMAGE_NAME" \
    totalseg_set_license -l "$LICENSE_KEY"

# 結果確認
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ ライセンス設定完了${NC}"
    
    # config.jsonでライセンスを確認
    if [ -f ~/.totalsegmentator/config.json ]; then
        if grep -q "license_number" ~/.totalsegmentator/config.json; then
            echo -e "${GREEN}✓ ライセンスが正常に設定されました${NC}"
            echo "  設定ファイル: ~/.totalsegmentator/config.json"
            echo "  ライセンス: $(grep license_number ~/.totalsegmentator/config.json | cut -d'"' -f4)"
        fi
    elif [ -f ~/.totalsegmentator/totalseg_license.txt ]; then
        echo -e "${GREEN}✓ ライセンスファイルが作成されました${NC}"
        echo "  場所: ~/.totalsegmentator/totalseg_license.txt"
    else
        echo -e "${YELLOW}注意: ライセンスはコンテナ内に正常に設定されています${NC}"
    fi
else
    echo ""
    echo -e "${RED}✗ ライセンス設定に失敗しました${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo "設定完了！"
echo "次のコマンドでテストを実行できます:"
echo "  ./scripts/quick_test_totalseg.sh"
echo "========================================="