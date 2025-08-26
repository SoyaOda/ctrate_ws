#!/bin/bash
# STEP3: TotalSegmentator公式Dockerで心臓セグメンテーション

set -e

echo "========================================="
echo "STEP 3: TotalSegmentator公式Docker"
echo "========================================="

# カラー出力
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 設定
LICENSE_KEY="aca_DHDKRFJMHUX0PK"
DOCKER_IMAGE="wasserth/totalsegmentator:2.10.0"

# 1. Dockerイメージのダウンロード
echo ""
echo "1. Dockerイメージをダウンロード"
echo "------------------------"
if ! docker images | grep -q "wasserth/totalsegmentator.*2.10.0"; then
    echo "イメージをダウンロード中..."
    docker pull "$DOCKER_IMAGE"
else
    echo -e "${GREEN}✓ イメージが既に存在します${NC}"
fi

# 2. ライセンス設定ディレクトリ
echo ""
echo "2. ライセンス設定"
echo "------------------------"
mkdir -p ~/.totalsegmentator

# ライセンス設定（既に設定済みの場合はスキップ）
if [ -f ~/.totalsegmentator/config.json ] && grep -q "$LICENSE_KEY" ~/.totalsegmentator/config.json; then
    echo -e "${GREEN}✓ ライセンスは既に設定されています${NC}"
else
    echo "ライセンスファイルを手動で作成します..."
    echo "$LICENSE_KEY" > ~/.totalsegmentator/license.txt
    echo "$LICENSE_KEY" > ~/.totalsegmentator/nnunet_license.txt
    cat > ~/.totalsegmentator/config.json << EOF
{
    "totalseg_id": "totalseg_GDZCPD8K",
    "send_usage_stats": true,
    "prediction_counter": 0,
    "license_number": "$LICENSE_KEY"
}
EOF
    echo -e "${GREEN}✓ ライセンスファイルを作成しました${NC}"
fi

# 3. テストデータで心臓セグメンテーション実行
echo ""
echo "3. heartchambers_highresセグメンテーション実行"
echo "------------------------"

# 入力ファイルと出力ディレクトリ
CT_FILE="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/ts_test"

# ファイル存在確認
if [ ! -f "$CT_FILE" ]; then
    echo -e "${RED}✗ 入力CTファイルが見つかりません: $CT_FILE${NC}"
    exit 1
fi

echo "入力: $CT_FILE"
echo "出力: $OUTPUT_DIR"
echo ""

# 出力ディレクトリを準備
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# セグメンテーション実行
echo "実行中..."
echo "注意: heartchambers_highresタスクは大きなモデルをダウンロードする必要があります（初回のみ）"
echo ""

# まず通常のtotalタスクで基本セグメンテーション
echo "基本セグメンテーションを実行中..."
docker run --rm --gpus all --ipc=host \
    -v "$HOME:$HOME" \
    -v ~/.totalsegmentator:/root/.totalsegmentator \
    "$DOCKER_IMAGE" \
    TotalSegmentator -i "$CT_FILE" -o "$OUTPUT_DIR" \
    --task total --device gpu --fast

# heartchambers_highresも試行（ライセンスが必要）
echo ""
echo "心臓専用セグメンテーションを試行中..."
docker run --rm --gpus all --ipc=host \
    -v "$HOME:$HOME" \
    -v ~/.totalsegmentator:/root/.totalsegmentator \
    "$DOCKER_IMAGE" \
    TotalSegmentator -i "$CT_FILE" -o "${OUTPUT_DIR}_heart" \
    --task heartchambers_highres --device gpu \
    -l "$LICENSE_KEY" 2>&1 | grep -v "^Downloading:" || {
        echo -e "${YELLOW}注意: heartchambers_highresはライセンス制限があります${NC}"
        echo "基本セグメンテーションの結果を使用します"
    }

# 4. 結果確認
echo ""
echo "4. 生成物の確認"
echo "------------------------"

if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${GREEN}✓ 出力ディレクトリが作成されました${NC}"
    echo ""
    
    # 心臓関連ファイルを確認
    echo "心臓関連ファイル:"
    ls "$OUTPUT_DIR" 2>/dev/null | grep -E "heart|atrium|ventricle|myocardium|aorta" || {
        echo -e "${YELLOW}心臓専用ファイルが見つかりません。全臓器を確認:${NC}"
        ls "$OUTPUT_DIR" | head -10
    }
    
    echo ""
    echo "生成ファイル数: $(ls "$OUTPUT_DIR" 2>/dev/null | wc -l)"
    
    # ファイルサイズも確認
    echo ""
    echo "ファイルサイズ:"
    du -sh "$OUTPUT_DIR"/* 2>/dev/null | head -5
else
    echo -e "${RED}✗ 出力ディレクトリが作成されませんでした${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo -e "${GREEN}STEP 3 完了！${NC}"
echo "次のステップ:"
echo "  STEP 4: ./scripts/step4_boa_setup.sh"
echo "========================================="