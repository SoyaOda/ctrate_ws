#!/bin/bash
# TotalSegmentator クイックテストスクリプト
# 正しいDockerイメージでの動作確認用

echo "========================================="
echo "TotalSegmentator クイックテスト"
echo "========================================="

# カラー出力の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Docker環境の確認
echo -e "\n${YELLOW}1. Docker環境の確認${NC}"
echo "------------------------"

# Dockerが実行可能か確認
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Dockerが見つかりません${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker OK${NC}"

# GPUが利用可能か確認
echo -e "\n${YELLOW}GPU確認:${NC}"
if docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU利用可能${NC}"
    GPU_FLAG="--gpus all"
else
    echo -e "${YELLOW}! GPUが利用できません。CPUモードで実行します${NC}"
    GPU_FLAG=""
fi

# 2. TotalSegmentator Dockerイメージの取得
echo -e "\n${YELLOW}2. TotalSegmentator Dockerイメージの確認${NC}"
echo "------------------------"

IMAGE_NAME="wasserth/totalsegmentator:2.10.0"

# イメージが既に存在するか確認
if docker images | grep -q "wasserth/totalsegmentator.*2.10.0"; then
    echo -e "${GREEN}✓ イメージが既に存在します: $IMAGE_NAME${NC}"
else
    echo "イメージをダウンロード中: $IMAGE_NAME"
    if docker pull "$IMAGE_NAME"; then
        echo -e "${GREEN}✓ イメージのダウンロード完了${NC}"
    else
        echo -e "${RED}✗ イメージのダウンロードに失敗しました${NC}"
        echo "インターネット接続を確認してください"
        exit 1
    fi
fi

# 3. バージョン確認
echo -e "\n${YELLOW}3. TotalSegmentatorバージョン確認${NC}"
echo "------------------------"

docker run --rm "$IMAGE_NAME" TotalSegmentator --version

# 4. ライセンス設定の確認
echo -e "\n${YELLOW}4. ライセンス設定${NC}"
echo "------------------------"

# デフォルトライセンスキー
DEFAULT_LICENSE_KEY="aca_DHDKRFJMHUX0PK"

if [ -f ~/.totalsegmentator/config.json ] && grep -q "license_number" ~/.totalsegmentator/config.json; then
    echo -e "${GREEN}✓ ライセンスが設定されています${NC}"
    echo "  ライセンス: $(grep license_number ~/.totalsegmentator/config.json | cut -d'"' -f4)"
else
    echo -e "${YELLOW}! ライセンスが設定されていません${NC}"
    echo "自動でライセンスを設定します..."
    
    LICENSE_KEY="$DEFAULT_LICENSE_KEY"
    
    mkdir -p ~/.totalsegmentator
    echo "ライセンスを登録中: $LICENSE_KEY"
    docker run --rm -it $GPU_FLAG \
        -v ~/.totalsegmentator:/root/.totalsegmentator \
        "$IMAGE_NAME" \
        totalseg_set_license -l "$LICENSE_KEY"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ライセンス登録完了${NC}"
    else
        echo -e "${RED}✗ ライセンス登録に失敗しました${NC}"
        echo "手動でライセンスを設定してください"
    fi
fi

# 5. テストデータの確認
echo -e "\n${YELLOW}5. テストデータの確認${NC}"
echo "------------------------"

# CT-RATEデータの検索
CT_FILES=(~/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/*/*/*.nii.gz)

if [ ${#CT_FILES[@]} -gt 0 ] && [ -f "${CT_FILES[0]}" ]; then
    TEST_CT="${CT_FILES[0]}"
    echo -e "${GREEN}✓ テストCTファイル: $TEST_CT${NC}"
else
    echo -e "${YELLOW}! CT-RATEデータが見つかりません${NC}"
    echo "他のNIfTIファイルを検索中..."
    
    # 他のNIfTIファイルを探す
    OTHER_NIFTI=$(find ~/ctrate_ws -name "*.nii.gz" -type f 2>/dev/null | head -1)
    if [ -n "$OTHER_NIFTI" ]; then
        TEST_CT="$OTHER_NIFTI"
        echo -e "${GREEN}代替ファイル: $TEST_CT${NC}"
    else
        echo -e "${RED}✗ テスト用のNIfTIファイルが見つかりません${NC}"
        echo "CTデータを配置してから再実行してください"
        exit 1
    fi
fi

# 6. 簡易テスト実行
echo -e "\n${YELLOW}6. 簡易セグメンテーションテスト${NC}"
echo "------------------------"

OUTPUT_DIR=~/ctrate_ws/outputs/quick_test
mkdir -p "$OUTPUT_DIR"

echo "テスト実行オプション:"
echo "1. 基本的な全身セグメンテーション (total)"
echo "2. 心臓セグメンテーション (heartchambers) ※ライセンス不要"
echo "3. 高精細心臓セグメンテーション (heartchambers_highres) ※ライセンス必要"
echo "4. スキップ"

read -p "選択 (1-4): " CHOICE

case $CHOICE in
    1)
        TASK="total"
        echo "全身セグメンテーションを実行中..."
        ;;
    2)
        TASK="heartchambers"
        echo "心臓セグメンテーションを実行中..."
        ;;
    3)
        TASK="heartchambers_highres"
        echo "高精細心臓セグメンテーションを実行中..."
        ;;
    4)
        echo "テスト実行をスキップしました"
        TASK=""
        ;;
    *)
        echo "無効な選択です"
        TASK=""
        ;;
esac

if [ -n "$TASK" ]; then
    # 実行
    if [ "$GPU_FLAG" ]; then
        DEVICE="gpu"
    else
        DEVICE="cpu"
    fi
    
    echo "実行コマンド:"
    echo "docker run --rm -it $GPU_FLAG --ipc=host \\"
    echo "    -v \"$HOME:$HOME\" \\"
    echo "    \"$IMAGE_NAME\" \\"
    echo "    TotalSegmentator -i \"$TEST_CT\" -o \"$OUTPUT_DIR\" \\"
    echo "    --task $TASK --device $DEVICE"
    
    # タイマー開始
    START_TIME=$SECONDS
    
    docker run --rm -it $GPU_FLAG --ipc=host \
        -v "$HOME:$HOME" \
        "$IMAGE_NAME" \
        TotalSegmentator -i "$TEST_CT" -o "$OUTPUT_DIR" \
        --task $TASK --device $DEVICE
    
    # 実行時間
    ELAPSED_TIME=$((SECONDS - START_TIME))
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ セグメンテーション完了 (${ELAPSED_TIME}秒)${NC}"
        
        # 結果の確認
        echo -e "\n生成されたファイル:"
        ls -la "$OUTPUT_DIR"/*.nii.gz 2>/dev/null | head -10
        
        FILE_COUNT=$(ls "$OUTPUT_DIR"/*.nii.gz 2>/dev/null | wc -l)
        echo -e "\n合計: ${FILE_COUNT} ファイル生成"
    else
        echo -e "\n${RED}✗ セグメンテーションに失敗しました${NC}"
    fi
fi

# 7. サマリー
echo -e "\n${YELLOW}=========================================${NC}"
echo -e "${YELLOW}テスト完了${NC}"
echo -e "${YELLOW}=========================================${NC}"

echo -e "\n環境サマリー:"
echo "- Dockerイメージ: $IMAGE_NAME"
echo "- GPU: $([ "$GPU_FLAG" ] && echo "利用可能" || echo "利用不可")"
echo "- ライセンス: $([ -f ~/.totalsegmentator/config.json ] && grep -q "license_number" ~/.totalsegmentator/config.json && echo "設定済み" || echo "未設定")"
echo "- 出力ディレクトリ: $OUTPUT_DIR"

echo -e "\n次のステップ:"
echo "1. 本番実行: ./scripts/step3_totalsegmentator_test.sh"
echo "2. パイプライン実行: ./scripts/run_pipeline.sh <CTファイル>"

echo -e "\n${GREEN}準備完了！${NC}"