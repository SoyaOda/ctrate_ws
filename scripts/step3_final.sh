#!/bin/bash
# STEP3: TotalSegmentator最終版（確実に動作するバージョン）

echo "========================================="
echo "STEP 3: TotalSegmentator実行（最終版）"
echo "========================================="

# 設定
DOCKER_IMAGE="wasserth/totalsegmentator:2.10.0"
CT_FILE="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/step3_final"

# カラー出力
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 入力ファイル確認
if [ ! -f "$CT_FILE" ]; then
    echo -e "${RED}エラー: CTファイルが見つかりません${NC}"
    exit 1
fi

echo "入力CT: $CT_FILE"
echo "出力先: $OUTPUT_DIR"
echo ""

# 既存の出力をクリーンアップ
if [ -d "$OUTPUT_DIR" ]; then
    echo "既存の出力ディレクトリをクリーンアップ中..."
    sudo rm -rf "$OUTPUT_DIR" 2>/dev/null || rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# TotalSegmentator実行（debug_testと同じ設定）
echo "TotalSegmentatorを実行中..."
echo "（初回実行時はモデルのダウンロードが必要です）"
echo ""

docker run --rm --gpus all --ipc=host \
    -v /home/soya:/home/soya \
    wasserth/totalsegmentator:2.10.0 \
    TotalSegmentator \
    -i "$CT_FILE" \
    -o "$OUTPUT_DIR/total" \
    --task total \
    --device gpu \
    --fast

echo ""
echo "========================================="
echo "結果確認"
echo "========================================="

# 結果確認
if [ -d "$OUTPUT_DIR/total" ]; then
    FILE_COUNT=$(ls "$OUTPUT_DIR/total" 2>/dev/null | wc -l)
    if [ "$FILE_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ セグメンテーション成功！${NC}"
        echo "生成ファイル数: $FILE_COUNT"
        echo ""
        echo "生成された臓器マスク:"
        ls -lh "$OUTPUT_DIR/total/" | head -20
        
        # 統計情報
        echo ""
        echo "ディスク使用量:"
        du -sh "$OUTPUT_DIR/total"
        
        # 重要な臓器の確認
        echo ""
        echo "主要臓器の確認:"
        for organ in liver kidney_left kidney_right lung_upper_lobe_left spleen; do
            if [ -f "$OUTPUT_DIR/total/${organ}.nii.gz" ]; then
                echo -e "  ${GREEN}✓${NC} ${organ}.nii.gz"
            else
                echo -e "  ${RED}✗${NC} ${organ}.nii.gz"
            fi
        done
        
        echo ""
        echo -e "${GREEN}=========================================${NC}"
        echo -e "${GREEN}STEP 3 完了！${NC}"
        echo -e "${GREEN}=========================================${NC}"
        echo ""
        echo "出力ディレクトリ: $OUTPUT_DIR/total"
        echo "次のステップ: STEP 4 (BOAセットアップ)"
    else
        echo -e "${RED}エラー: ファイルが生成されませんでした${NC}"
        exit 1
    fi
else
    echo -e "${RED}エラー: 出力ディレクトリが作成されませんでした${NC}"
    echo "Dockerログを確認してください"
    exit 1
fi