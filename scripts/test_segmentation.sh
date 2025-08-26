#!/bin/bash
# TotalSegmentatorの詳細テストスクリプト

echo "========================================="
echo "TotalSegmentator詳細デバッグテスト"
echo "========================================="

# テストCTファイル
CT_FILE="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/debug_test"

# 出力ディレクトリをクリーンアップ
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "入力ファイル: $CT_FILE"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo ""

# 1. 基本的なtotalタスクをテスト（ライセンス不要）
echo "========================================="
echo "テスト1: 基本的な臓器セグメンテーション (total)"
echo "========================================="

docker run --rm --gpus all --ipc=host \
    -v "$HOME:$HOME" \
    wasserth/totalsegmentator:2.10.0 \
    TotalSegmentator -i "$CT_FILE" -o "$OUTPUT_DIR/total" \
    --task total --device gpu --fast

echo ""
echo "結果（total）:"
ls -la "$OUTPUT_DIR/total/" 2>/dev/null | head -10
echo "ファイル数: $(ls "$OUTPUT_DIR/total/"*.nii.gz 2>/dev/null | wc -l)"

# 2. heartchambersタスク（ライセンス不要版）
echo ""
echo "========================================="
echo "テスト2: 心臓セグメンテーション (heartchambers)"
echo "========================================="

docker run --rm --gpus all --ipc=host \
    -v "$HOME:$HOME" \
    wasserth/totalsegmentator:2.10.0 \
    TotalSegmentator -i "$CT_FILE" -o "$OUTPUT_DIR/heart" \
    --task heartchambers --device gpu

echo ""
echo "結果（heartchambers）:"
ls -la "$OUTPUT_DIR/heart/" 2>/dev/null | grep heart
echo "心臓関連ファイル数: $(ls "$OUTPUT_DIR/heart/"*heart*.nii.gz 2>/dev/null | wc -l)"

# 3. verboseモードでheartchambers_highresを実行
echo ""
echo "========================================="
echo "テスト3: 高精細心臓セグメンテーション (heartchambers_highres) - Verbose"
echo "========================================="

docker run --rm --gpus all --ipc=host \
    -v "$HOME:$HOME" \
    wasserth/totalsegmentator:2.10.0 \
    TotalSegmentator -i "$CT_FILE" -o "$OUTPUT_DIR/heart_highres" \
    --task heartchambers_highres --device gpu --verbose

echo ""
echo "結果（heartchambers_highres）:"
ls -la "$OUTPUT_DIR/heart_highres/" 2>/dev/null | grep -E "(heart|ventricle|atrium|myocardium)"
echo "高精細心臓ファイル数: $(ls "$OUTPUT_DIR/heart_highres/"*.nii.gz 2>/dev/null | wc -l)"

# 4. 結果サマリー
echo ""
echo "========================================="
echo "テスト結果サマリー"
echo "========================================="

echo "1. total: $(ls "$OUTPUT_DIR/total/"*.nii.gz 2>/dev/null | wc -l) ファイル"
echo "2. heartchambers: $(ls "$OUTPUT_DIR/heart/"*.nii.gz 2>/dev/null | wc -l) ファイル"
echo "3. heartchambers_highres: $(ls "$OUTPUT_DIR/heart_highres/"*.nii.gz 2>/dev/null | wc -l) ファイル"

# 成功したタスクを特定
if [ $(ls "$OUTPUT_DIR/total/"*.nii.gz 2>/dev/null | wc -l) -gt 0 ]; then
    echo ""
    echo "✓ totalタスクは正常に動作しています"
fi

if [ $(ls "$OUTPUT_DIR/heart/"*heart*.nii.gz 2>/dev/null | wc -l) -gt 0 ]; then
    echo "✓ heartchambersタスクは正常に動作しています"
fi

if [ $(ls "$OUTPUT_DIR/heart_highres/"*.nii.gz 2>/dev/null | wc -l) -gt 0 ]; then
    echo "✓ heartchambers_highresタスクは正常に動作しています"
else
    echo ""
    echo "! heartchambers_highresが動作していない可能性があります"
    echo "  考えられる原因:"
    echo "  - ライセンスの形式が正しくない"
    echo "  - タスク名が変更された"
    echo "  - 入力データの形式が適合しない"
fi

echo ""
echo "========================================="
echo "デバッグ完了"
echo "========================================="