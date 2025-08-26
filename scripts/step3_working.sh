#!/bin/bash
# STEP3: TotalSegmentator実行スクリプト（動作確認版）

echo "========================================="
echo "STEP 3: TotalSegmentator実行"
echo "========================================="

# 設定
DOCKER_IMAGE="wasserth/totalsegmentator:2.10.0"
CT_FILE="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/step3_output"

echo "入力: $CT_FILE"
echo "出力: $OUTPUT_DIR"

# 出力ディレクトリ準備
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 基本セグメンテーション実行
echo ""
echo "基本セグメンテーションを実行中..."
docker run --rm --gpus all --ipc=host \
    -v /home/soya:/home/soya \
    "$DOCKER_IMAGE" \
    TotalSegmentator \
    -i "$CT_FILE" \
    -o "$OUTPUT_DIR" \
    --task total \
    --device gpu \
    --fast

# 結果確認
echo ""
echo "結果確認:"
if [ -d "$OUTPUT_DIR/total" ]; then
    echo "✓ セグメンテーション完了（totalサブディレクトリ内）"
    echo "生成ファイル数: $(ls "$OUTPUT_DIR/total" | wc -l)"
    ls "$OUTPUT_DIR/total" | head -10
elif [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
    echo "✓ セグメンテーション完了"
    echo "生成ファイル数: $(ls "$OUTPUT_DIR" | wc -l)"
    ls "$OUTPUT_DIR" | head -10
else
    echo "✗ セグメンテーション失敗"
fi

echo ""
echo "========================================="
echo "STEP 3 完了"
echo "========================================="
