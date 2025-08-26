#!/bin/bash
# heartchambers_highresタスクのテストスクリプト

echo "========================================="
echo "心臓セグメンテーションテスト"
echo "========================================="

# 入力と出力パス
INPUT_FILE="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/heartchambers_test"
LICENSE="aca_DHDKRFJMHUX0PK"

# 出力ディレクトリをクリーンアップ
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "入力: $INPUT_FILE"
echo "出力: $OUTPUT_DIR"
echo "ライセンス: $LICENSE"
echo ""

# heartchambers_highresタスクを実行
echo "実行中..."
docker run --rm --gpus all --ipc=host \
    -v /home/soya:/home/soya \
    -v ~/.totalsegmentator:/root/.totalsegmentator \
    wasserth/totalsegmentator:2.10.0 \
    TotalSegmentator \
    -i "$INPUT_FILE" \
    -o "$OUTPUT_DIR" \
    --task heartchambers_highres \
    --device gpu \
    -l "$LICENSE"

# 結果確認
echo ""
echo "結果:"
if [ -d "$OUTPUT_DIR" ]; then
    ls -la "$OUTPUT_DIR" | head -20
    echo "生成ファイル数: $(ls "$OUTPUT_DIR" | wc -l)"
else
    echo "出力ディレクトリが作成されませんでした"
fi