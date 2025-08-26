#!/bin/bash
# heartchambers_highresの権限問題を修正

echo "========================================="
echo "heartchambers_highres修正スクリプト"
echo "========================================="

# 権限を修正
echo "権限を修正中..."
sudo chown -R soya:soya ~/.totalsegmentator/ 2>/dev/null || true

# 必要なディレクトリを作成
mkdir -p ~/.totalsegmentator/nnunet/results

# 再度heartchambers_highresを実行
INPUT_FILE="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/heartchambers_fixed"
LICENSE="aca_DHDKRFJMHUX0PK"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "heartchambers_highresタスクを再実行..."
echo "入力: $INPUT_FILE"
echo "出力: $OUTPUT_DIR"
echo ""

# Dockerコンテナを実行（ユーザー権限を指定）
docker run --rm --gpus all --ipc=host \
    --user $(id -u):$(id -g) \
    -v /home/soya:/home/soya \
    -v ~/.totalsegmentator:/home/soya/.totalsegmentator \
    -e HOME=/home/soya \
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
    ls -la "$OUTPUT_DIR"
    echo ""
    echo "生成ファイル数: $(ls "$OUTPUT_DIR" 2>/dev/null | wc -l)"
    
    # 心臓関連ファイルを確認
    echo ""
    echo "心臓関連ファイル:"
    ls "$OUTPUT_DIR" 2>/dev/null | grep -E "heart|atrium|ventricle|myocardium|aorta" || echo "心臓関連ファイルが見つかりません"
else
    echo "出力ディレクトリが作成されませんでした"
fi