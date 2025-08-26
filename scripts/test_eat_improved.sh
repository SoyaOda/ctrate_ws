#!/bin/bash
# 改善版EAT抽出のテスト実行

# 入力ファイル
CT_NIFTI="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
TOTALSEG_DIR="/home/soya/ctrate_ws/outputs/totalseg_output/heart_chambers_output"
HARTA_DIR="/home/soya/ctrate_ws/outputs/test_harta_fixed/harta_output/test_valid_2_a_1_fixed"
SLICE_MAP="/home/soya/ctrate_ws/outputs/test_harta_fixed/slice_map.json"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/eat_improved_test"

echo "=================================================="
echo "IMPROVED EAT EXTRACTION TEST"
echo "=================================================="
echo "[INFO] CT Input: $CT_NIFTI"
echo "[INFO] TotalSeg: $TOTALSEG_DIR"
echo "[INFO] HARTA: $HARTA_DIR"
echo "[INFO] Output: $OUTPUT_DIR"
echo ""

# まずTotalSegmentatorが大血管も出力しているか確認
echo "[INFO] Checking for vessel segmentations..."
if [ -f "$TOTALSEG_DIR/aorta.nii.gz" ]; then
    echo "  ✓ aorta.nii.gz found"
else
    echo "  ✗ aorta.nii.gz NOT found - will run TotalSegmentator with full task"
fi

if [ -f "$TOTALSEG_DIR/pulmonary_artery.nii.gz" ]; then
    echo "  ✓ pulmonary_artery.nii.gz found"
else
    echo "  ✗ pulmonary_artery.nii.gz NOT found - will run TotalSegmentator with full task"
fi

# 大血管が無い場合は再実行
if [ ! -f "$TOTALSEG_DIR/aorta.nii.gz" ] || [ ! -f "$TOTALSEG_DIR/pulmonary_artery.nii.gz" ]; then
    echo ""
    echo "[INFO] Running TotalSegmentator to get vessels..."
    
    # .envからライセンスキー読み込み
    if [ -f "/home/soya/ctrate_ws/.env" ]; then
        source /home/soya/ctrate_ws/.env
    fi
    
    docker run --rm \
        -v /home/soya:/home/soya \
        wasserth/totalsegmentator:2.10.0 \
        TotalSegmentator \
        -i "$CT_NIFTI" \
        -o "$TOTALSEG_DIR" \
        -ta total \
        -l "$TOTALSEG_LICENSE_KEY" \
        --fast
    
    echo "[INFO] TotalSegmentator completed"
fi

# 改善版スクリプト実行
echo ""
echo "[INFO] Running improved EAT extraction..."
docker run --rm \
    -v /home/soya:/home/soya \
    wasserth/totalsegmentator:2.10.0 \
    python3 /home/soya/ctrate_ws/scripts/extract_eat_with_totalseg_improved.py \
    "$CT_NIFTI" \
    "$TOTALSEG_DIR" \
    "$OUTPUT_DIR" \
    --harta-png-dir "$HARTA_DIR" \
    --slice-map "$SLICE_MAP" \
    --hu-min -190 \
    --hu-max -30 \
    --shell-max 15.0 \
    --shell-seed 6.0 \
    --harta-min-overlap 0.15

# 結果確認
if [ -f "$OUTPUT_DIR/eat_summary.json" ]; then
    echo ""
    echo "[SUCCESS] EAT extraction completed!"
    echo "Results:"
    cat "$OUTPUT_DIR/eat_summary.json"
else
    echo "[ERROR] Failed to extract EAT"
    exit 1
fi