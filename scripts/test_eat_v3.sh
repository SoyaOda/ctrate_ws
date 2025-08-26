#!/bin/bash
# 改善版v3 EAT抽出のテスト実行（過拡張対策版）

CT_NIFTI="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
TOTALSEG_DIR="/home/soya/ctrate_ws/outputs/totalseg_output/heart_chambers_output"
HARTA_DIR="/home/soya/ctrate_ws/outputs/test_harta_fixed/harta_output/test_valid_2_a_1_fixed"
SLICE_MAP="/home/soya/ctrate_ws/outputs/test_harta_fixed/slice_map.json"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/eat_v3_test"

echo "=================================================="
echo "IMPROVED EAT EXTRACTION v3 (Domain-Limited)"
echo "=================================================="
echo "[INFO] CT Input: $CT_NIFTI"
echo "[INFO] TotalSeg: $TOTALSEG_DIR"
echo "[INFO] HARTA: $HARTA_DIR"
echo "[INFO] Output: $OUTPUT_DIR"
echo ""

# 改善版v3実行
echo "[INFO] Running improved EAT extraction v3..."
docker run --rm \
    -v /home/soya:/home/soya \
    wasserth/totalsegmentator:2.10.0 \
    python3 /home/soya/ctrate_ws/scripts/extract_eat_with_totalseg_v3.py \
    "$CT_NIFTI" \
    "$TOTALSEG_DIR" \
    "$OUTPUT_DIR" \
    --harta-png-dir "$HARTA_DIR" \
    --slice-map "$SLICE_MAP" \
    --hu-min -190 \
    --hu-max -30 \
    --shell-max 12.0 \
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