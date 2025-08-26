#!/bin/bash
# EAT抽出パイプラインのテストスクリプト

echo "========================================="
echo "EAT抽出テスト (HARTA + TotalSegmentator)"
echo "========================================="

# ベースディレクトリ
BASE_DIR="/home/soya/ctrate_ws"
DATA_DIR="$BASE_DIR/data/CT-RATE-v2/dataset/valid_fixed"
OUTPUT_BASE="$BASE_DIR/outputs/eat_extraction_test"

# テストケース
TEST_CASE="valid_2_a_1"
CT_FILE="$DATA_DIR/valid_2/valid_2_a/${TEST_CASE}.nii.gz"
HARTA_OUTPUT="$BASE_DIR/outputs/test_harta_fixed/harta_output/test_valid_2_a_1_fixed"
SLICE_MAP="$BASE_DIR/outputs/test_harta_fixed/slice_map.json"

# 出力ディレクトリ
OUTPUT_DIR="$OUTPUT_BASE/${TEST_CASE}"
mkdir -p "$OUTPUT_DIR"

# ファイル存在確認
echo "入力ファイル確認..."
if [ ! -f "$CT_FILE" ]; then
    echo "[ERROR] CT file not found: $CT_FILE"
    exit 1
fi

if [ ! -d "$HARTA_OUTPUT" ]; then
    echo "[ERROR] HARTA output not found: $HARTA_OUTPUT"
    exit 1
fi

if [ ! -f "$SLICE_MAP" ]; then
    echo "[ERROR] Slice map not found: $SLICE_MAP"
    exit 1
fi

echo "✓ 入力ファイル確認完了"
echo ""

# Step 1: TotalSegmentatorで心臓構造を取得
echo "========================================="
echo "Step 1: TotalSegmentator実行"
echo "========================================="

TOTALSEG_OUTPUT="$OUTPUT_DIR/totalseg"

# TotalSegmentatorがすでに実行済みか確認
if [ -f "$TOTALSEG_OUTPUT/heart_myocardium.nii.gz" ]; then
    echo "TotalSegmentator結果が既に存在します。スキップします。"
    SKIP_TOTALSEG="--skip_totalseg"
else
    echo "TotalSegmentatorを実行中..."
    SKIP_TOTALSEG=""
fi

# Step 2: EAT抽出スクリプト実行
echo ""
echo "========================================="
echo "Step 2: EAT抽出"
echo "========================================="

# 複数のシェル厚さでテスト
SHELL_THICKNESSES="5.0 6.0 8.0 10.0"

for THICKNESS in $SHELL_THICKNESSES; do
    echo ""
    echo "シェル厚さ: ${THICKNESS}mm"
    echo "----------------------------------------"
    
    OUTPUT_SUBDIR="$OUTPUT_DIR/shell_${THICKNESS}mm"
    mkdir -p "$OUTPUT_SUBDIR"
    
    python3 "$BASE_DIR/scripts/extract_eat_with_totalseg.py" \
        --ct_nifti "$CT_FILE" \
        --harta_png_dir "$HARTA_OUTPUT" \
        --slice_map_json "$SLICE_MAP" \
        --output_dir "$OUTPUT_SUBDIR" \
        --shell_thickness "$THICKNESS" \
        --hu_min -190 \
        --hu_max -30 \
        $SKIP_TOTALSEG
    
    # 結果を読み取る
    if [ -f "$OUTPUT_SUBDIR/eat_summary.json" ]; then
        VOLUME=$(python3 -c "import json; print(json.load(open('$OUTPUT_SUBDIR/eat_summary.json'))['eat_volume_ml'])")
        echo "  → EATボリューム: ${VOLUME} ml"
        
        # TotalSegmentatorの結果をコピー（次回のために）
        if [ ! -d "$TOTALSEG_OUTPUT" ] && [ -d "$OUTPUT_SUBDIR/totalseg" ]; then
            cp -r "$OUTPUT_SUBDIR/totalseg" "$TOTALSEG_OUTPUT"
        fi
    else
        echo "  ✗ 処理失敗"
    fi
done

# Step 3: 結果サマリー
echo ""
echo "========================================="
echo "結果サマリー"
echo "========================================="

echo "シェル厚さ別のEATボリューム:"
for THICKNESS in $SHELL_THICKNESSES; do
    OUTPUT_SUBDIR="$OUTPUT_DIR/shell_${THICKNESS}mm"
    if [ -f "$OUTPUT_SUBDIR/eat_summary.json" ]; then
        VOLUME=$(python3 -c "import json; print(json.load(open('$OUTPUT_SUBDIR/eat_summary.json'))['eat_volume_ml'])")
        VOXELS=$(python3 -c "import json; print(json.load(open('$OUTPUT_SUBDIR/eat_summary.json'))['eat_voxels'])")
        echo "  ${THICKNESS}mm: ${VOLUME} ml (${VOXELS} voxels)"
    fi
done

echo ""
echo "========================================="
echo "可視化用コマンド"
echo "========================================="
echo "元のCT画像:"
echo "  itksnap $CT_FILE"
echo ""
echo "EATマスク (6mm shell):"
echo "  itksnap $CT_FILE -s $OUTPUT_DIR/shell_6.0mm/eat_final.nii.gz"
echo ""
echo "HARTAとの比較:"
echo "  元のHARTA出力: $HARTA_OUTPUT/fat/"
echo "  改良後EAT: $OUTPUT_DIR/shell_6.0mm/eat_final.nii.gz"

echo ""
echo "処理完了: $OUTPUT_DIR"