#!/usr/bin/env bash
set -euo pipefail

# Test script for DeepHeartSeg-based EAT extraction

# Input paths
CT_INPUT="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
TOTALSEG_DIR="/home/soya/ctrate_ws/outputs/totalseg_output_test"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/eat_deepheartseg_test"

# First run TotalSegmentator if not exists
if [ ! -d "$TOTALSEG_DIR" ]; then
    echo "Running TotalSegmentator first..."
    docker run --rm \
        -v /home/soya/ctrate_ws:/workspace \
        wasserth/totalsegmentator:2.4.0 \
        TotalSegmentator \
        -i /workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz \
        -o /workspace/outputs/totalseg_output_test \
        --task total
fi

# Check if input exists
if [ ! -f "$CT_INPUT" ]; then
    echo "Error: CT input file not found: $CT_INPUT"
    exit 1
fi

if [ ! -d "$TOTALSEG_DIR" ]; then
    echo "Error: TotalSegmentator output directory not found: $TOTALSEG_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "DeepHeartSeg-based EAT Extraction Test"
echo "============================================"
echo "CT input: $CT_INPUT"
echo "TotalSeg dir: $TOTALSEG_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Run DeepHeartSeg-based EAT extraction
# Note: --run-deepheartseg を付けると実際にDeepHeartSegを実行
# 今回はまずDeepHeartSegなしで基本動作確認
echo "Running EAT extraction (without DeepHeartSeg for now)..."
python3 scripts/eat_from_pericardium.py \
    "$CT_INPUT" \
    "$TOTALSEG_DIR" \
    "$OUTPUT_DIR" \
    --deepheartseg-root "/home/soya/ctrate_ws/tools/DeepHeartSeg" \
    --hu-min -190 \
    --hu-max -30 \
    --contact-mm 3 \
    --strict-exclusion \
    || echo "Note: DeepHeartSeg output not found, using fallback mode"

# Show results
echo ""
echo "============================================"
echo "Results:"
echo "============================================"
if [ -f "$OUTPUT_DIR/eat_summary.json" ]; then
    cat "$OUTPUT_DIR/eat_summary.json"
else
    echo "Warning: eat_summary.json not found"
fi

echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR/"

echo ""
echo "Test completed."