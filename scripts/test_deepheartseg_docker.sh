#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "DeepHeartSeg-based EAT Test (Docker)"
echo "============================================"

# Input paths
CT_INPUT="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
OUTPUT_BASE="/home/soya/ctrate_ws/outputs/deepheartseg_test"

# Create output directories
mkdir -p "$OUTPUT_BASE/totalseg"
mkdir -p "$OUTPUT_BASE/eat_result"

echo "Step 1: Running TotalSegmentator..."
docker run --rm \
    -v /home/soya/ctrate_ws:/workspace \
    wasserth/totalsegmentator:2.10.0 \
    TotalSegmentator \
    -i /workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz \
    -o /workspace/outputs/deepheartseg_test/totalseg \
    --task total \
    --fast

echo ""
echo "Step 2: Running DeepHeartSeg-based EAT extraction..."

# Run using docker with Python environment
docker run --rm \
    -v /home/soya/ctrate_ws:/workspace \
    wasserth/totalsegmentator:2.10.0 \
    python3 /workspace/scripts/eat_from_pericardium.py \
    /workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz \
    /workspace/outputs/deepheartseg_test/totalseg \
    /workspace/outputs/deepheartseg_test/eat_result \
    --hu-min -190 \
    --hu-max -30 \
    --contact-mm 3 \
    --strict-exclusion

echo ""
echo "Results:"
if [ -f "$OUTPUT_BASE/eat_result/eat_summary.json" ]; then
    cat "$OUTPUT_BASE/eat_result/eat_summary.json"
fi

echo ""
echo "Output files:"
ls -lh "$OUTPUT_BASE/eat_result/"

echo "Test completed."