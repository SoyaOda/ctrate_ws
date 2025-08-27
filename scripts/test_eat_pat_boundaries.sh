#!/bin/bash
# Test script for improved EAT+PAT extraction with anatomical boundaries

set -e

echo "=================================="
echo "Testing Improved EAT+PAT Extraction"
echo "=================================="

# Test data paths
INPUT_CT="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
TOTALSEG_DIR="/home/soya/ctrate_ws/outputs/batch_results/valid_2_valid_2_a_1/segmentation"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/eat_pat_anatomical_test"

# Check if input files exist
if [ ! -f "$INPUT_CT" ]; then
    echo "Error: Input CT file not found: $INPUT_CT"
    exit 1
fi

if [ ! -d "$TOTALSEG_DIR" ]; then
    echo "Error: TotalSegmentator output directory not found: $TOTALSEG_DIR"
    exit 1
fi

# Clean and create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Input CT: $INPUT_CT"
echo "TotalSeg: $TOTALSEG_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Run improved extraction
echo "Running EAT+PAT extraction with anatomical boundaries..."
python3 /home/soya/ctrate_ws/scripts/extract_eat_pat_integrated.py \
    "$INPUT_CT" \
    "$TOTALSEG_DIR" \
    "$OUTPUT_DIR" \
    --dilation-radius 15.0 \
    --verbose

# Check results
echo ""
echo "Checking output files..."
if [ -f "$OUTPUT_DIR/masks/eat_pat.nii.gz" ]; then
    echo "✓ EAT+PAT mask created"
else
    echo "✗ EAT+PAT mask not found"
fi

if [ -f "$OUTPUT_DIR/masks/shell.nii.gz" ]; then
    echo "✓ Shell mask created"
else
    echo "✗ Shell mask not found"
fi

if [ -f "$OUTPUT_DIR/statistics/eat_pat_analysis.json" ]; then
    echo "✓ Statistics file created"
    echo ""
    echo "Statistics:"
    python3 -c "
import json
with open('$OUTPUT_DIR/statistics/eat_pat_analysis.json', 'r') as f:
    stats = json.load(f)
    print(f'  EAT+PAT volume: {stats[\"volumes_ml\"][\"eat_pat\"]} ml')
    print(f'  Shell volume: {stats[\"volumes_ml\"][\"shell\"]} ml')
    print(f'  Fat fraction in shell: {stats[\"ratios_percent\"][\"fat_fraction_in_shell\"]}%')
    if 'parameters' in stats:
        params = stats['parameters']
        if 'heart_z_range' in params:
            z_range = params['heart_z_range']
            print(f'  Z-range: {z_range[0]} to {z_range[1]} (total: {z_range[1]-z_range[0]+1} slices)')
"
else
    echo "✗ Statistics file not found"
fi

echo ""
echo "Test completed!"