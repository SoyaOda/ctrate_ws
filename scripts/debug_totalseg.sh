#!/bin/bash
# TotalSegmentatorのデバッグスクリプト

echo "=== TotalSegmentator Debug Script ==="
echo "Time: $(date)"
echo ""

# 入力ファイルの確認
INPUT_FILE="/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT_DIR="/mnt/e/oda/Chest-Xray-Create-Dataset/data/CT-RATE-v2/test_debug"

echo "1. Checking input file..."
if [ -f "$INPUT_FILE" ]; then
    echo "   ✓ Input file exists"
    echo "   File size: $(ls -lh $INPUT_FILE | awk '{print $5}')"
else
    echo "   ✗ Input file not found!"
    exit 1
fi

echo ""
echo "2. Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader

echo ""
echo "3. Creating output directory..."
mkdir -p $OUTPUT_DIR
echo "   Output dir: $OUTPUT_DIR"

echo ""
echo "4. Running TotalSegmentator with verbose output..."
echo "   Command: docker run --rm --gpus all --ipc=host -v $(dirname $INPUT_FILE):/input -v $OUTPUT_DIR:/output wasserth/totalsegmentator:2.10.0 TotalSegmentator -i /input/$(basename $INPUT_FILE) -o /output --task total --ml --verbose"
echo ""
echo "=== Starting TotalSegmentator ==="

# タイムアウト付きで実行（5分）
timeout 300 docker run --rm \
    --gpus all \
    --ipc=host \
    -v $(dirname $INPUT_FILE):/input \
    -v $OUTPUT_DIR:/output \
    wasserth/totalsegmentator:2.10.0 \
    TotalSegmentator \
    -i /input/$(basename $INPUT_FILE) \
    -o /output \
    --task total \
    --ml \
    --verbose

EXIT_CODE=$?

echo ""
echo "=== TotalSegmentator finished ==="
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 124 ]; then
    echo "✗ Process timed out after 5 minutes"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Process completed successfully"
    echo ""
    echo "5. Checking output files..."
    ls -la $OUTPUT_DIR | head -10
else
    echo "✗ Process failed with exit code: $EXIT_CODE"
fi

echo ""
echo "=== End of debug script ==="