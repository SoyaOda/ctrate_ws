#!/bin/bash

# TotalSegmentatorのtissue_typesタスクを実行するスクリプト
# 内臓脂肪（torso_fat）を含む組織タイプのセグメンテーション

set -e

# ライセンスキー（tissue_typesタスクに必要）
LICENSE_KEY="aca_DHDKRFJMHUX0PK"

# 入力と出力のパス
INPUT_CT="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/batch_results/valid_2_valid_2_a_1/tissue_types"

echo "================================================"
echo "Running TotalSegmentator with tissue_types task"
echo "================================================"
echo ""
echo "Input CT: $INPUT_CT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# 出力ディレクトリを作成
mkdir -p "$OUTPUT_DIR"

# TotalSegmentatorを実行（tissue_typesタスク）
echo "Starting TotalSegmentator tissue_types task..."
echo "This may take several minutes..."
echo ""

# Dockerを使用してTotalSegmentatorを実行
# 注意: --ml オプションは使用しない（個別ファイル出力のため）
# ライセンスキーを渡す
docker run --rm \
    --gpus all \
    -v "$INPUT_CT:/input.nii.gz:ro" \
    -v "$OUTPUT_DIR:/output" \
    wasserth/totalsegmentator:2.10.0 \
    TotalSegmentator -i /input.nii.gz -o /output --task tissue_types -l "$LICENSE_KEY"

echo ""
echo "================================================"
echo "Checking output files..."
echo "================================================"
echo ""

# 期待されるファイルをチェック
if [ -f "$OUTPUT_DIR/torso_fat.nii.gz" ]; then
    echo "✓ torso_fat.nii.gz found (visceral fat)"
else
    echo "✗ torso_fat.nii.gz NOT found"
fi

if [ -f "$OUTPUT_DIR/subcutaneous_fat.nii.gz" ]; then
    echo "✓ subcutaneous_fat.nii.gz found"
else
    echo "✗ subcutaneous_fat.nii.gz NOT found"
fi

if [ -f "$OUTPUT_DIR/skeletal_muscle.nii.gz" ]; then
    echo "✓ skeletal_muscle.nii.gz found"
else
    echo "✗ skeletal_muscle.nii.gz NOT found"
fi

echo ""
echo "All files in output directory:"
ls -lh "$OUTPUT_DIR"

echo ""
echo "================================================"
echo "Tissue types segmentation completed!"
echo "================================================"