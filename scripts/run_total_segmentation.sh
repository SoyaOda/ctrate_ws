#!/bin/bash
# TotalSegmentatorでtotalタスクを実行（肺・腹部臓器を含む全臓器セグメンテーション）

CT_PATH="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_2.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation_total"

echo "Running TotalSegmentator with total task..."
echo "This will generate lung, liver, stomach, spleen and other organ masks"
echo "Input: $CT_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Docker実行（totalタスク）
docker run --rm \
  -v /home/soya/ctrate_ws:/workspace \
  wasserth/totalsegmentator:2.10.0 \
  TotalSegmentator \
  -i /workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_2.nii.gz \
  -o /workspace/outputs/batch_results/valid_1_valid_1_a_2/segmentation_total \
  --task total

echo ""
echo "Checking output files..."
ls -la $OUTPUT_DIR/ | grep -E "lung|liver|stomach|spleen|inferior_vena_cava|pulmonary_vein"

