#!/bin/bash
# 3症例すべてのEAT+PAT抽出とVisualizationを実行

set -e

echo "=========================================="
echo "EAT+PAT Anatomical Boundary Extraction"
echo "=========================================="

# 症例1 (valid_1_a_1)
echo ""
echo "[1/3] Processing valid_1_a_1..."
echo "----------------------------------------"
python3 /home/soya/ctrate_ws/scripts/extract_eat_pat_integrated.py \
    /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz \
    /home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_1/segmentation \
    /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v1_a_1 \
    --dilation-radius 15.0 \
    --verbose

echo "Visualizing valid_1_a_1..."
python3 /home/soya/ctrate_ws/scripts/visualize_eat_pat_integrated.py \
    /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz \
    /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v1_a_1/masks \
    /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v1_a_1/visualizations \
    --verbose

# 症例2 (valid_1_a_2)
echo ""
echo "[2/3] Processing valid_1_a_2..."
echo "----------------------------------------"
python3 /home/soya/ctrate_ws/scripts/extract_eat_pat_integrated.py \
    /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_2.nii.gz \
    /home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation \
    /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v1_a_2 \
    --dilation-radius 15.0 \
    --verbose

echo "Visualizing valid_1_a_2..."
python3 /home/soya/ctrate_ws/scripts/visualize_eat_pat_integrated.py \
    /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_2.nii.gz \
    /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v1_a_2/masks \
    /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v1_a_2/visualizations \
    --verbose

# 症例3 (valid_2_a_1)
echo ""
echo "[3/3] Processing valid_2_a_1..."
echo "----------------------------------------"
python3 /home/soya/ctrate_ws/scripts/extract_eat_pat_integrated.py \
    /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz \
    /home/soya/ctrate_ws/outputs/batch_results/valid_2_valid_2_a_1/segmentation \
    /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v2_a_1 \
    --dilation-radius 15.0 \
    --verbose

echo "Visualizing valid_2_a_1..."
python3 /home/soya/ctrate_ws/scripts/visualize_eat_pat_integrated.py \
    /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz \
    /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v2_a_1/masks \
    /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v2_a_1/visualizations \
    --verbose

echo ""
echo "=========================================="
echo "All processing completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v1_a_1/"
echo "  - /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v1_a_2/"
echo "  - /home/soya/ctrate_ws/outputs/eat_pat_anatomical_v2_a_1/"
echo ""
echo "Each contains:"
echo "  - masks/: Segmentation masks (heart, shell, eat_pat, visceral_fat)"
echo "  - visualizations/: Visualization images"
echo "  - statistics/: JSON statistics file"