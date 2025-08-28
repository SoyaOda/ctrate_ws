#!/bin/bash
# 現状版と改善版のEAT+PAT抽出を比較（特にvalid_1_a_2のslice_141付近）

set -e

echo "==========================================="
echo "EAT+PAT Extraction Method Comparison"
echo "Focus: valid_1_a_2, slice 141±1"
echo "==========================================="

# 対象症例（横隔膜付近の問題が顕著なケース）
CT_PATH="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_2.nii.gz"
TOTALSEG_DIR="/home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation"

# 出力ディレクトリ
CURRENT_DIR="/home/soya/ctrate_ws/outputs/eat_pat_current_v1_a_2"
IMPROVED_DIR="/home/soya/ctrate_ws/outputs/eat_pat_improved_v1_a_2"

echo ""
echo "Target CT: valid_1_a_2"
echo "TotalSeg: $TOTALSEG_DIR"
echo ""

# 1. 現状版の実行
echo "[1/4] Running CURRENT version..."
echo "----------------------------------------"
python3 /home/soya/ctrate_ws/scripts/extract_eat_pat_integrated.py \
    "$CT_PATH" \
    "$TOTALSEG_DIR" \
    "$CURRENT_DIR" \
    --dilation-radius 15.0 \
    --verbose

# 2. 改善版の実行
echo ""
echo "[2/4] Running IMPROVED version..."
echo "----------------------------------------"
python3 /home/soya/ctrate_ws/scripts/extract_eat_pat_improved.py \
    "$CT_PATH" \
    "$TOTALSEG_DIR" \
    "$IMPROVED_DIR" \
    --dilation-radius 15.0 \
    --verbose

# 3. 両方の可視化実行
echo ""
echo "[3/4] Visualizing results..."
echo "----------------------------------------"

echo "Visualizing CURRENT version..."
python3 /home/soya/ctrate_ws/scripts/visualize_eat_pat_integrated.py \
    "$CT_PATH" \
    "$CURRENT_DIR/masks" \
    "$CURRENT_DIR/visualizations" \
    --verbose

echo ""
echo "Visualizing IMPROVED version..."
python3 /home/soya/ctrate_ws/scripts/visualize_eat_pat_integrated.py \
    "$CT_PATH" \
    "$IMPROVED_DIR/masks" \
    "$IMPROVED_DIR/visualizations" \
    --verbose

# 4. 統計比較
echo ""
echo "[4/4] Statistical Comparison"
echo "==========================================="

echo ""
echo "CURRENT VERSION:"
echo "----------------"
python3 -c "
import json
with open('$CURRENT_DIR/statistics/eat_pat_analysis.json', 'r') as f:
    stats = json.load(f)
    print(f'  EAT+PAT volume: {stats[\"volumes_ml\"][\"eat_pat\"]} ml')
    print(f'  Shell volume: {stats[\"volumes_ml\"][\"shell\"]} ml')
    print(f'  Fat fraction: {stats[\"ratios_percent\"][\"fat_fraction_in_shell\"]}%')
    if 'parameters' in stats and 'heart_z_range' in stats['parameters']:
        z_range = stats['parameters']['heart_z_range']
        print(f'  Z-range: {z_range[0]} to {z_range[1]} ({z_range[1]-z_range[0]+1} slices)')
"

echo ""
echo "IMPROVED VERSION:"
echo "-----------------"
python3 -c "
import json
with open('$IMPROVED_DIR/statistics/eat_pat_analysis.json', 'r') as f:
    stats = json.load(f)
    print(f'  EAT+PAT volume: {stats[\"volumes_ml\"][\"eat_pat\"]} ml')
    print(f'  Shell volume: {stats[\"volumes_ml\"][\"shell\"]} ml')
    print(f'  Fat fraction: {stats[\"ratios_percent\"][\"fat_fraction_in_shell\"]}%')
    if 'parameters' in stats and 'heart_z_range' in stats['parameters']:
        z_range = stats['parameters']['heart_z_range']
        print(f'  Z-range: {z_range[0]} to {z_range[1]} ({z_range[1]-z_range[0]+1} slices)')
    if 'improvements' in stats['parameters']:
        print(f'  Improvements: {len(stats[\"parameters\"][\"improvements\"])} methods applied')
"

echo ""
echo "VOLUME CHANGES:"
echo "----------------"
python3 -c "
import json
with open('$CURRENT_DIR/statistics/eat_pat_analysis.json', 'r') as f:
    current = json.load(f)
with open('$IMPROVED_DIR/statistics/eat_pat_analysis.json', 'r') as f:
    improved = json.load(f)

eat_change = improved['volumes_ml']['eat_pat'] - current['volumes_ml']['eat_pat']
eat_pct = (eat_change / current['volumes_ml']['eat_pat'] * 100) if current['volumes_ml']['eat_pat'] > 0 else 0

shell_change = improved['volumes_ml']['shell'] - current['volumes_ml']['shell']
shell_pct = (shell_change / current['volumes_ml']['shell'] * 100) if current['volumes_ml']['shell'] > 0 else 0

print(f'  EAT+PAT: {eat_change:+.2f} ml ({eat_pct:+.1f}%)')
print(f'  Shell: {shell_change:+.2f} ml ({shell_pct:+.1f}%)')

if 'parameters' in current and 'heart_z_range' in current['parameters']:
    curr_z = current['parameters']['heart_z_range']
    impr_z = improved['parameters']['heart_z_range']
    print(f'  Z-range change: [{curr_z[0]}-{curr_z[1]}] → [{impr_z[0]}-{impr_z[1]}]')
    print(f'    Bottom shift: {impr_z[0] - curr_z[0]:+d} slices')
    print(f'    Top shift: {impr_z[1] - curr_z[1]:+d} slices')
"

echo ""
echo "==========================================="
echo "Comparison completed!"
echo ""
echo "Key slice visualizations to examine:"
echo "  Current:  $CURRENT_DIR/visualizations/slices/slice_141.png"
echo "  Improved: $IMPROVED_DIR/visualizations/slices/slice_141.png"
echo ""
echo "Also check adjacent slices:"
echo "  - slice_140.png (one slice above)"
echo "  - slice_142.png (one slice below)"
echo ""
echo "Full visualization directories:"
echo "  - $CURRENT_DIR/visualizations/"
echo "  - $IMPROVED_DIR/visualizations/"
echo "==========================================="