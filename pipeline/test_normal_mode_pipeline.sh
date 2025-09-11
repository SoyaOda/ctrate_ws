#!/bin/bash

# 通常モード（--fastなし、--roi_subsetなし）でのパイプラインテスト
# 8月31日の実行と同等の設定で再現性を確認

set -e

echo "================================================"
echo "🔬 通常モードパイプライン 再現性テスト"
echo "================================================"
echo "📅 実行日時: $(date)"
echo ""

# 設定
CT_FILE="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/pipeline/normal_mode_test"
PROJECT_ROOT="/home/soya/ctrate_ws"

echo "📁 処理対象CT: $(basename $CT_FILE)"
echo "📂 出力ディレクトリ: $OUTPUT_DIR"
echo ""

# 既存結果のクリーンアップ
if [ -d "$OUTPUT_DIR" ]; then
    echo "🧹 既存結果をクリーンアップ中..."
    echo "odaSOYA6480" | sudo -S rm -rf "$OUTPUT_DIR" 2>/dev/null
fi
mkdir -p "$OUTPUT_DIR"

# ログファイル
LOG_FILE="$OUTPUT_DIR/pipeline_log_$(date +%Y%m%d_%H%M%S).txt"

echo "========================================" | tee -a "$LOG_FILE"
echo "ステップ1: TotalSegmentator（通常モード）" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

STEP1_START=$(date +%s)

# 出力ディレクトリ
TOTALSEG_DIR="$OUTPUT_DIR/totalseg_normal"
mkdir -p "$TOTALSEG_DIR"

# TotalSegmentator実行（通常モード: --fastなし、--roi_subsetなし）
echo "🎯 TotalSegmentator実行（通常モード、全臓器）..." | tee -a "$LOG_FILE"

docker exec totalseg-daemon \
    TotalSegmentator \
    -i "/workspace/${CT_FILE#$PROJECT_ROOT/}" \
    -o "/workspace/${TOTALSEG_DIR#$PROJECT_ROOT/}" \
    --task total 2>&1 | tee -a "$LOG_FILE"

# tissue_typesタスク（torso_fat取得）
echo "🔬 tissue_typesタスク実行..." | tee -a "$LOG_FILE"

TISSUE_DIR="$TOTALSEG_DIR/tissue"
mkdir -p "$TISSUE_DIR"

docker exec totalseg-daemon \
    TotalSegmentator \
    -i "/workspace/${CT_FILE#$PROJECT_ROOT/}" \
    -o "/workspace/${TISSUE_DIR#$PROJECT_ROOT/}" \
    --task tissue_types \
    -l aca_DHDKRFJMHUX0PK 2>&1 | tee -a "$LOG_FILE"

# torso_fatコピー
if [ -f "$TISSUE_DIR/torso_fat.nii.gz" ]; then
    cp "$TISSUE_DIR/torso_fat.nii.gz" "$TOTALSEG_DIR/"
    echo "✅ torso_fatマスクコピー完了" | tee -a "$LOG_FILE"
fi

# 肺マスク統合（EAT/PAT抽出スクリプト用）
echo "🫁 肺マスク統合中..." | tee -a "$LOG_FILE"

python3 - <<EOF
import nibabel as nib
import numpy as np
from pathlib import Path

totalseg_dir = Path("$TOTALSEG_DIR")

# 左肺統合
left_lobes = ['lung_upper_lobe_left.nii.gz', 'lung_lower_lobe_left.nii.gz']
merged_left = None
affine = None

for lobe in left_lobes:
    lobe_path = totalseg_dir / lobe
    if lobe_path.exists():
        img = nib.load(str(lobe_path))
        data = img.get_fdata() > 0
        if merged_left is None:
            merged_left = data
            affine = img.affine
        else:
            merged_left = merged_left | data

if merged_left is not None:
    nib.save(nib.Nifti1Image(merged_left.astype(np.uint8), affine),
            str(totalseg_dir / "lung_left.nii.gz"))
    print("  ✅ lung_left.nii.gz生成")

# 右肺統合
right_lobes = ['lung_upper_lobe_right.nii.gz', 'lung_middle_lobe_right.nii.gz', 'lung_lower_lobe_right.nii.gz']
merged_right = None

for lobe in right_lobes:
    lobe_path = totalseg_dir / lobe
    if lobe_path.exists():
        img = nib.load(str(lobe_path))
        data = img.get_fdata() > 0
        if merged_right is None:
            merged_right = data
            affine = img.affine
        else:
            merged_right = merged_right | data

if merged_right is not None:
    nib.save(nib.Nifti1Image(merged_right.astype(np.uint8), affine),
            str(totalseg_dir / "lung_right.nii.gz"))
    print("  ✅ lung_right.nii.gz生成")
EOF

STEP1_END=$(date +%s)
STEP1_TIME=$((STEP1_END - STEP1_START))

echo "⏱️ ステップ1完了: ${STEP1_TIME}秒" | tee -a "$LOG_FILE"

# 生成マスク数確認
MASK_COUNT=$(ls -1 "$TOTALSEG_DIR"/*.nii.gz 2>/dev/null | wc -l)
echo "📊 生成マスク数: $MASK_COUNT" | tee -a "$LOG_FILE"

# 心臓マスクのサイズ確認
if [ -f "$TOTALSEG_DIR/heart.nii.gz" ]; then
    python3 - <<EOF
import nibabel as nib
import numpy as np

heart = nib.load("$TOTALSEG_DIR/heart.nii.gz")
heart_data = heart.get_fdata()
voxel_count = np.sum(heart_data > 0)
voxel_size = np.prod(heart.header.get_zooms())
volume_ml = voxel_count * voxel_size / 1000

print(f"🫀 心臓統計:")
print(f"  - Voxel数: {int(voxel_count):,}")
print(f"  - 体積: {volume_ml:.2f} ml")
EOF
fi | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "ステップ2: EAT/PAT抽出" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

STEP2_START=$(date +%s)

# EAT/PAT抽出
EAT_PAT_DIR="$OUTPUT_DIR/eat_pat_output"

echo "🫀 EAT/PAT抽出実行中..." | tee -a "$LOG_FILE"

timeout 600 docker run --rm --gpus all \
    --user $(id -u):$(id -g) \
    -v "$PROJECT_ROOT:/workspace" \
    wasserth/totalsegmentator:2.10.0 \
    python3 /workspace/scripts/extract_eat_pat_improved_v5_4.py \
    "/workspace/${CT_FILE#$PROJECT_ROOT/}" \
    "/workspace/${TOTALSEG_DIR#$PROJECT_ROOT/}" \
    "/workspace/${EAT_PAT_DIR#$PROJECT_ROOT/}" 2>&1 | tee -a "$LOG_FILE"

STEP2_END=$(date +%s)
STEP2_TIME=$((STEP2_END - STEP2_START))

echo "⏱️ ステップ2完了: ${STEP2_TIME}秒" | tee -a "$LOG_FILE"

# EAT/PAT統計確認
if [ -f "$EAT_PAT_DIR/statistics/eat_pat_analysis.json" ]; then
    echo "📊 EAT/PAT統計:" | tee -a "$LOG_FILE"
    python3 - <<EOF
import json

with open("$EAT_PAT_DIR/statistics/eat_pat_analysis.json") as f:
    data = json.load(f)
    
print(f"  - 心臓体積: {data['volumes_ml']['heart']:.2f} ml")
print(f"  - EAT+PAT体積: {data['volumes_ml']['eat_pat']:.2f} ml")
print(f"  - Fat fraction: {data['ratios_percent']['fat_fraction_in_shell']:.2f}%")
print(f"  - ILAM設定:")
print(f"    - lung_contact_mm: {data['parameters']['ilam_settings']['lung_contact_mm']}")
print(f"    - contact_eps_mm: {data['parameters']['ilam_settings']['contact_eps_mm']}")
print(f"    - abdomen_gap_mm: {data['parameters']['ilam_settings']['abdomen_gap_mm']}")
print(f"    - max_heart_dist_mm: {data['parameters']['ilam_settings']['max_heart_dist_mm']}")
EOF
fi | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "ステップ3: DeepDRR生成" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

STEP3_START=$(date +%s)

# DeepDRR出力ディレクトリ
DEEPDRR_DIR="$OUTPUT_DIR/deepdrr_output"
mkdir -p "$DEEPDRR_DIR"

echo "📸 DeepDRR生成実行中..." | tee -a "$LOG_FILE"

timeout 300 docker run --rm --gpus all \
    -v "$PROJECT_ROOT:/workspace" \
    deepdrr-japan-mirror:latest bash -c "
    pip3 install torch torchvision pydicom opencv-python scipy --no-cache-dir > /dev/null 2>&1 &&
    cd /workspace &&
    python3 - <<'PYEOF'
import sys
sys.path.append('/workspace/deepdrr')
exec(open('/workspace/deepdrr/test_deepdrr_final_pa_fixed.py').read().replace(
    'outdir = Path(\"/workspace/deepdrr/outputs\")',
    'outdir = Path(\"$DEEPDRR_DIR\")'
))
PYEOF
    " 2>&1 | tee -a "$LOG_FILE"

STEP3_END=$(date +%s)
STEP3_TIME=$((STEP3_END - STEP3_START))

echo "⏱️ ステップ3完了: ${STEP3_TIME}秒" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "ステップ4: マスク投影" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

STEP4_START=$(date +%s)

# マスク投影出力ディレクトリ
PROJECTION_DIR="$OUTPUT_DIR/mask_projection_output"
mkdir -p "$PROJECTION_DIR"

echo "🎭 マスク投影実行中..." | tee -a "$LOG_FILE"

timeout 300 docker run --rm --gpus all \
    -v "$PROJECT_ROOT:/workspace" \
    deepdrr-japan-mirror:latest bash -c "
    pip3 install torch torchvision pydicom opencv-python scipy --no-cache-dir > /dev/null 2>&1 &&
    cd /workspace &&
    python3 - <<'PYEOF'
import sys
sys.path.append('/workspace/mask_projection/src')
content = open('/workspace/mask_projection/src/mask_deepdrr_projection_thickness_gpt5pro8.py').read()
content = content.replace(
    'mask_nifti = Path(\"/workspace/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz\")',
    'mask_nifti = Path(\"$EAT_PAT_DIR/masks/eat_pat.nii.gz\")'
)
content = content.replace(
    'outdir = Path(\"/workspace/mask_projection/outputs\")',
    'outdir = Path(\"$PROJECTION_DIR\")'
)
exec(content)
PYEOF
    " 2>&1 | tee -a "$LOG_FILE"

STEP4_END=$(date +%s)
STEP4_TIME=$((STEP4_END - STEP4_START))

echo "⏱️ ステップ4完了: ${STEP4_TIME}秒" | tee -a "$LOG_FILE"

# 生成ファイル確認
echo "" | tee -a "$LOG_FILE"
echo "📊 生成ファイル確認:" | tee -a "$LOG_FILE"
if [ -d "$PROJECTION_DIR" ]; then
    ls -la "$PROJECTION_DIR"/*.png 2>/dev/null | tee -a "$LOG_FILE"
fi

# 全体のサマリ
TOTAL_TIME=$((STEP1_TIME + STEP2_TIME + STEP3_TIME + STEP4_TIME))

echo "" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "🎉 通常モードパイプライン完了" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "📊 実行時間サマリ:" | tee -a "$LOG_FILE"
echo "  - ステップ1 (TotalSeg通常): ${STEP1_TIME}秒" | tee -a "$LOG_FILE"
echo "  - ステップ2 (EAT/PAT抽出): ${STEP2_TIME}秒" | tee -a "$LOG_FILE"
echo "  - ステップ3 (DeepDRR生成): ${STEP3_TIME}秒" | tee -a "$LOG_FILE"
echo "  - ステップ4 (マスク投影): ${STEP4_TIME}秒" | tee -a "$LOG_FILE"
echo "  - 総実行時間: ${TOTAL_TIME}秒" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📂 出力ディレクトリ: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "📄 ログファイル: $LOG_FILE" | tee -a "$LOG_FILE"

# 8月31日の結果との比較
echo ""
echo "================================================"
echo "📊 8月31日の結果との比較"
echo "================================================"

if [ -f "$EAT_PAT_DIR/masks/eat_pat.nii.gz" ] && [ -f "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz" ]; then
    echo "🔍 マスクファイルサイズ比較:"
    ls -lh "$EAT_PAT_DIR/masks/eat_pat.nii.gz" /home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz
    
    # MD5ハッシュ比較
    echo ""
    echo "🔐 MD5ハッシュ比較:"
    md5sum "$EAT_PAT_DIR/masks/eat_pat.nii.gz" /home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz
fi

echo ""
echo "✅ テスト完了"