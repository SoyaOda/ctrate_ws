#!/bin/bash
# STEP4: EAT/PAT抽出（Docker版）

echo "========================================="
echo "STEP4: EAT/PAT抽出"
echo "========================================="

# 設定
BASE_DIR="/home/soya/ctrate_ws"
CT_PATH="$BASE_DIR/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
MYOCARDIUM_PATH="$BASE_DIR/outputs/step3_heartchambers/heart_myocardium.nii.gz"
OUTPUT_DIR="$BASE_DIR/outputs/step4_eat_pat"

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

# Pythonスクリプトの内容を直接実行
docker run --rm --gpus all \
    -v "$HOME:/home/soya" \
    wasserth/totalsegmentator:2.10.0 \
    python3 -c "
#!/usr/bin/env python3
import sys
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from pathlib import Path

# 設定
ct_path = '$CT_PATH'
myocardium_path = '$MYOCARDIUM_PATH'
output_dir = '$OUTPUT_DIR'

print('読み込み中:', ct_path)

# CT画像と心筋マスク読み込み
ct_nii = nib.load(ct_path)
ct_data = ct_nii.get_fdata()

if not os.path.exists(myocardium_path):
    print(f'警告: 心筋セグメンテーションが見つかりません: {myocardium_path}')
    # 代替パスを試す
    alt_path = myocardium_path.replace('heart_myocardium', 'heart').replace('step3_heartchambers', 'step3_optimized')
    if os.path.exists(alt_path):
        myocardium_path = alt_path
        print(f'使用: {alt_path}')

myocardium_nii = nib.load(myocardium_path)
myocardium_mask = myocardium_nii.get_fdata() > 0

# ボクセルサイズ取得
spacing = ct_nii.header.get_zooms()[:3]
voxel_volume_ml = np.prod(spacing) / 1000.0

print(f'CT画像サイズ: {ct_data.shape}')
print(f'ボクセルサイズ: {spacing} mm')
print(f'ボクセル体積: {voxel_volume_ml:.6f} ml')

# 脂肪組織の閾値設定（HU値: -190 ~ -30）
fat_mask = (ct_data >= -190) & (ct_data <= -30)

# 3D構造要素（26近傍）
struct_elem = generate_binary_structure(3, 2)

# 膨張距離計算
min_spacing = min(spacing)
dilation_5mm = int(np.ceil(5.0 / min_spacing))
dilation_10mm = int(np.ceil(10.0 / min_spacing))

print(f'膨張ステップ数: EAT={dilation_5mm}, PAT={dilation_10mm}')

# EAT領域作成（心筋から5mm膨張）
print('EAT領域計算中...')
eat_roi = myocardium_mask.copy()
for i in range(dilation_5mm):
    eat_roi = binary_dilation(eat_roi, struct_elem)
    if (i+1) % 2 == 0:
        print(f'  膨張: {i+1}/{dilation_5mm}')

# PAT領域作成（心筋から10mm膨張）  
print('PAT領域計算中...')
pat_roi = myocardium_mask.copy()
for i in range(dilation_10mm):
    pat_roi = binary_dilation(pat_roi, struct_elem)
    if (i+1) % 2 == 0:
        print(f'  膨張: {i+1}/{dilation_10mm}')

# EAT/PAT抽出
eat_mask = fat_mask & eat_roi & ~myocardium_mask
pat_mask = fat_mask & pat_roi & ~myocardium_mask

# 体積計算
eat_volume_ml = np.sum(eat_mask) * voxel_volume_ml
pat_volume_ml = np.sum(pat_mask) * voxel_volume_ml

# 保存
Path(output_dir).mkdir(parents=True, exist_ok=True)
eat_path = os.path.join(output_dir, 'eat_5mm.nii.gz')
pat_path = os.path.join(output_dir, 'pat_10mm.nii.gz')

print(f'保存中: {eat_path}')
nib.save(nib.Nifti1Image(eat_mask.astype(np.uint8), ct_nii.affine, ct_nii.header), eat_path)

print(f'保存中: {pat_path}')
nib.save(nib.Nifti1Image(pat_mask.astype(np.uint8), ct_nii.affine, ct_nii.header), pat_path)

# 結果表示
print('')
print('=== EAT/PAT抽出結果 ===')
print(f'EAT体積: {eat_volume_ml:.2f} ml ({np.sum(eat_mask)} voxels)')
print(f'PAT体積: {pat_volume_ml:.2f} ml ({np.sum(pat_mask)} voxels)')
print(f'EAT保存先: {eat_path}')
print(f'PAT保存先: {pat_path}')

# 統計情報保存
stats_path = os.path.join(output_dir, 'eat_pat_stats.txt')
with open(stats_path, 'w') as f:
    f.write('EAT/PAT Extraction Statistics\\n')
    f.write('=' * 40 + '\\n')
    f.write(f'EAT volume: {eat_volume_ml:.2f} ml\\n')
    f.write(f'PAT volume: {pat_volume_ml:.2f} ml\\n')
"

echo ""
echo "========================================="
echo "結果確認"
echo "========================================="

if [ -f "$OUTPUT_DIR/eat_5mm.nii.gz" ] && [ -f "$OUTPUT_DIR/pat_10mm.nii.gz" ]; then
    echo "✓ 成功: EAT/PATマスクが生成されました"
    ls -lh "$OUTPUT_DIR"/*.nii.gz
    
    if [ -f "$OUTPUT_DIR/eat_pat_stats.txt" ]; then
        echo ""
        cat "$OUTPUT_DIR/eat_pat_stats.txt"
    fi
else
    echo "✗ エラー: 出力ファイルが生成されませんでした"
fi