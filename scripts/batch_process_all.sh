#!/bin/bash
# 複数のCTデータに対してSTEP3(TotalSegmentator)とSTEP4(EAT/PAT)を実行

echo "========================================="
echo "バッチ処理: 全CTデータのEAT/PAT抽出"
echo "========================================="

# ベースディレクトリ
BASE_DIR="/home/soya/ctrate_ws"
DATA_DIR="$BASE_DIR/data/CT-RATE-v2/dataset/valid_fixed"
OUTPUT_BASE="$BASE_DIR/outputs/batch_results"

# 処理するCTファイルのリスト取得（全ての.nii.gzファイル）
CT_FILES=$(find "$DATA_DIR" -name "*.nii.gz" -type f | sort)

# 総ファイル数
TOTAL_FILES=$(echo "$CT_FILES" | wc -l)
echo "処理対象: $TOTAL_FILES 個のCTファイル"
echo ""

# 処理カウンタ
COUNT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# 結果サマリーファイル
SUMMARY_FILE="$OUTPUT_BASE/processing_summary.txt"
mkdir -p "$OUTPUT_BASE"
echo "EAT/PAT Batch Processing Summary" > "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "Started at: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 各CTファイルを処理
for CT_FILE in $CT_FILES; do
    COUNT=$((COUNT + 1))
    
    # パスから識別子を抽出（例: valid_1_a_1）
    CASE_ID=$(echo "$CT_FILE" | sed 's/.*\/\(valid_[0-9]\)\/\(valid_[0-9]_[a-z]\)\/\(valid_[0-9]_[a-z]_[0-9]\).*/\1_\3/')
    
    echo "----------------------------------------"
    echo "[$COUNT/$TOTAL_FILES] 処理中: $CASE_ID"
    echo "CTファイル: $CT_FILE"
    
    # 出力ディレクトリ設定
    CASE_OUTPUT_DIR="$OUTPUT_BASE/$CASE_ID"
    mkdir -p "$CASE_OUTPUT_DIR"
    
    # STEP3: TotalSegmentator (heartchambers_highres)
    echo "  STEP3: 心臓セグメンテーション実行中..."
    SEGMENTATION_OUTPUT="$CASE_OUTPUT_DIR/segmentation"
    mkdir -p "$SEGMENTATION_OUTPUT"
    
    # Docker経由でTotalSegmentator実行
    timeout 600 docker run --rm --gpus all --ipc=host \
        --shm-size=8g \
        -v "$HOME:/home/soya" \
        -v "$HOME/.totalsegmentator:/root/.totalsegmentator" \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        wasserth/totalsegmentator:2.10.0 \
        bash -c "
            TotalSegmentator \
                -i '$CT_FILE' \
                -o /tmp/output \
                --task heartchambers_highres \
                --device gpu \
                --force_split \
                --robust_crop \
                --body_seg 2>/dev/null && \
            cp -r /tmp/output/* '$SEGMENTATION_OUTPUT/' 2>/dev/null
        " > /dev/null 2>&1
    
    # セグメンテーション結果確認
    if [ -f "$SEGMENTATION_OUTPUT/heart_myocardium.nii.gz" ]; then
        echo "  ✓ セグメンテーション成功"
        
        # STEP4: EAT/PAT抽出
        echo "  STEP4: EAT/PAT抽出中..."
        EAT_PAT_OUTPUT="$CASE_OUTPUT_DIR/eat_pat"
        mkdir -p "$EAT_PAT_OUTPUT"
        
        # Docker経由でEAT/PAT抽出
        docker run --rm --gpus all \
            -v "$HOME:/home/soya" \
            wasserth/totalsegmentator:2.10.0 \
            python3 -c "
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from pathlib import Path

# ファイルパス
ct_path = '$CT_FILE'
myocardium_path = '$SEGMENTATION_OUTPUT/heart_myocardium.nii.gz'
output_dir = '$EAT_PAT_OUTPUT'

# データ読み込み
ct_nii = nib.load(ct_path)
ct_data = ct_nii.get_fdata()
myocardium_nii = nib.load(myocardium_path)
myocardium_mask = myocardium_nii.get_fdata() > 0

# ボクセルサイズ
spacing = ct_nii.header.get_zooms()[:3]
voxel_volume_ml = np.prod(spacing) / 1000.0

# 脂肪マスク（HU値: -190 ~ -30）
fat_mask = (ct_data >= -190) & (ct_data <= -30)

# 膨張処理
struct_elem = generate_binary_structure(3, 2)
min_spacing = min(spacing)
dilation_5mm = int(np.ceil(5.0 / min_spacing))
dilation_10mm = int(np.ceil(10.0 / min_spacing))

# EAT（5mm膨張）
eat_roi = myocardium_mask.copy()
for _ in range(dilation_5mm):
    eat_roi = binary_dilation(eat_roi, struct_elem)

# PAT（10mm膨張）
pat_roi = myocardium_mask.copy()
for _ in range(dilation_10mm):
    pat_roi = binary_dilation(pat_roi, struct_elem)

# 脂肪抽出
eat_mask = fat_mask & eat_roi & ~myocardium_mask
pat_mask = fat_mask & pat_roi & ~myocardium_mask

# 体積計算
eat_volume_ml = np.sum(eat_mask) * voxel_volume_ml
pat_volume_ml = np.sum(pat_mask) * voxel_volume_ml

# 保存
Path(output_dir).mkdir(parents=True, exist_ok=True)
nib.save(nib.Nifti1Image(eat_mask.astype(np.uint8), ct_nii.affine, ct_nii.header), 
         f'{output_dir}/eat_5mm.nii.gz')
nib.save(nib.Nifti1Image(pat_mask.astype(np.uint8), ct_nii.affine, ct_nii.header), 
         f'{output_dir}/pat_10mm.nii.gz')

# 結果出力
print(f'EAT:{eat_volume_ml:.2f}ml,PAT:{pat_volume_ml:.2f}ml')
" 2>/dev/null | tail -1 > "$EAT_PAT_OUTPUT/volumes.txt"
        
        # 結果確認
        if [ -f "$EAT_PAT_OUTPUT/eat_5mm.nii.gz" ] && [ -f "$EAT_PAT_OUTPUT/pat_10mm.nii.gz" ]; then
            VOLUMES=$(cat "$EAT_PAT_OUTPUT/volumes.txt" 2>/dev/null)
            echo "  ✓ EAT/PAT抽出成功: $VOLUMES"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            
            # サマリーに記録
            echo "$CASE_ID: SUCCESS - $VOLUMES" >> "$SUMMARY_FILE"
        else
            echo "  ✗ EAT/PAT抽出失敗"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "$CASE_ID: FAILED - EAT/PAT extraction error" >> "$SUMMARY_FILE"
        fi
    else
        echo "  ✗ セグメンテーション失敗"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "$CASE_ID: FAILED - Segmentation error" >> "$SUMMARY_FILE"
    fi
    
    echo ""
done

# 最終サマリー
echo "========================================="
echo "処理完了"
echo "========================================="
echo "総処理数: $TOTAL_FILES"
echo "成功: $SUCCESS_COUNT"
echo "失敗: $FAIL_COUNT"
echo ""
echo "結果保存先: $OUTPUT_BASE"
echo "サマリーファイル: $SUMMARY_FILE"

# サマリーファイルに最終結果追記
echo "" >> "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "Completed at: $(date)" >> "$SUMMARY_FILE"
echo "Total: $TOTAL_FILES, Success: $SUCCESS_COUNT, Failed: $FAIL_COUNT" >> "$SUMMARY_FILE"

# 結果一覧表示
if [ -f "$SUMMARY_FILE" ]; then
    echo ""
    echo "=== 処理結果一覧 ==="
    grep -E "valid_.*: " "$SUMMARY_FILE"
fi