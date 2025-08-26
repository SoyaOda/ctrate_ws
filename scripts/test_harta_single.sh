#!/bin/bash
# HARTAの単一ケーステストスクリプト

echo "========================================="
echo "HARTA単体テスト: 1ケースのEAT抽出"
echo "========================================="

# ベースディレクトリ
BASE_DIR="/home/soya/ctrate_ws"
HARTA_ENV="$BASE_DIR/harta_env"
HARTA_REPO="$BASE_DIR/tools/HARTA"

# テストケース
TEST_NIFTI="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
TEST_CASE_ID="test_valid_2_a_1"
TEST_OUTPUT_DIR="$BASE_DIR/outputs/test_harta"
TEST_DICOM_DIR="$TEST_OUTPUT_DIR/dicom/$TEST_CASE_ID"
TEST_HARTA_OUTPUT="$TEST_OUTPUT_DIR/harta_output/$TEST_CASE_ID"
TEST_NIFTI_OUTPUT="$TEST_OUTPUT_DIR/nifti/$TEST_CASE_ID"

# 出力ディレクトリ作成
mkdir -p "$TEST_OUTPUT_DIR"
mkdir -p "$TEST_DICOM_DIR"
mkdir -p "$TEST_HARTA_OUTPUT"
mkdir -p "$TEST_NIFTI_OUTPUT"

echo "テストケース: $TEST_CASE_ID"
echo "入力NIfTI: $TEST_NIFTI"
echo ""

# ステップ1: NIfTI → DICOM変換
echo "[1/4] NIfTI → DICOM変換..."
if bash "$BASE_DIR/scripts/nifti_to_dicom.sh" "$TEST_NIFTI" "$TEST_DICOM_DIR"; then
    echo "✓ DICOM変換成功"
    echo "  変換先: $TEST_DICOM_DIR"
    echo "  DICOMファイル数: $(ls -1 $TEST_DICOM_DIR/*.dcm 2>/dev/null | wc -l)"
    echo ""
    
    # Python環境有効化
    source "$HARTA_ENV/bin/activate"
    
    # ステップ2: HARTA実行
    echo "[2/4] HARTA EATセグメンテーション実行..."
    echo "  入力DICOM: $TEST_DICOM_DIR"
    echo "  出力先: $TEST_HARTA_OUTPUT"
    
    python "$BASE_DIR/scripts/run_harta_auto.py" \
        --harta_repo "$HARTA_REPO" \
        --dicom_case "$TEST_DICOM_DIR" \
        --out_dir "$TEST_HARTA_OUTPUT" \
        --python python
    
    if [ $? -eq 0 ]; then
        echo "✓ HARTAセグメンテーション完了"
        echo ""
        
        # 出力ファイル確認
        echo "HARTA出力ファイル確認:"
        echo "-------------------"
        if [ -d "$TEST_HARTA_OUTPUT" ]; then
            echo "ディレクトリ構造:"
            ls -la "$TEST_HARTA_OUTPUT/" | head -n 10
            
            # サブディレクトリを確認
            for subdir in "$TEST_HARTA_OUTPUT"/*; do
                if [ -d "$subdir" ]; then
                    dirname=$(basename "$subdir")
                    file_count=$(ls -1 "$subdir" 2>/dev/null | wc -l)
                    echo "  $dirname/: $file_count files"
                fi
            done
        fi
        echo ""
        
        # ステップ3: NIfTI変換
        echo "[3/4] 3D NIfTI変換..."
        NIFTI_FILE="$TEST_NIFTI_OUTPUT/eat_harta.nii.gz"
        
        python "$BASE_DIR/scripts/harta_masks_to_nifti.py" \
            --dicom_dir "$TEST_DICOM_DIR" \
            --harta_output_dir "$TEST_HARTA_OUTPUT" \
            --out_nifti "$NIFTI_FILE"
        
        if [ -f "$NIFTI_FILE" ]; then
            echo "✓ NIfTI変換成功"
            echo "  出力: $NIFTI_FILE"
            echo ""
            
            # ステップ4: ボリューム確認
            echo "[4/4] ボリューム計算..."
            docker run --rm \
                -v "$HOME:/home/soya" \
                wasserth/totalsegmentator:2.10.0 \
                python3 -c "
import nibabel as nib
import numpy as np
eat_nii = nib.load('$NIFTI_FILE')
eat_mask = eat_nii.get_fdata() > 0
spacing = eat_nii.header.get_zooms()[:3]
voxel_volume_ml = np.prod(spacing) / 1000.0
eat_volume_ml = np.sum(eat_mask) * voxel_volume_ml
print(f'EATボリューム: {eat_volume_ml:.2f} ml')
print(f'マスクサイズ: {eat_mask.shape}')
print(f'スペーシング: {spacing}')
print(f'非ゼロボクセル数: {np.sum(eat_mask)}')
"
        else
            echo "✗ NIfTI変換失敗"
        fi
    else
        echo "✗ HARTAセグメンテーション失敗"
    fi
    
    # Python環境を無効化
    deactivate
else
    echo "✗ DICOM変換失敗"
fi

echo ""
echo "========================================="
echo "テスト完了"
echo "出力ディレクトリ: $TEST_OUTPUT_DIR"
echo "========================================="