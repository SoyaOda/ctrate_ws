#!/bin/bash
# 修正版HARTAテストスクリプト - 座標系保持版

echo "========================================="
echo "HARTA座標系保持版 テスト"
echo "========================================="

# ベースディレクトリ
BASE_DIR="/home/soya/ctrate_ws"
HARTA_ENV="$BASE_DIR/harta_env"
HARTA_REPO="$BASE_DIR/tools/HARTA"

# テストケース
TEST_NIFTI="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
TEST_CASE_ID="test_valid_2_a_1_fixed"
TEST_OUTPUT_DIR="$BASE_DIR/outputs/test_harta_fixed"
TEST_DICOM_DIR="$TEST_OUTPUT_DIR/dicom/$TEST_CASE_ID"
TEST_HARTA_OUTPUT="$TEST_OUTPUT_DIR/harta_output/$TEST_CASE_ID"
TEST_NIFTI_OUTPUT="$TEST_OUTPUT_DIR/nifti/$TEST_CASE_ID"

# クリーンアップ（前回の実行結果を削除）
echo "前回の結果をクリーンアップ中..."
rm -rf "$TEST_OUTPUT_DIR"

# 出力ディレクトリ作成
mkdir -p "$TEST_OUTPUT_DIR"
mkdir -p "$TEST_DICOM_DIR"
mkdir -p "$TEST_HARTA_OUTPUT"
mkdir -p "$TEST_NIFTI_OUTPUT"

echo "テストケース: $TEST_CASE_ID"
echo "入力NIfTI: $TEST_NIFTI"
echo ""

# Python環境有効化
source "$HARTA_ENV/bin/activate"

# ステップ1: NIfTI → DICOM変換（座標系保持版）
echo "[1/4] NIfTI → DICOM変換（座標系保持版）..."
python "$BASE_DIR/scripts/nifti_to_dicom_preserve_geom.py" \
    -i "$TEST_NIFTI" \
    -o "$TEST_DICOM_DIR"

if [ $? -eq 0 ]; then
    echo "✓ DICOM変換成功（座標系保持）"
    echo "  変換先: $TEST_DICOM_DIR"
    echo "  DICOMファイル数: $(ls -1 $TEST_DICOM_DIR/*.dcm 2>/dev/null | wc -l)"
    echo "  座標系情報: $TEST_OUTPUT_DIR/dicom/${TEST_CASE_ID}_slice_map.json"
    echo ""
    
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
            for subdir in "$TEST_HARTA_OUTPUT"/*; do
                if [ -d "$subdir" ]; then
                    dirname=$(basename "$subdir")
                    file_count=$(ls -1 "$subdir" 2>/dev/null | wc -l)
                    echo "  $dirname/: $file_count files"
                fi
            done
        fi
        echo ""
        
        # ステップ3: 3D NIfTI変換（座標系保持版）
        echo "[3/4] 3D NIfTI変換（座標系保持版）..."
        NIFTI_FILE="$TEST_NIFTI_OUTPUT/eat_harta_fixed.nii.gz"
        
        python "$BASE_DIR/scripts/harta_masks_to_nifti_preserve_geom.py" \
            --png_dir "$TEST_HARTA_OUTPUT/fat" \
            --slice_map "$TEST_OUTPUT_DIR/dicom/${TEST_CASE_ID}_slice_map.json" \
            --ref "$TEST_NIFTI" \
            --out "$NIFTI_FILE"
        
        if [ -f "$NIFTI_FILE" ]; then
            echo "✓ NIfTI変換成功（座標系保持）"
            echo "  出力: $NIFTI_FILE"
            echo ""
            
            # ステップ4: 座標系検証
            echo "[4/4] 座標系検証..."
            python -c "
import SimpleITK as sitk
import numpy as np

# オリジナルとセグメンテーション読み込み
orig = sitk.ReadImage('$TEST_NIFTI')
seg = sitk.ReadImage('$NIFTI_FILE')

print('='*60)
print('座標系比較:')
print('='*60)

# サイズ比較
orig_size = orig.GetSize()
seg_size = seg.GetSize()
print(f'サイズ - オリジナル: {orig_size}, セグメント: {seg_size}')
if orig_size == seg_size:
    print('  ✓ サイズ一致')
else:
    print('  ⚠️ サイズ不一致')

# スペーシング比較
orig_spacing = orig.GetSpacing()
seg_spacing = seg.GetSpacing()
spacing_diff = np.array(orig_spacing) - np.array(seg_spacing)
print(f'スペーシング - オリジナル: {orig_spacing}')
print(f'            セグメント: {seg_spacing}')
if np.max(np.abs(spacing_diff)) < 0.001:
    print('  ✓ スペーシング一致')
else:
    print(f'  ⚠️ スペーシング不一致 (差: {spacing_diff})')

# 原点比較
orig_origin = orig.GetOrigin()
seg_origin = seg.GetOrigin()
origin_diff = np.array(orig_origin) - np.array(seg_origin)
print(f'原点 - オリジナル: {orig_origin}')
print(f'      セグメント: {seg_origin}')
if np.max(np.abs(origin_diff)) < 0.001:
    print('  ✓ 原点一致')
else:
    print(f'  ⚠️ 原点不一致 (差: {origin_diff})')

# Direction matrix比較
orig_dir = np.array(orig.GetDirection()).reshape(3,3)
seg_dir = np.array(seg.GetDirection()).reshape(3,3)
dir_diff = orig_dir - seg_dir
print(f'Direction matrix差の最大値: {np.max(np.abs(dir_diff)):.6f}')
if np.max(np.abs(dir_diff)) < 0.001:
    print('  ✓ Direction matrix一致')
else:
    print('  ⚠️ Direction matrix不一致')

# ボリューム計算
seg_array = sitk.GetArrayFromImage(seg)
voxel_volume_ml = np.prod(seg_spacing) / 1000.0
eat_volume_ml = np.sum(seg_array) * voxel_volume_ml
print(f'\\nEATボリューム: {eat_volume_ml:.2f} ml')
print(f'非ゼロボクセル数: {np.sum(seg_array)}')

# テスト点での座標確認
test_points = [
    [100, 100, 50],
    [256, 256, 100],
    [400, 400, 150]
]
print('\\nテスト点での物理座標比較:')
for pt in test_points:
    orig_phys = orig.TransformIndexToPhysicalPoint(pt)
    seg_phys = seg.TransformIndexToPhysicalPoint(pt)
    diff = np.array(orig_phys) - np.array(seg_phys)
    print(f'  インデックス {pt}:')
    print(f'    オリジナル: {orig_phys}')
    print(f'    セグメント: {seg_phys}')
    if np.max(np.abs(diff)) < 0.001:
        print('    ✓ 一致')
    else:
        print(f'    ⚠️ 不一致 (差: {diff})')
"
        else
            echo "✗ NIfTI変換失敗"
        fi
    else
        echo "✗ HARTAセグメンテーション失敗"
    fi
else
    echo "✗ DICOM変換失敗"
fi

# Python環境を無効化
deactivate

echo ""
echo "========================================="
echo "テスト完了"
echo "出力ディレクトリ: $TEST_OUTPUT_DIR"
echo "========================================="