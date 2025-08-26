#!/bin/bash
# HARTAを使用したEAT抽出バッチ処理スクリプト

echo "========================================="
echo "HARTAバッチ処理: CTデータのEAT抽出"
echo "========================================="

# ベースディレクトリ
BASE_DIR="/home/soya/ctrate_ws"
DATA_DIR="$BASE_DIR/data/CT-RATE-v2/dataset/valid_fixed"
OUTPUT_BASE="$BASE_DIR/outputs/harta_batch_results"
DICOM_BASE="$BASE_DIR/data/dicom_converted"
HARTA_ENV="$BASE_DIR/harta_env"
HARTA_REPO="$BASE_DIR/tools/HARTA"

# 環境確認
if [ ! -d "$HARTA_ENV" ]; then
    echo "[ERROR] HARTA環境が見つかりません: $HARTA_ENV"
    exit 1
fi

if [ ! -d "$HARTA_REPO" ]; then
    echo "[ERROR] HARTAリポジトリが見つかりません: $HARTA_REPO"
    exit 1
fi

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_BASE"
mkdir -p "$DICOM_BASE"

# 処理するCTファイルのリスト取得（テスト用に最初の3つ）
CT_FILES=$(find "$DATA_DIR" -name "*.nii.gz" -type f | sort | head -n 3)

# 総ファイル数
TOTAL_FILES=$(echo "$CT_FILES" | wc -l)
echo "処理対象: $TOTAL_FILES 個のCTファイル"
echo ""

# 処理カウンタ
COUNT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# 結果サマリーファイル
SUMMARY_FILE="$OUTPUT_BASE/harta_processing_summary.txt"
echo "HARTA EAT Batch Processing Summary" > "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "Started at: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Python環境有効化
source "$HARTA_ENV/bin/activate"

# 各CTファイルを処理
for CT_FILE in $CT_FILES; do
    COUNT=$((COUNT + 1))
    
    # ケースIDを抽出（ファイル名から）
    CASE_ID=$(basename "$CT_FILE" .nii.gz)
    
    echo "----------------------------------------"
    echo "[$COUNT/$TOTAL_FILES] 処理中: $CASE_ID"
    echo "CTファイル: $CT_FILE"
    
    # 出力ディレクトリ設定
    CASE_OUTPUT_DIR="$OUTPUT_BASE/$CASE_ID"
    DICOM_DIR="$DICOM_BASE/$CASE_ID"
    HARTA_OUTPUT_DIR="$CASE_OUTPUT_DIR/harta_output"
    NIFTI_OUTPUT_DIR="$CASE_OUTPUT_DIR/nifti"
    
    mkdir -p "$CASE_OUTPUT_DIR"
    mkdir -p "$HARTA_OUTPUT_DIR"
    mkdir -p "$NIFTI_OUTPUT_DIR"
    
    # ステップ1: NIfTI → DICOM変換
    echo "  [1/4] NIfTI → DICOM変換中..."
    if bash "$BASE_DIR/scripts/nifti_to_dicom.sh" "$CT_FILE" "$DICOM_DIR" > /dev/null 2>&1; then
        echo "  ✓ DICOM変換成功"
        
        # ステップ2: HARTA自動セグメンテーション実行
        echo "  [2/4] HARTA EATセグメンテーション実行中..."
        HARTA_LOG="$CASE_OUTPUT_DIR/harta_log.txt"
        
        if python "$BASE_DIR/scripts/run_harta_auto.py" \
            --harta_repo "$HARTA_REPO" \
            --dicom_case "$DICOM_DIR" \
            --out_dir "$HARTA_OUTPUT_DIR" \
            --python python > "$HARTA_LOG" 2>&1; then
            
            # ログから結果を抽出
            if grep -q "SUCCESS:" "$HARTA_LOG"; then
                HARTA_RESULT=$(grep "SUCCESS:" "$HARTA_LOG" | tail -1)
                echo "  ✓ HARTAセグメンテーション成功"
                echo "    $HARTA_RESULT"
                
                # ステップ3: HARTA出力を3D NIfTIに変換
                echo "  [3/4] 3D NIfTI変換中..."
                NIFTI_FILE="$NIFTI_OUTPUT_DIR/eat_harta.nii.gz"
                
                if python "$BASE_DIR/scripts/harta_masks_to_nifti.py" \
                    --dicom_dir "$DICOM_DIR" \
                    --harta_output_dir "$HARTA_OUTPUT_DIR" \
                    --out_nifti "$NIFTI_FILE" > /dev/null 2>&1; then
                    
                    echo "  ✓ NIfTI変換成功"
                    
                    # ステップ4: ボリューム計算と記録
                    echo "  [4/4] ボリューム計算中..."
                    # Docker経由でボリューム計算
                    VOLUME_RESULT=$(docker run --rm \
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
print(f'{eat_volume_ml:.2f}')
" 2>/dev/null)
                    
                    if [ ! -z "$VOLUME_RESULT" ]; then
                        echo "  ✓ EAT抽出完了: ${VOLUME_RESULT} ml"
                        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                        echo "$CASE_ID: SUCCESS - EAT: ${VOLUME_RESULT} ml (HARTA)" >> "$SUMMARY_FILE"
                    else
                        echo "  ✗ ボリューム計算失敗"
                        FAIL_COUNT=$((FAIL_COUNT + 1))
                        echo "$CASE_ID: PARTIAL - HARTA completed but volume calc failed" >> "$SUMMARY_FILE"
                    fi
                else
                    echo "  ✗ NIfTI変換失敗"
                    FAIL_COUNT=$((FAIL_COUNT + 1))
                    echo "$CASE_ID: FAILED - NIfTI conversion error" >> "$SUMMARY_FILE"
                fi
            else
                echo "  ✗ HARTAセグメンテーション失敗"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                echo "$CASE_ID: FAILED - HARTA segmentation error" >> "$SUMMARY_FILE"
                # エラーログ表示
                if [ -f "$HARTA_LOG" ]; then
                    echo "  [ERROR LOG]"
                    tail -n 5 "$HARTA_LOG"
                fi
            fi
        else
            echo "  ✗ HARTA実行エラー"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "$CASE_ID: FAILED - HARTA execution error" >> "$SUMMARY_FILE"
        fi
    else
        echo "  ✗ DICOM変換失敗"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "$CASE_ID: FAILED - DICOM conversion error" >> "$SUMMARY_FILE"
    fi
    
    echo ""
done

# Python環境を無効化
deactivate

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
    grep -E ": SUCCESS|: FAILED|: PARTIAL" "$SUMMARY_FILE" | tail -n 10
fi