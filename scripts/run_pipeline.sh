#!/bin/bash
"""
統合パイプライン実行スクリプト
STEP3～STEP6を順番に実行します
"""

echo "========================================="
echo "CT-RATE EAT/PAT解析パイプライン"
echo "========================================="

# 作業ディレクトリ
WORKDIR=~/ctrate_ws
cd "$WORKDIR" || exit 1

# Python仮想環境のアクティベート
if [ -f "$WORKDIR/env/bin/activate" ]; then
    source "$WORKDIR/env/bin/activate"
elif [ -f "$WORKDIR/.venv/bin/activate" ]; then
    source "$WORKDIR/.venv/bin/activate"
else
    echo "警告: Python仮想環境が見つかりません"
fi

# 引数チェック
if [ $# -lt 1 ]; then
    echo "使用法: $0 <CTファイルパス> [出力ディレクトリ]"
    echo ""
    echo "例:"
    echo "  $0 ~/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    echo "  $0 ~/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz ~/ctrate_ws/outputs/case1"
    exit 1
fi

CT_FILE="$1"
OUTPUT_BASE="${2:-$WORKDIR/outputs/pipeline_output}"

# CTファイルの存在確認
if [ ! -f "$CT_FILE" ]; then
    echo "エラー: CTファイルが見つかりません: $CT_FILE"
    exit 1
fi

# ケース名の取得
CASE_NAME=$(basename "$CT_FILE" .nii.gz)
echo "処理ケース: $CASE_NAME"

# 出力ディレクトリの作成
OUTPUT_DIR="$OUTPUT_BASE/$CASE_NAME"
mkdir -p "$OUTPUT_DIR"/{segmentation,eat_pat,projections}

echo ""
echo "出力ディレクトリ: $OUTPUT_DIR"
echo ""

# ========================================
# STEP 1: TotalSegmentatorで心臓セグメンテーション
# ========================================
echo "----------------------------------------"
echo "STEP 1: 心臓セグメンテーション実行中..."
echo "----------------------------------------"

SEGMENTATION_DIR="$OUTPUT_DIR/segmentation"

# TotalSegmentatorライセンスチェック
DEFAULT_LICENSE_KEY="aca_DHDKRFJMHUX0PK"

# config.jsonでライセンスを確認
if [ ! -f ~/.totalsegmentator/config.json ] || ! grep -q "license_number" ~/.totalsegmentator/config.json; then
    echo "警告: TotalSegmentatorライセンスが設定されていません"
    echo "自動でライセンスを設定します..."
    
    mkdir -p ~/.totalsegmentator
    docker run --rm -it --gpus all \
        -v ~/.totalsegmentator:/root/.totalsegmentator \
        wasserth/totalsegmentator:2.10.0 \
        totalseg_set_license -l "$DEFAULT_LICENSE_KEY"
    
    if [ $? -ne 0 ]; then
        echo "エラー: ライセンス設定に失敗しました"
        exit 1
    fi
    echo "ライセンスが設定されました"
else
    echo "ライセンス確認: OK ($(grep license_number ~/.totalsegmentator/config.json | cut -d'"' -f4))"
fi

# セグメンテーション実行
docker run --rm --gpus all --ipc=host \
    -v "$HOME:$HOME" \
    wasserth/totalsegmentator:2.10.0 \
    TotalSegmentator -i "$CT_FILE" -o "$SEGMENTATION_DIR" \
    --task heartchambers_highres --device gpu --robust_crop --body_seg

# 心筋マスクの確認
MYOCARDIUM_MASK="$SEGMENTATION_DIR/heart_myocardium.nii.gz"
if [ ! -f "$MYOCARDIUM_MASK" ]; then
    echo "エラー: 心筋セグメンテーションが失敗しました"
    exit 1
fi

echo "✓ 心臓セグメンテーション完了"

# ========================================
# STEP 2: EAT/PAT抽出
# ========================================
echo ""
echo "----------------------------------------"
echo "STEP 2: EAT/PAT抽出中..."
echo "----------------------------------------"

EAT_PAT_DIR="$OUTPUT_DIR/eat_pat"

python "$WORKDIR/scripts/eat_pat_extraction.py" \
    "$CT_FILE" \
    "$MYOCARDIUM_MASK" \
    -o "$EAT_PAT_DIR" \
    --eat-distance 5.0 \
    --pat-distance 10.0

# EAT/PATマスクの確認
EAT_MASK=$(ls "$EAT_PAT_DIR"/*_eat_5mm.nii.gz 2>/dev/null | head -1)
PAT_MASK=$(ls "$EAT_PAT_DIR"/*_pat_10mm.nii.gz 2>/dev/null | head -1)

if [ -z "$EAT_MASK" ] || [ -z "$PAT_MASK" ]; then
    echo "エラー: EAT/PAT抽出が失敗しました"
    exit 1
fi

echo "✓ EAT/PAT抽出完了"
echo "  EAT: $EAT_MASK"
echo "  PAT: $PAT_MASK"

# ========================================
# STEP 3: 2D投影生成
# ========================================
echo ""
echo "----------------------------------------"
echo "STEP 3: 2D投影生成中..."
echo "----------------------------------------"

PROJECTION_DIR="$OUTPUT_DIR/projections"

# EAT投影
echo "EAT投影生成中..."
python "$WORKDIR/scripts/generate_2d_projections.py" \
    "$CT_FILE" \
    "$EAT_MASK" \
    -o "$PROJECTION_DIR/eat" \
    --all-axes

# PAT投影
echo "PAT投影生成中..."
python "$WORKDIR/scripts/generate_2d_projections.py" \
    "$CT_FILE" \
    "$PAT_MASK" \
    -o "$PROJECTION_DIR/pat" \
    --all-axes

echo "✓ 2D投影生成完了"

# ========================================
# 結果サマリー
# ========================================
echo ""
echo "========================================="
echo "パイプライン実行完了!"
echo "========================================="
echo ""
echo "結果ファイル:"
echo "  セグメンテーション: $SEGMENTATION_DIR"
echo "  EAT/PATマスク: $EAT_PAT_DIR"
echo "  2D投影: $PROJECTION_DIR"
echo ""

# ボリューム情報の表示
if [ -f "$EAT_PAT_DIR/${CASE_NAME}_volumes.txt" ]; then
    echo "ボリューム測定結果:"
    cat "$EAT_PAT_DIR/${CASE_NAME}_volumes.txt"
fi

echo ""
echo "全処理完了: $CASE_NAME"