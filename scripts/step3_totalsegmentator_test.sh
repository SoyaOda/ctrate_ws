#!/bin/bash
# STEP3: TotalSegmentatorの公式Dockerで心臓セグメントのテスト実行

echo "========================================="
echo "STEP3: TotalSegmentator心臓セグメントテスト"
echo "========================================="

# TotalSegmentatorディレクトリの作成
mkdir -p ~/.totalsegmentator

# 1. Docker イメージのpull
echo "1. TotalSegmentator Dockerイメージをダウンロード中..."
# 正しい公式イメージ: wasserth/totalsegmentator
docker pull wasserth/totalsegmentator:2.10.0

# 2. ライセンス設定
echo ""
echo "2. TotalSegmentatorライセンスの設定"

# デフォルトライセンスキー
DEFAULT_LICENSE_KEY="aca_DHDKRFJMHUX0PK"
LICENSE_KEY="$DEFAULT_LICENSE_KEY"

echo "ライセンスキーを設定します: $LICENSE_KEY"

# ライセンスをコンテナに登録
echo "ライセンスを登録中..."
docker run --rm -it --gpus all \
    -v ~/.totalsegmentator:/root/.totalsegmentator \
    wasserth/totalsegmentator:2.10.0 \
    totalseg_set_license -l "$LICENSE_KEY"

# 3. テスト用NIfTIファイルの選択
echo ""
echo "3. テスト用CT画像の選択"

# CT-RATE v2データの検索
CT_FILES=(~/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/*/*/*.nii.gz)

if [ ${#CT_FILES[@]} -eq 0 ]; then
    echo "エラー: CT-RATE v2データが見つかりません"
    echo "データをダウンロードしてから再実行してください"
    exit 1
fi

# 最初のファイルを使用
CT_FILE="${CT_FILES[0]}"
echo "使用するCTファイル: $CT_FILE"

# 出力ディレクトリの作成
OUTPUT_DIR=~/ctrate_ws/outputs/ts_test
mkdir -p "$OUTPUT_DIR"

# 4. heartchambers_highresでの推論実行
echo ""
echo "4. 心臓セグメンテーション実行中..."
echo "（GPUを使用、処理には数分かかる場合があります）"

docker run --rm -it --gpus all --ipc=host \
    -v "$HOME:$HOME" \
    wasserth/totalsegmentator:2.10.0 \
    TotalSegmentator -i "$CT_FILE" -o "$OUTPUT_DIR" \
    --task heartchambers_highres --device gpu --robust_crop --body_seg

# 5. 結果の確認
echo ""
echo "5. 生成されたセグメンテーション結果:"
echo "----------------------------------------"

if ls "$OUTPUT_DIR"/heart_*.nii.gz 1> /dev/null 2>&1; then
    ls -la "$OUTPUT_DIR"/heart_*.nii.gz
    echo ""
    echo "✓ 心臓セグメンテーションが正常に完了しました"
    echo ""
    echo "生成されたファイル:"
    for file in "$OUTPUT_DIR"/heart_*.nii.gz; do
        basename "$file"
    done
else
    echo "エラー: 心臓セグメンテーションファイルが生成されていません"
    echo "ログを確認してください"
    exit 1
fi

echo ""
echo "========================================="
echo "STEP3 完了: 次はSTEP4（BOA環境構築）に進んでください"
echo "========================================="