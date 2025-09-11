#!/bin/bash
# バッチ推論スクリプト
set -e

echo "🚀 バッチ推論開始"

# 引数確認
if [ "$#" -lt 2 ]; then
    echo "使用法: $0 <output_base_dir> <ct_file1> [ct_file2] ..."
    exit 1
fi

OUTPUT_BASE="$1"
shift  # 最初の引数を削除

# 各CTファイルを処理
for CT_FILE in "$@"; do
    # ファイル名から出力ディレクトリ名を生成
    BASENAME=$(basename "$CT_FILE" .nii.gz)
    OUTPUT_DIR="$OUTPUT_BASE/$BASENAME"
    
    echo "🔄 処理中: $CT_FILE → $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    
    # TotalSegmentator実行
    TotalSegmentator -i "$CT_FILE" -o "$OUTPUT_DIR" --task total --fast
    
    echo "✅ 完了: $BASENAME"
done

echo "🎉 バッチ処理完了"
