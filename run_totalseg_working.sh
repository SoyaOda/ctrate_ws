#!/bin/bash
# TotalSegmentator動作確認スクリプト

echo "========================================="
echo "TotalSegmentator動作テスト"
echo "========================================="

# 変数設定
INPUT="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT="/home/soya/ctrate_ws/outputs/working_test"

# 出力ディレクトリを準備
echo "出力ディレクトリを準備..."
rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

# 実行
echo ""
echo "TotalSegmentatorを実行中..."
echo "入力: $INPUT"
echo "出力: $OUTPUT"
echo ""

# タイムアウトを設定して実行（5分）
timeout 300 docker run --rm --gpus all --ipc=host \
    -v /home/soya:/home/soya \
    wasserth/totalsegmentator:2.10.0 \
    bash -c "
        cd /tmp && \
        TotalSegmentator \
            -i '$INPUT' \
            -o /tmp/output \
            --task total \
            --device gpu \
            --fast && \
        cp -r /tmp/output/* '$OUTPUT/' 2>/dev/null
    "

# 結果確認
echo ""
echo "========================================="
echo "結果確認"
echo "========================================="

if [ -d "$OUTPUT" ]; then
    FILE_COUNT=$(find "$OUTPUT" -name "*.nii.gz" 2>/dev/null | wc -l)
    if [ "$FILE_COUNT" -gt 0 ]; then
        echo "✓ 成功: $FILE_COUNT 個のファイルが生成されました"
        echo ""
        echo "生成ファイル:"
        ls -la "$OUTPUT"/*.nii.gz 2>/dev/null | head -10
    else
        echo "✗ ファイルが生成されませんでした"
        
        # /tmpディレクトリを確認
        echo ""
        echo "一時ディレクトリを確認中..."
        docker run --rm -v /tmp:/tmp alpine ls -la /tmp/ | grep -E "output|totalseg|nnunet"
    fi
else
    echo "✗ 出力ディレクトリが作成されませんでした"
fi

echo ""
echo "========================================="
echo "完了"
echo "========================================="