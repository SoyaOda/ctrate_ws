#!/bin/bash
# TotalSegmentator 高速化テスト実行スクリプト

set -e  # エラー時に停止

echo "🚀 TotalSegmentator 高速化テスト"
echo "======================================"

# 設定
PIPELINE_DIR="/home/soya/ctrate_ws/pipeline"
TEST_SCRIPT="$PIPELINE_DIR/totalseg_speed_test.py"
OUTPUT_DIR="$PIPELINE_DIR/totalseg_speed_test_results"
LICENSE_KEY=""  # 必要に応じて設定

# 引数確認
if [ "$#" -eq 0 ]; then
    echo "使用方法:"
    echo "  単一症例テスト: $0 --single <CT_FILE_PATH>"
    echo "  バッチテスト:   $0 --batch"
    echo ""
    echo "例:"
    echo "  $0 --single /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    echo "  $0 --batch"
    exit 1
elif [ "$1" == "--single" ]; then
    if [ "$#" -ne 2 ]; then
        echo "❌ エラー: --single にはCTファイルパスが必要です"
        echo "使用方法: $0 --single <CT_FILE_PATH>"
        exit 1
    fi
    echo "🧪 単一症例テスト: $2"
    python3 "$TEST_SCRIPT" --single-test "$2" --output-dir "$OUTPUT_DIR" --license-key "$LICENSE_KEY"
elif [ "$1" == "--batch" ]; then
    echo "🔄 バッチテスト: 3症例"
    python3 "$TEST_SCRIPT" --output-dir "$OUTPUT_DIR" --license-key "$LICENSE_KEY"
else
    echo "❌ エラー: 不明なオプション '$1'"
    echo "使用方法:"
    echo "  単一症例テスト: $0 --single <CT_FILE_PATH>"
    echo "  バッチテスト:   $0 --batch"
    exit 1
fi

echo "✅ テスト完了"
echo "📊 結果: $OUTPUT_DIR"

# 結果サマリ表示
if [ -f "$OUTPUT_DIR/batch_test_results.json" ]; then
    echo ""
    echo "📈 結果サマリ:"
    python3 -c "
import json
with open('$OUTPUT_DIR/batch_test_results.json', 'r') as f:
    results = json.load(f)
summary = results.get('summary', {})
print(f'  総症例数: {summary.get(\"total_cases\", 0)}')
print(f'  成功症例: {summary.get(\"successful_cases\", 0)}')
print(f'  平均高速化: {summary.get(\"average_speedup\", 0):.2f}x')
print(f'  平均Dice係数: {summary.get(\"average_dice_score\", 0):.3f}')
print(f'  最低Dice係数: {summary.get(\"min_dice_score\", 0):.3f}')
"
fi