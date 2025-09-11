#!/bin/bash

echo "🚀 完全最適化統合パイプライン テスト実行"
echo "========================================"

# デフォルトCTファイル
DEFAULT_CT="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"

# 引数から CT ファイルを取得（デフォルト使用可能）
CT_FILE=${1:-$DEFAULT_CT}

# CTファイル存在確認
if [ ! -f "$CT_FILE" ]; then
    echo "❌ エラー: CTファイルが見つかりません: $CT_FILE"
    echo "使用方法: $0 [CT_FILE_PATH]"
    echo "例: $0 /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    exit 1
fi

echo "📁 処理対象CT: $(basename $CT_FILE)"
echo "📊 ファイルサイズ: $(du -sh "$CT_FILE" | cut -f1)"
echo ""

# 永続化コンテナ確認
echo "🔍 永続化コンテナ確認..."
if ! docker ps | grep -q "totalseg-daemon"; then
    echo "❌ 永続化コンテナが動作していません。起動中..."
    python3 /home/soya/ctrate_ws/pipeline/optimize_docker_env.py --start-daemon
    
    # 起動確認
    sleep 5
    if ! docker ps | grep -q "totalseg-daemon"; then
        echo "❌ 永続化コンテナの起動に失敗しました"
        exit 1
    fi
fi

echo "✅ 永続化コンテナ確認完了"
echo ""

# 出力ディレクトリ準備
OUTPUT_DIR="/home/soya/ctrate_ws/pipeline/full_pipeline_test"
echo "📂 出力ディレクトリ: $OUTPUT_DIR"

# 既存結果のクリーンアップ
if [ -d "$OUTPUT_DIR" ]; then
    echo "🧹 既存結果をクリーンアップ中..."
    echo "odaSOYA6480" | sudo -S rm -rf "$OUTPUT_DIR" 2>/dev/null
fi

echo ""
echo "⏰ 完全最適化パイプライン実行開始: $(date)"
echo "========================================"

# 完全最適化パイプライン実行
python3 /home/soya/ctrate_ws/pipeline/full_optimized_pipeline.py \
    --ct-file "$CT_FILE" \
    --output-dir "$OUTPUT_DIR"

# 実行結果確認
EXIT_CODE=$?
echo ""
echo "========================================"
echo "⏰ 実行終了: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 完全最適化パイプライン成功!"
    
    # 結果ファイル一覧
    echo ""
    echo "📄 生成された結果ファイル:"
    find "$OUTPUT_DIR" -type f -name "*.json" -o -name "*.nii.gz" -o -name "*.png" | head -20
    
    # 最新の結果JSONを表示
    LATEST_RESULT=$(find "$OUTPUT_DIR" -name "full_pipeline_results_*.json" -type f -exec ls -t {} \; | head -1)
    if [ -f "$LATEST_RESULT" ]; then
        echo ""
        echo "📊 最新結果サマリ:"
        echo "=================="
        python3 -c "
import json
with open('$LATEST_RESULT', 'r') as f:
    data = json.load(f)
if data.get('success'):
    summary = data.get('summary', {})
    print(f'✅ 成功: {summary.get(\"total_time\", 0):.1f}秒')
    print(f'  - TotalSeg(ROI): {summary.get(\"step1_time\", 0):.1f}秒')
    print(f'  - EAT/PAT抽出: {summary.get(\"step2_time\", 0):.1f}秒')
    print(f'  - DeepDRR生成: {summary.get(\"step3_time\", 0):.1f}秒')
    print(f'  - マスク投影: {summary.get(\"step4_time\", 0):.1f}秒')
    print(f'  - 生成マスク: {summary.get(\"generated_masks\", 0)}個')
else:
    print(f'❌ 失敗: {data.get(\"failed_at\", \"unknown\")}')
"
    fi
    
else
    echo "❌ 完全最適化パイプライン失敗 (Exit Code: $EXIT_CODE)"
fi

echo ""
echo "📁 詳細結果: $OUTPUT_DIR"
echo "🔧 コンテナ状態: docker ps | grep totalseg-daemon"