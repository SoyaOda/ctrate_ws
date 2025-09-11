#!/bin/bash
# 単一症例テストスクリプト
# valid_1_a_1.nii.gz を使用してパイプライン全体のテストを実行

set -e  # エラー時に停止

echo "🧪 CTスキャン処理パイプライン - 単一症例テスト"
echo "=================================================="

# 設定
CT_FILE="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/pipeline/test_outputs/single_case_test"
PIPELINE_SCRIPT="/home/soya/ctrate_ws/pipeline/ct_processing_pipeline.py"

# ファイル存在確認
if [ ! -f "$CT_FILE" ]; then
    echo "❌ エラー: CTファイルが見つかりません: $CT_FILE"
    exit 1
fi

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "❌ エラー: パイプラインスクリプトが見つかりません: $PIPELINE_SCRIPT"
    exit 1
fi

# 出力ディレクトリクリア（テスト前に）
if [ -d "$OUTPUT_DIR" ]; then
    echo "🧹 既存出力ディレクトリをクリア: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

echo "📁 入力ファイル: $CT_FILE"
echo "📂 出力ディレクトリ: $OUTPUT_DIR"
echo ""

# GPU状態確認
echo "🖥️ GPU状態確認:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | head -1
echo ""

# システム情報
echo "💻 システム情報:"
echo "  CPU: $(nproc) cores"
echo "  Memory: $(free -h | awk '/^Mem:/{print $2}')"
echo "  Disk Free: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# 依存関係チェック
echo "🔧 依存関係チェック:"
python3 -c "import nvidia_ml_py3 as nvml; print('✅ nvidia-ml-py3 OK')" 2>/dev/null || echo "⚠️  nvidia-ml-py3 なし (GPU監視無効)"
python3 -c "import psutil; print('✅ psutil OK')" 2>/dev/null || (echo "❌ psutil required" && exit 1)
docker --version > /dev/null && echo "✅ Docker OK" || (echo "❌ Docker required" && exit 1)
echo ""

# メイン実行
echo "🚀 パイプライン実行開始"
echo "時刻: $(date)"
START_TIME=$(date +%s)

# パイプライン実行
python3 "$PIPELINE_SCRIPT" "$CT_FILE" "$OUTPUT_DIR" --verbose

RESULT=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "⏱️ 実行完了"
echo "時刻: $(date)"
echo "実行時間: ${DURATION}秒 ($((DURATION / 60))分$((DURATION % 60))秒)"

if [ $RESULT -eq 0 ]; then
    echo "✅ テスト成功!"
    
    # 結果確認
    echo ""
    echo "📊 出力結果確認:"
    
    # Step 1結果
    if [ -f "$OUTPUT_DIR/step1_eat_pat/masks/eat_pat.nii.gz" ]; then
        echo "  ✅ Step 1 (EAT/PAT抽出) 完了"
        EAT_SIZE=$(stat -f%z "$OUTPUT_DIR/step1_eat_pat/masks/eat_pat.nii.gz" 2>/dev/null || stat -c%s "$OUTPUT_DIR/step1_eat_pat/masks/eat_pat.nii.gz" 2>/dev/null)
        echo "    EATマスクサイズ: $(($EAT_SIZE / 1024 / 1024))MB"
    else
        echo "  ❌ Step 1 (EAT/PAT抽出) 未完了"
    fi
    
    # Step 2結果
    DRR_COUNT=$(find "$OUTPUT_DIR/step2_drr" -name "*.png" 2>/dev/null | wc -l)
    if [ $DRR_COUNT -gt 0 ]; then
        echo "  ✅ Step 2 (DRR作成) 完了: ${DRR_COUNT}個のPNG"
    else
        echo "  ❌ Step 2 (DRR作成) 未完了"
    fi
    
    # Step 3結果
    PROJ_COUNT=$(find "$OUTPUT_DIR/step3_mask_projection" -name "*.png" 2>/dev/null | wc -l)
    if [ $PROJ_COUNT -gt 0 ]; then
        echo "  ✅ Step 3 (マスク投影) 完了: ${PROJ_COUNT}個のPNG"
    else
        echo "  ❌ Step 3 (マスク投影) 未完了"
    fi
    
    # リソースログ確認
    if [ -f "$OUTPUT_DIR/logs/resource_usage.json" ]; then
        echo "  ✅ リソースログ保存済み"
        LOG_SIZE=$(stat -f%z "$OUTPUT_DIR/logs/resource_usage.json" 2>/dev/null || stat -c%s "$OUTPUT_DIR/logs/resource_usage.json" 2>/dev/null)
        echo "    ログサイズ: $(($LOG_SIZE / 1024))KB"
    else
        echo "  ⚠️ リソースログ未保存"
    fi
    
    # 総出力サイズ
    TOTAL_SIZE=$(du -sk "$OUTPUT_DIR" | cut -f1)
    echo "  📦 総出力サイズ: $((TOTAL_SIZE / 1024))MB"
    
else
    echo "❌ テスト失敗 (終了コード: $RESULT)"
fi

echo ""
echo "🔗 結果確認方法:"
echo "  出力ディレクトリ: $OUTPUT_DIR"
echo "  リソースログ: $OUTPUT_DIR/logs/resource_usage.json"
echo "  分析実行: python3 pipeline/analyze_performance.py $OUTPUT_DIR/logs"

exit $RESULT