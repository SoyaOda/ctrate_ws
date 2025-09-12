#!/bin/bash

# 統合パイプラインのテストスクリプト
# 事前に必要なセグメンテーション全てを含めて実行

set -e  # エラーで停止

# カラー出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}統合マスク生成パイプライン テスト実行${NC}"
echo -e "${GREEN}========================================${NC}"

# 入力ファイルと出力ディレクトリの設定
INPUT_CT="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT_DIR="/home/soya/ctrate_ws/mask_generation/test_output_$(date +%Y%m%d_%H%M%S)"

# 入力ファイルの確認
if [ ! -f "$INPUT_CT" ]; then
    echo -e "${RED}エラー: 入力CTファイルが見つかりません: $INPUT_CT${NC}"
    exit 1
fi

echo -e "${YELLOW}入力CT: $INPUT_CT${NC}"
echo -e "${YELLOW}出力先: $OUTPUT_DIR${NC}"
echo ""

# パイプライン実行
echo -e "${GREEN}パイプライン実行中...${NC}"
cd /home/soya/ctrate_ws

python3 mask_generation/integrated_mask_pipeline.py \
    "$INPUT_CT" \
    "$OUTPUT_DIR" \
    --verbose

# 結果の確認
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}パイプライン実行完了！${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # 時間測定結果を表示
    if [ -f "$OUTPUT_DIR/timing_summary.txt" ]; then
        echo ""
        echo -e "${YELLOW}実行時間サマリー:${NC}"
        cat "$OUTPUT_DIR/timing_summary.txt"
    fi
    
    # 統計情報を表示
    if [ -f "$OUTPUT_DIR/statistics/eat_pat_analysis.json" ]; then
        echo ""
        echo -e "${YELLOW}EAT+PAT解析結果:${NC}"
        python3 -c "
import json
with open('$OUTPUT_DIR/statistics/eat_pat_analysis.json') as f:
    stats = json.load(f)
    print(f\"  EAT+PAT volume: {stats['volumes_ml']['eat_pat']:.2f} ml\")
    print(f\"  Shell volume: {stats['volumes_ml']['shell']:.2f} ml\")
    print(f\"  Fat fraction in shell: {stats['ratios_percent']['fat_fraction_in_shell']:.1f}%\")
"
    fi
    
    # 生成されたファイルのリスト
    echo ""
    echo -e "${YELLOW}生成されたマスクファイル:${NC}"
    ls -lh "$OUTPUT_DIR/eat_pat_output/masks/" 2>/dev/null || echo "マスクファイルが見つかりません"
    
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}パイプライン実行エラー！${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}ログファイルを確認してください: $OUTPUT_DIR/*.log${NC}"
    exit 1
fi