#!/bin/bash
# TotalSegmentator バッチ処理実行スクリプト

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}TotalSegmentator バッチ処理システム${NC}"
echo -e "${GREEN}==================================================${NC}"

# デフォルト設定
CONFIG_FILE="ct_rate_v2_config.json"
OUTPUT_DIR="/mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/TotalSegmentator_valid_fixed"
TASK="total"

# 引数処理
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --task|-t)
            TASK="$2"
            shift 2
            ;;
        --help|-h)
            echo "使用方法: $0 [オプション]"
            echo "オプション:"
            echo "  -c, --config <file>   設定ファイル (デフォルト: ct_rate_v2_config.json)"
            echo "  -o, --output <dir>    出力ディレクトリ (デフォルト: ./outputs)"
            echo "  -t, --task <task>     タスク名 (デフォルト: total)"
            echo "  -h, --help            このヘルプを表示"
            exit 0
            ;;
        *)
            echo -e "${RED}不明なオプション: $1${NC}"
            exit 1
            ;;
    esac
done

# 設定ファイルの確認
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}設定ファイルが見つかりません: $CONFIG_FILE${NC}"
    echo "設定ファイルを生成しています..."
    python3 create_dataset_config.py -o "$CONFIG_FILE" -t "$TASK"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}設定ファイルの生成に失敗しました${NC}"
        exit 1
    fi
fi

# 出力ディレクトリの作成
mkdir -p "$OUTPUT_DIR"

# 実行確認
echo -e "${YELLOW}以下の設定で実行します:${NC}"
echo "  設定ファイル: $CONFIG_FILE"
echo "  出力ディレクトリ: $OUTPUT_DIR"
echo "  タスク: $TASK"
echo ""
echo -e "${YELLOW}続行しますか? (y/n)${NC}"
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "中止しました"
    exit 0
fi

# バッチ処理を実行
echo -e "${GREEN}バッチ処理を開始します...${NC}"
python3 batch_processor.py \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR" \
    --task "$TASK"

# 結果確認
if [ $? -eq 0 ]; then
    echo -e "${GREEN}==================================================${NC}"
    echo -e "${GREEN}バッチ処理が完了しました！${NC}"
    echo -e "${GREEN}結果: $OUTPUT_DIR${NC}"
    echo -e "${GREEN}==================================================${NC}"
else
    echo -e "${RED}エラーが発生しました${NC}"
    exit 1
fi