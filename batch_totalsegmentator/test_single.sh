#!/bin/bash
# 単一ファイルでテスト実行するスクリプト

echo "TotalSegmentator テスト実行（1ファイルのみ）"
echo "================================================"

# テスト用の設定ファイルを作成
cat > test_config.json << EOF
{
  "description": "テスト用（1ファイルのみ）",
  "created_date": "$(date +'%Y-%m-%d %H:%M:%S')",
  "task": "total",
  "total_files": 1,
  "ct_files": [
    {
      "path": "/mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz",
      "name": "valid_1_valid_1_a_1",
      "relative_path": "valid_1/valid_1_a/valid_1_a_1.nii.gz"
    }
  ]
}
EOF

echo "テスト設定ファイルを作成しました: test_config.json"
echo ""
echo "出力先: /mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/TotalSegmentator_valid_fixed/valid_1/valid_1_a/valid_1_a_1/"
echo ""
echo "実行しますか? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    python3 batch_processor.py \
        --config test_config.json \
        --output /mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/TotalSegmentator_valid_fixed \
        --task total
else
    echo "中止しました"
fi