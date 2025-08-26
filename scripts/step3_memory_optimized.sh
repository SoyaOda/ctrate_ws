#!/bin/bash
# STEP3: メモリ最適化版TotalSegmentator実行スクリプト

echo "========================================="
echo "STEP3: TotalSegmentator (メモリ最適化版)"
echo "========================================="

# 入力と出力
INPUT="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT="/home/soya/ctrate_ws/outputs/step3_optimized"

# メモリ状況確認
echo "現在のメモリ状況:"
free -h
echo ""

# 出力ディレクトリ準備
rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

echo "実行中（メモリ最適化設定）..."
echo "入力: $INPUT"
echo "出力: $OUTPUT"
echo ""

# メモリ最適化実行
docker run --rm --gpus all --ipc=host \
  --shm-size=4g \
  -v /home/soya:/home/soya \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  wasserth/totalsegmentator:2.10.0 \
  bash -lc "
    ulimit -n 4096;
    export CUDA_VISIBLE_DEVICES=0;
    TotalSegmentator \
      -i '$INPUT' \
      -o /tmp/output \
      --task total \
      --device gpu \
      --fast \
      --body_seg \
      --robust_crop \
      --force_split && \
    mkdir -p '$OUTPUT' && \
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
        echo ""
        echo "OOMの確認："
        dmesg -T | grep -i "out of memory" | tail -5
    fi
else
    echo "✗ 出力ディレクトリが作成されませんでした"
fi

echo ""
echo "メモリ状況（実行後）:"
free -h