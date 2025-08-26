#!/bin/bash
# STEP3: TotalSegmentator heartchambers_highres実行スクリプト

echo "========================================="
echo "STEP3: TotalSegmentator (heartchambers_highres)"
echo "========================================="

# 入力と出力
INPUT="/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
OUTPUT="/home/soya/ctrate_ws/outputs/step3_heartchambers"

# 出力ディレクトリ作成
mkdir -p "$OUTPUT"

# メモリ状況確認
echo -e "\n現在のメモリ状況:"
free -h

# 実行
echo -e "\n実行中（heartchambers_highres タスク）..."
echo "入力: $INPUT"
echo "出力: $OUTPUT"
echo ""

# スレッド制限とメモリ最適化の環境変数設定
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# Dockerコンテナで実行（メモリ最適化版）
timeout 600 docker run --rm --gpus all --ipc=host \
    --shm-size=4g \
    -v /home/soya:/home/soya \
    -v /home/soya/.totalsegmentator:/root/.totalsegmentator \
    -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    wasserth/totalsegmentator:2.10.0 \
    bash -c "
        ulimit -n 4096 && \
        cd /tmp && \
        TotalSegmentator \
            -i '$INPUT' \
            -o /tmp/output \
            --task heartchambers_highres \
            --device gpu \
            --force_split && \
        cp -r /tmp/output/* '$OUTPUT/' 2>/dev/null
    "

# 結果確認
echo ""
echo "========================================="
echo "結果確認"
echo "========================================="

if [ -d "$OUTPUT" ] && [ "$(ls -A $OUTPUT 2>/dev/null)" ]; then
    FILE_COUNT=$(ls "$OUTPUT"/*.nii.gz 2>/dev/null | wc -l)
    echo "✓ 成功: $FILE_COUNT 個のファイルが生成されました"
    echo ""
    echo "生成ファイル:"
    ls -lh "$OUTPUT"/*.nii.gz | head -10
    
    # 心臓関連ファイル確認
    echo ""
    echo "心臓チャンバー関連ファイル:"
    ls "$OUTPUT"/*.nii.gz | grep -iE "atrium|ventricle|myocardium|appendage" || echo "見つかりませんでした"
else
    echo "✗ エラー: 出力ファイルが生成されませんでした"
fi

# メモリ状況（実行後）
echo -e "\nメモリ状況（実行後）:"
free -h