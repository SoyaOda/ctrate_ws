# 推奨コマンド一覧

## 環境セットアップ
```bash
# Python仮想環境作成・有効化
python3 -m venv env
source env/bin/activate

# 依存パッケージインストール
pip install nibabel numpy scipy matplotlib imageio huggingface_hub datasets
```

## Docker関連
```bash
# GPU確認
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi

# TotalSegmentatorイメージダウンロード
docker pull wasserth/totalsegmentator:2.10.0
```

## パイプライン実行コマンド
```bash
# クイックテスト（動作確認）
./scripts/quick_test_totalseg.sh

# STEP3: TotalSegmentator心臓セグメント
./scripts/step3_totalsegmentator_test.sh

# STEP4: BOA環境構築
./scripts/step4_boa_setup.sh

# STEP5: EAT/PAT抽出
python scripts/eat_pat_extraction.py <CT_PATH> <MYOCARDIUM_PATH> -o <OUTPUT_DIR>

# STEP6: 2D投影生成
python scripts/generate_2d_projections.py <CT_PATH> <FAT_MASK_PATH> -o <OUTPUT_DIR> --all-axes

# 統合パイプライン
./scripts/run_pipeline.sh <CT_FILE_PATH> [OUTPUT_DIR]
```

## TotalSegmentatorライセンス設定
```bash
# ライセンスキー設定
docker run --rm -it --gpus all \
    -v ~/.totalsegmentator:/root/.totalsegmentator \
    wasserth/totalsegmentator:2.10.0 \
    totalseg_set_license -l YOUR_LICENSE_KEY
```

## トラブルシューティング
```bash
# Docker関連
docker compose logs
docker compose down && docker compose up -d

# GPU確認
nvidia-smi

# ディスク容量確認
df -h
```