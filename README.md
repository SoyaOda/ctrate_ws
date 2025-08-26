# CT-RATE EAT/PAT 解析パイプライン

CT-RATE v2データセットから心臓周囲脂肪（EAT/PAT）を抽出し、2D投影画像を生成するパイプラインです。

## 実装済みスクリプト

### クイックテスト（動作確認）
```bash
./scripts/quick_test_totalseg.sh
```
- Docker環境とGPUの確認
- TotalSegmentatorイメージの自動ダウンロード
- ライセンス設定の対話的サポート
- 簡易テスト実行

### STEP3: TotalSegmentator心臓セグメント
```bash
./scripts/step3_totalsegmentator_test.sh
```
- 正しい公式Dockerイメージ: `wasserth/totalsegmentator:2.10.0`
- heartchambers_highresタスクで心臓構造をセグメント
- ライセンスキーの設定が必要（例: aca_DHDKRFJMHUX0PK）

### STEP4: BOA環境構築
```bash
./scripts/step4_boa_setup.sh
```
- Body-and-Organ-Analysis (BOA)のDocker環境を構築
- TotalSegmentatorと体組成解析を統合
- docker-compose-win.ymlを使用（Windows環境向け）

### STEP5: EAT/PAT抽出
```bash
python scripts/eat_pat_extraction.py <CT_PATH> <MYOCARDIUM_PATH> -o <OUTPUT_DIR>
```
- 心筋マスクから指定距離（5mm/10mm）の脂肪組織を抽出
- HU値範囲: -190 ～ -30
- 体積計算とマスク出力

### STEP6: 2D投影生成
```bash
python scripts/generate_2d_projections.py <CT_PATH> <FAT_MASK_PATH> -o <OUTPUT_DIR> --all-axes
```
- DRR（デジタル再構成X線画像）生成
- 脂肪マップの2D投影
- AP/PA/LAT/RLAT全軸対応

### 統合パイプライン
```bash
./scripts/run_pipeline.sh <CT_FILE_PATH> [OUTPUT_DIR]
```
全ステップを自動実行する統合スクリプト

## セットアップ

### 1. 依存パッケージのインストール
```bash
python3 -m venv env
source env/bin/activate
pip install nibabel numpy scipy matplotlib imageio
```

### 2. TotalSegmentatorライセンス設定
```bash
./scripts/step3_totalsegmentator_test.sh
# プロンプトでライセンスキーを入力
```

### 3. データ準備
CT-RATE v2データ（*_fixed/）を以下に配置:
```
~/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/
```

## 使用例

### 単一症例の処理
```bash
# 統合パイプラインで実行
./scripts/run_pipeline.sh ~/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz

# 個別ステップ実行
# 1. セグメンテーション（手動またはBOA経由）
# 2. EAT/PAT抽出
python scripts/eat_pat_extraction.py \
    data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz \
    outputs/ts_test/heart_myocardium.nii.gz \
    -o outputs/eat_pat

# 3. 2D投影生成
python scripts/generate_2d_projections.py \
    data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz \
    outputs/eat_pat/valid_1_a_1_pat_10mm.nii.gz \
    -o outputs/projections \
    --all-axes
```

## 出力ファイル

### EAT/PAT抽出結果
- `*_eat_5mm.nii.gz`: EATマスク（心筋から5mm）
- `*_pat_10mm.nii.gz`: PATマスク（心筋から10mm）
- `*_volumes.txt`: 体積測定結果

### 2D投影結果
- `*_drr_*.png`: DRR画像
- `*_fat_*.png`: 脂肪投影マップ
- `*_overlay_*.png`: オーバーレイ画像
- `*_viz.png`: 3画像並列可視化

## トラブルシューティング

### Dockerイメージが見つからないエラー
```bash
# 誤: totalsegmentator/totalsegmentator (存在しない)
# 正: wasserth/totalsegmentator:2.10.0

# 正しいイメージをダウンロード
docker pull wasserth/totalsegmentator:2.10.0
```

### GPU認識エラー
```bash
# WSL2でGPU確認
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

### TotalSegmentatorライセンスエラー
```bash
# ライセンス再設定（正しいイメージ名を使用）
docker run --rm -it --gpus all \
    -v ~/.totalsegmentator:/root/.totalsegmentator \
    wasserth/totalsegmentator:2.10.0 \
    totalseg_set_license -l YOUR_LICENSE_KEY
```

### BOAコンテナエラー
```bash
cd ~/ctrate_ws/tools/BOA
# ログ確認
docker compose logs
# 再起動
docker compose down
docker compose up -d
```

## 参考資料
- [CT-RATE Dataset](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [Body-and-Organ-Analysis](https://github.com/UMEssen/Body-and-Organ-Analysis)
- [VLM3D Challenge](https://vlm3dchallenge.com/)