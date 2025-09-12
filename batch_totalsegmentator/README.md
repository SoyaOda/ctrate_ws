# TotalSegmentator バッチ処理システム

CT-RATE-v2データセット全体に対してTotalSegmentatorを効率的に実行するシステムです。

## 🚀 クイックスタート

```bash
cd /home/soya/ctrate_ws/batch_totalsegmentator

# 全ファイルを処理
./run_batch.sh

# または1ファイルでテスト
./test_single.sh
```

## 📁 システム構成

```
batch_totalsegmentator/
├── batch_processor.py          # メインのバッチ処理スクリプト
├── create_dataset_config.py    # CT-RATE-v2用設定ファイル生成
├── generate_config.py          # 汎用設定ファイル生成  
├── run_batch.sh               # 実行用シェルスクリプト
├── test_single.sh             # テスト実行スクリプト
└── README.md                  # このファイル
```

## 💡 主な機能

- ✅ **効率的なバッチ処理**: モデルは初回のみダウンロード（約230MB）
- ✅ **自動スキップ**: 処理済みファイルを自動検出してスキップ
- ✅ **エラー復旧**: 失敗したファイルは自動的に再試行
- ✅ **進捗管理**: `progress.json`で処理状況を記録
- ✅ **メモリ最適化**: RAM不足対策のオプション設定済み
- ✅ **ディレクトリ構造維持**: 元のフォルダ構造を保持して出力

## 📊 データ構造

### 入力（Windows H:ドライブ）
```
H:\Chest-Xray-Create-Dataset\data\CT-RATE-v2\dataset\valid_fixed\
└── valid_1\
    └── valid_1_a\
        ├── valid_1_a_1.nii.gz
        └── valid_1_a_2.nii.gz
```

### 出力（同じH:ドライブ）
```
H:\Chest-Xray-Create-Dataset\data\CT-RATE-v2\dataset\TotalSegmentator_valid_fixed\
└── valid_1\
    └── valid_1_a\
        ├── valid_1_a_1\           # 各CTファイルごとのフォルダ
        │   ├── adrenal_gland_left.nii.gz
        │   ├── aorta.nii.gz
        │   └── ... (117個のマスクファイル)
        └── valid_1_a_2\
            └── ... (117個のマスクファイル)
```

## 🔧 詳細な使い方

### 1. 設定ファイルの生成

```bash
# CT-RATE-v2データセット全体の設定ファイルを生成
python3 create_dataset_config.py

# データセットの確認のみ（ファイル数の確認）
python3 create_dataset_config.py --check
```

### 2. バッチ処理の実行

```bash
# 基本実行（自動的に処理済みをスキップ）
python3 batch_processor.py \
    --config ct_rate_v2_config.json \
    --output /mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/TotalSegmentator_valid_fixed \
    --task total

# 最初から全て再処理
python3 batch_processor.py \
    --config ct_rate_v2_config.json \
    --output /mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/TotalSegmentator_valid_fixed \
    --task total \
    --no-resume

# レポートのみ生成
python3 batch_processor.py \
    --config ct_rate_v2_config.json \
    --output /mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/TotalSegmentator_valid_fixed \
    --report
```

## 📝 オプション説明

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--config, -c` | 設定ファイルパス | 必須 |
| `--output, -o` | 出力ディレクトリ | 必須 |
| `--task, -t` | TotalSegmentatorタスク | total |
| `--license-key, -l` | ライセンスキー | aca_DHDKRFJMHUX0PK |
| `--no-resume` | 最初から実行（スキップ無効） | False |
| `--report` | レポートのみ生成 | False |

### タスクの種類

- `total`: 全臓器セグメンテーション（117クラス）★推奨
- `tissue_types`: 組織タイプ（脂肪・筋肉など）
- `heartchambers_highres`: 心臓の詳細セグメンテーション
- `body`: 体領域のセグメンテーション

## 🔍 進捗の確認

```bash
# ログファイルの確認（別ターミナルで）
tail -f /mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/TotalSegmentator_valid_fixed/logs/batch_*.log

# 進捗状況の確認
cat /mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/TotalSegmentator_valid_fixed/progress.json | python3 -m json.tool | head -20

# 処理済みファイル数の確認
cat /mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/TotalSegmentator_valid_fixed/progress.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'完了: {len(d[\"completed\"])}, 失敗: {len(d[\"failed\"])}')"
```

## ⚠️ エラー対処

### メモリ不足エラー
```
RuntimeError: Background workers died... RAM was full...
```

**対策済み**: 
- スレッド数を1に制限（`--nr_thr_resamp 1 --nr_thr_saving 1`）
- Dockerのメモリ制限設定（16GB RAM, 32GB swap）

### 処理が止まった場合
1. `Ctrl+C`で中断
2. 同じコマンドを再実行（自動的に続きから再開）

### Dockerエラー
```bash
# Docker権限の確認
docker ps

# GPUの確認  
nvidia-smi
```

## 📈 処理時間の目安

- **初回（1ファイル目）**: 約7-10分（モデルダウンロード含む）
- **2ファイル目以降**: 各5-7分
- **全3039ファイル**: 約250-350時間（10-15日）

## 🔄 スキップ機能の仕組み

1. **実際のファイルをチェック**
   - 出力ディレクトリに100個以上のマスクファイルが存在 → スキップ
   
2. **エラーファイルは再試行**
   - `progress.json`の`failed`リストに記録
   - 次回実行時に自動的に再試行

3. **強制再処理**
   - `--no-resume`オプションで全て最初から処理

## 📋 出力ファイルリスト（totalタスクの場合）

各CTファイルごとに117個のマスクファイルが生成されます：

- 臓器: liver, spleen, kidney_left, kidney_right, stomach, pancreas...
- 血管: aorta, inferior_vena_cava, portal_vein, pulmonary_artery...
- 骨: rib_left_1-12, rib_right_1-12, vertebrae_C1-L5, hip_left, hip_right...
- 筋肉: iliopsoas_left, iliopsoas_right, autochthon_left, autochthon_right...
- その他: gallbladder, esophagus, trachea, thyroid_gland...

## 💾 必要なリソース

- **GPU**: NVIDIA GPU（CUDA対応）必須
- **メモリ**: 16GB以上推奨
- **ディスク**: 1CTあたり約100-200MB（全体で約300-600GB）

## 📞 トラブルシューティング

問題が発生した場合：
1. ログファイルを確認
2. `progress.json`の内容を確認
3. 失敗したファイルのパスを特定
4. 個別にテスト実行してエラーを確認

## 🎯 推奨ワークフロー

1. **まずテスト実行**
   ```bash
   ./test_single.sh
   ```

2. **問題なければ全体処理開始**
   ```bash
   ./run_batch.sh
   ```

3. **定期的に進捗確認**
   ```bash
   # 別ターミナルで
   watch -n 60 'cat /mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/TotalSegmentator_valid_fixed/progress.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"完了: {len(d[\"completed\"])}/3039\")"'
   ```

4. **エラーが多い場合は中断して調査**
   ```bash
   # Ctrl+Cで中断
   # レポート生成
   python3 batch_processor.py -c ct_rate_v2_config.json -o /mnt/h/Chest-Xray-Create-Dataset/data/CT-RATE-v2/dataset/TotalSegmentator_valid_fixed --report
   ```