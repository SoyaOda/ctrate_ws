# DeepDRR Implementation for CT-RATE

このディレクトリには、CT-RATE v2データセットからDeepDRRを用いてDRR（Digitally Reconstructed Radiograph）を生成するための実装が含まれています。

## 問題と解決策

### CUDA 12.9 互換性問題
- **問題**: DeepDRR 1.1.3はCUDA 12.xで廃止されたtexture reference APIを使用しているため、CUDA 12.9環境ではコンパイルエラーが発生します
- **解決策**: `md_files/answer1.md` に基づき、CUDA 11.8環境でDeepDRRを使用します

## セットアップオプション

### オプション 1: Docker環境（推奨）

```bash
# Docker環境でDeepDRRを実行
cd /home/soya/ctrate_ws/deepdrr
./run_deepdrr_docker.sh
```

**特徴:**
- 環境の分離と再現性
- CUDA 11.8が事前設定済み
- システム環境への影響なし

### オプション 2: Conda環境

```bash
# conda環境をセットアップ
cd /home/soya/ctrate_ws/deepdrr
./setup_conda_env.sh

# 環境をアクティベート
conda activate deepdrr-cuda11

# テスト実行
python test_deepdrr_cuda11.py
```

**要件:**
- Miniconda/Mambaがインストール済み
- nvidia チャンネルアクセス可能

## ファイル構成

```
deepdrr/
├── README.md                           # このファイル
├── test_deepdrr_cuda11.py             # 単体テストスクリプト
├── environment-deepdrr-cuda11.yml     # conda環境定義
├── Dockerfile.cuda11                  # Docker環境定義
├── run_deepdrr_docker.sh              # Docker実行スクリプト
├── setup_conda_env.sh                 # conda環境セットアップ
└── outputs/                           # DRR出力ディレクトリ
```

## テストスクリプトの機能

`test_deepdrr_cuda11.py` は以下を実行します：

1. **環境テスト**: DeepDRRインポート、CUDA環境の確認
2. **DRR生成**: CT-RATE v2データからPA/LAT viewのDRRを生成
3. **可視化**: 生成したDRRの比較画像作成
4. **保存**: 16bit PNGとしてDRRを保存

### 入力データ
- **CTデータ**: `/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz`

### 出力データ
- **PA view**: `outputs/deepdrr_PA.png`
- **LAT view**: `outputs/deepdrr_LAT.png`  
- **比較画像**: `outputs/deepdrr_comparison.png`

## 実行手順

### Docker環境での実行

```bash
# 1. Docker環境セットアップ・実行
cd /home/soya/ctrate_ws/deepdrr
./run_deepdrr_docker.sh

# 2. コンテナ内でテスト実行（自動実行されます）
# python3 test_deepdrr_cuda11.py

# 3. 結果確認
ls -la outputs/
```

### Conda環境での実行

```bash
# 1. conda環境セットアップ
./setup_conda_env.sh

# 2. 環境アクティベート
conda activate deepdrr-cuda11

# 3. CUDA環境確認
nvcc --version  # 11.8であることを確認
python -c "import pycuda.driver as drv; drv.init(); print(f'GPU: {drv.Device(0).name()}')"

# 4. テスト実行
python test_deepdrr_cuda11.py

# 5. 結果確認
ls -la outputs/
```

## トラブルシューティング

### CUDA 12.9環境で直接実行した場合のエラー
```
identifier 'texture' is undefined
texture<T, 3, mode> tex
```
➡️ **解決策**: CUDA 11.8環境（Docker/conda）を使用してください

### Docker環境でGPUが認識されない場合
```bash
# nvidia-container-toolkitが必要
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### conda環境でCUDA 11.8が見つからない場合
```bash
# nvidia チャンネルの確認
conda config --show channels
conda config --add channels nvidia
```

## 参考資料

- `md_files/answer1.md`: CUDA互換性問題の詳細分析と解決策
- [DeepDRR GitHub](https://github.com/arcadelab/deepdrr)
- [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)

## 次のステップ

単体テストが成功したら、以下の拡張が可能です：

1. **バッチ処理**: 複数のCTデータを一括処理
2. **パラメータ調整**: 投影角度、検出器設定の最適化
3. **品質評価**: 生成されたDRRの品質評価
4. **マスク投影**: EAT Patマスクデータの重畳表示