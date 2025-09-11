# CTスキャン処理パイプライン

CTスキャンから脂肪抽出、DRR作成、マスク投影までを統合した処理パイプラインです。数万例の大規模処理に対応するため、リソース監視とパフォーマンス分析機能を内蔵しています。

## 🎯 機能概要

### 処理ステップ
1. **Step 1: PAT+EAT抽出** - 心臓周囲脂肪組織の抽出
2. **Step 2: DRR作成** - DeepDRRによるX線画像生成  
3. **Step 3: マスク投影** - 脂肪マスクの2D投影画像作成

### 主要機能
- 📊 **リアルタイムリソース監視** (CPU、GPU メモリ、処理時間)
- 🔄 **バッチ処理対応** (数万例の自動処理)
- 📈 **パフォーマンス分析** (スケーラビリティ評価)
- 🛡️ **エラーハンドリング** (堅牢な例外処理)

## 🚀 クイックスタート

### 単一症例のテスト
```bash
# テスト実行
./pipeline/test_single_case.sh

# 結果確認
ls pipeline/test_outputs/single_case_test/
```

### 単一症例の手動実行
```bash
python3 pipeline/ct_processing_pipeline.py \\
    data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz \\
    outputs/pipeline_test \\
    --verbose
```

### バッチ処理
```bash
# ドライラン（実際の処理なし、ファイル確認のみ）
python3 pipeline/batch_process_ct.py \\
    data/CT-RATE-v2/dataset/valid_fixed/ \\
    outputs/batch_results/ \\
    --dry-run --verbose

# 実際のバッチ処理
python3 pipeline/batch_process_ct.py \\
    data/CT-RATE-v2/dataset/valid_fixed/ \\
    outputs/batch_results/ \\
    --max-workers 1 --verbose
```

### パフォーマンス分析
```bash
python3 pipeline/analyze_performance.py \\
    outputs/batch_results/logs/ \\
    --output-dir analysis_results/
```

## 📁 ディレクトリ構造

```
pipeline/
├── ct_processing_pipeline.py    # メインパイプライン
├── batch_process_ct.py          # バッチ処理システム  
├── analyze_performance.py       # パフォーマンス分析
├── test_single_case.sh          # 単一症例テスト
└── README.md                    # このファイル

# 出力構造例
outputs/
└── case_name/
    ├── step1_eat_pat/           # PAT+EAT抽出結果
    │   ├── masks/
    │   │   └── eat_pat.nii.gz
    │   └── visualization/
    ├── step2_drr/               # DRR生成結果
    │   ├── drr_pa.png
    │   └── drr_lat.png
    ├── step3_mask_projection/   # マスク投影結果
    │   ├── mask_projection_pa.png
    │   └── thickness_map.png
    └── logs/
        └── resource_usage.json  # リソースログ
```

## ⚙️ 設定と前提条件

### システム要件
- **OS**: Linux (WSL2対応)
- **GPU**: NVIDIA GPU (CUDA 11.8+)
- **Memory**: 16GB以上推奨
- **Storage**: 症例あたり ~500MB の空き容量

### 必要なソフトウェア
- Docker (GPU サポート有効)
- Python 3.8+
- 必要なPythonパッケージ:
  ```bash
  pip install nvidia-ml-py3 psutil matplotlib numpy
  ```

### Docker Images
使用するDockerイメージが利用可能であることを確認してください:
- `wasserth/totalsegmentator:2.10.0`
- `deepdrr-japan-mirror:latest`

## 📊 リソース監視機能

### 監視項目
- **CPU使用率** - リアルタイム監視
- **メモリ使用量** - システム全体とプロセス別
- **GPU情報** - VRAM使用量、GPU使用率
- **処理時間** - ステップ別とトータル時間

### ログ形式
```json
{
  "system_info": {
    "cpu_count": 8,
    "memory_total_gb": 32.0,
    "gpu_count": 1,
    "gpu_info": [...]
  },
  "resource_logs": [
    {
      "timestamp": 1234567890.123,
      "step": "step1",
      "stage": "start",
      "cpu_percent": 45.2,
      "memory_percent": 68.1,
      "gpu_usage": [...]
    }
  ]
}
```

## 🔧 トラブルシューティング

### よくある問題

#### 1. GPU メモリ不足
```bash
# 現在のGPU使用量確認
nvidia-smi

# 解決策: メモリ使用量の監視と最適化
python3 pipeline/analyze_performance.py logs/
```

#### 2. Docker権限エラー
```bash
# Dockerグループに追加
sudo usermod -aG docker $USER
# 再ログインが必要
```

#### 3. ディスク容量不足
```bash
# 容量確認
df -h
# 古いファイル削除
find outputs/ -name "*.nii.gz" -mtime +30 -delete
```

### デバッグモード
```bash
# 詳細ログでパイプライン実行
python3 pipeline/ct_processing_pipeline.py \\
    input.nii.gz output_dir/ --verbose

# 各ステップの個別実行も可能
python3 scripts/extract_eat_pat_improved_v5_4.py --help
```

## 📈 パフォーマンス最適化

### 推奨設定
- **バッチサイズ**: 通常は `--max-workers 1` (GPU制約)
- **並列処理**: 複数GPU環境でのみ並列数を増加
- **メモリ管理**: GPU VRAM 80%未満を維持

### スケーラビリティ評価
```bash
# 小規模テスト (10例)
python3 pipeline/batch_process_ct.py test_data/ results/ --max-workers 1

# パフォーマンス分析
python3 pipeline/analyze_performance.py results/logs/

# スケール推定の確認
cat analysis_results/performance_summary.txt
```

## 🧪 テスト

### 単体テスト
```bash
# パイプライン全体テスト
./pipeline/test_single_case.sh

# ステップ別テスト
python3 scripts/extract_eat_pat_improved_v5_4.py --help
```

### バッチ処理テスト
```bash
# ドライラン (安全)
python3 pipeline/batch_process_ct.py test_data/ results/ --dry-run

# 小規模実行 (3-5症例)
python3 pipeline/batch_process_ct.py small_test/ results/ --verbose
```

## 📞 サポート

### エラー発生時の情報収集
1. **リソースログ**: `outputs/*/logs/resource_usage.json`
2. **バッチレポート**: `outputs/batch_processing_report.json`
3. **システム情報**: `nvidia-smi`, `free -h`, `df -h`

### 改善提案
パフォーマンス分析結果に基づく最適化提案が自動生成されます:
```bash
python3 pipeline/analyze_performance.py logs/ 
cat analysis_results/performance_summary.txt
```

---

## 🎯 数万例処理への道筋

1. **ベンチマーク実行** - `test_single_case.sh`
2. **小規模バッチ** - 100例程度でテスト
3. **パフォーマンス分析** - ボトルネック特定
4. **最適化実装** - 分析結果に基づく改善
5. **大規模展開** - 段階的にスケールアップ

詳細は `analyze_performance.py` の実行結果を参照してください。