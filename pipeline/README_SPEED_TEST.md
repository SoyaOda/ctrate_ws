# TotalSegmentator 高速化テストガイド

TotalSegmentator（total + tissue_types）の高速化手法をテスト・評価するための包括的なテストスイートです。

## 🎯 テスト目的

1. **処理時間短縮**: 現在10-15分/例 → 目標3-5分/例
2. **品質維持**: 高速化によるマスク品質劣化の評価
3. **大規模処理実現可能性**: 数万例処理の時間予測

## 🚀 高速化手法

### A. TotalSegmentator最適化
- **--fast**: 標準1.5mm → 3mmの軽量モデル（total taskのみ）
- **--roi_subset**: EAT/PAT抽出に必要な臓器のみセグメンテーション
- **連続処理**: total + tissue_types を同一コンテナ内で実行

### B. Docker環境最適化
- **事前ビルド済みイメージ**: モデル重み事前ダウンロード
- **永続化コンテナ**: デーモン化によるオーバーヘッド削減

## 📊 テストデータ

以下3つのCT-RATE-v2データを使用：

| ファイル | サイズ | 説明 |
|---------|--------|------|
| valid_1_a_1.nii.gz | 336MB | 大サイズCT |
| valid_1_a_2.nii.gz | 151MB | 中サイズCT |
| valid_2_a_1.nii.gz | 75MB | 小サイズCT |

## 🔧 実行手順

### 1. 基本的な高速化テスト

```bash
# 単一症例テスト
./pipeline/run_totalseg_speed_test.sh --single /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz

# 3症例バッチテスト  
./pipeline/run_totalseg_speed_test.sh --batch
```

### 2. Docker環境最適化テスト

```bash
# 最適化イメージビルド
python3 pipeline/optimize_docker_env.py --build-image

# 永続化コンテナ起動
python3 pipeline/optimize_docker_env.py --start-daemon

# 環境ベンチマーク
python3 pipeline/optimize_docker_env.py --benchmark /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz
```

### 3. 結果分析・レポート生成

```bash
# 包括的分析レポート生成
python3 pipeline/analyze_speed_test_results.py --results-dir pipeline/totalseg_speed_test_results --output-dir pipeline/analysis_reports
```

## 📈 品質評価指標

### Dice係数による品質評価
- **Excellent**: ≥0.95
- **Very Good**: ≥0.90  
- **Good**: ≥0.85
- **Acceptable**: ≥0.80
- **Poor**: <0.80

### 重要マスク
- `heart_myocardium.nii.gz`: 心筋マスク（EAT抽出の基準）
- `torso_fat.nii.gz`: 体幹脂肪マスク（PAT抽出の材料）
- `lung_left.nii.gz`, `lung_right.nii.gz`: 肺マスク（境界判定用）

## 📊 期待される結果

### パフォーマンス目標
| 手法 | 処理時間目標 | 高速化率 |
|------|-------------|---------|
| 通常版 | 10-15分/例 | 1.0x |
| --fast + --roi_subset | 3-7分/例 | 2-3x |
| + Docker最適化 | 2-5分/例 | 3-5x |

### 品質要件
- 重要マスクのDice係数 ≥0.85
- EAT/PAT抽出に影響するマスクの品質維持

## 🔍 トラブルシューティング

### よくある問題

1. **tissue_typesでライセンスエラー**
   ```bash
   # ライセンスキー指定
   python3 pipeline/totalseg_speed_test.py --license-key YOUR_LICENSE_KEY
   ```

2. **GPU メモリ不足**
   ```bash
   # Docker実行前にGPUメモリクリア
   docker stop $(docker ps -q)
   nvidia-smi
   ```

3. **--fastが効かない**
   - tissue_typesタスクでは--fast使用不可（仕様）
   - totalタスクでのみ有効

### ログ確認
```bash
# 実行ログ確認
tail -f pipeline/totalseg_speed_test_results/progress_*.json

# GPU使用状況監視
watch -n 1 nvidia-smi
```

## 📋 出力ファイル構造

```
pipeline/totalseg_speed_test_results/
├── batch_test_results.json          # バッチテスト結果
├── progress_valid_1_a_1.json        # 症例別進捗
├── valid_1_a_1/                     # 症例別結果
│   ├── baseline/                    # 通常版結果
│   │   ├── total/                   # totalタスク出力
│   │   └── tissue_types/            # tissue_typesタスク出力
│   └── fast_optimized/              # 高速化版結果
│       ├── total/
│       └── tissue_types/
└── ...

pipeline/analysis_reports/
├── comprehensive_analysis_report.json  # 包括分析結果
├── speed_optimization_summary.txt      # テキストサマリ
├── performance_comparison.png          # パフォーマンス比較グラフ
├── quality_comparison.png              # 品質比較グラフ
└── scalability_prediction.png          # スケーラビリティ予測
```

## 🎯 数万例処理への展開

分析結果を基に：

1. **高速化効果確認**: 目標3-5x高速化の達成確認
2. **品質評価**: Dice係数≥0.85の品質維持確認  
3. **時間予測**: 3万例処理時間の現実性評価
4. **最適化提案**: Docker環境・並列処理等の追加改善案

成功すれば、数万例処理を数週間〜数ヶ月で完了可能な高速化パイプラインが実現できます。