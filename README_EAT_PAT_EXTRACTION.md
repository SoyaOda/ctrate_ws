# EAT+PAT Extraction Pipeline with TotalSegmentator

## 概要

このパイプラインは、心臓CTスキャンから**EAT (Epicardial Adipose Tissue：心外膜脂肪組織)** と **PAT (Paracardial Adipose Tissue：心膜周囲脂肪組織)** を抽出・定量化するためのツールセットです。

TotalSegmentatorの出力を利用して、心臓周囲の脂肪組織を正確に同定し、可視化・統計解析を行います。

## 主要コンポーネント

### 1. Shell（心臓周囲関心領域）
- **定義**: 心臓表面から15mm膨張した領域（心臓自体を除く）
- **目的**: EAT+PATを含む解析対象領域を定義
- **実装**: EDT（ユークリッド距離変換）による等方的膨張

### 2. EAT+PAT
- **定義**: Shell内に存在する脂肪組織
- **EAT**: 心外膜と心膜の間の脂肪
- **PAT**: 心膜外側の脂肪
- **正常範囲**: 50-200ml

### 3. Visceral Fat（内臓脂肪）
- **定義**: TotalSegmentatorのtissue_typesタスクで検出される`torso_fat`
- **用途**: EAT+PATと全身内臓脂肪の比率計算

## インストール要件

```bash
# 必要なPythonパッケージ
pip install nibabel numpy scipy matplotlib

# TotalSegmentator（Docker版推奨）
docker pull wasserth/totalsegmentator:2.10.0
```

## 使用方法

### ステップ1: TotalSegmentatorでセグメンテーション実行

#### 1-1. 基本的な臓器セグメンテーション
```bash
# 心臓と臓器のセグメンテーション
TotalSegmentator -i input_ct.nii.gz -o output_dir
```

#### 1-2. 内臓脂肪セグメンテーション（必須）
**重要**: EAT+PAT抽出には`torso_fat.nii.gz`が必須です。tissue_typesタスクを実行してください。

```bash
# tissue_typesタスクでtorso_fat（内臓脂肪）を生成
TotalSegmentator -i input_ct.nii.gz -o output_dir --task tissue_types -l $LICENSE_KEY
```

生成されるファイル:
- `torso_fat.nii.gz` - 内臓脂肪マスク（必須）
- `subcutaneous_fat.nii.gz` - 皮下脂肪マスク
- `skeletal_muscle.nii.gz` - 骨格筋マスク

**注意事項**:
- tissue_typesタスクにはライセンスキーが必要です（研究用途は無料）
- ライセンスキーは`.env`ファイルまたは環境変数で設定
- **同じ出力ディレクトリ**に両方のタスクを実行してください
- tissue_typesは`--fast`オプションと非互換です

提供されているスクリプトを使用（推奨）：

```bash
# tissue_typesタスクの自動実行
./scripts/run_tissue_types.sh input_ct.nii.gz output_dir
```

このスクリプトは：
- 自動的にライセンスキーを.envから読み込み
- Dockerコンテナで実行
- 出力ファイルの確認を実施

### ステップ2: EAT+PAT抽出

```bash
python scripts/extract_eat_pat_integrated.py \
  input_ct.nii.gz \
  totalseg_output_dir \
  output_dir \
  --dilation-radius 15.0  # Shell半径（mm）
```

#### 出力構造
```
output_dir/
├── masks/
│   ├── heart.nii.gz        # 心臓マスク
│   ├── shell.nii.gz        # Shell（関心領域）マスク
│   ├── eat_pat.nii.gz      # EAT+PATマスク
│   └── visceral_fat.nii.gz # 内臓脂肪マスク
└── statistics/
    └── eat_pat_analysis.json # 統計情報
```

### ステップ3: 可視化

```bash
python scripts/visualize_eat_pat_integrated.py \
  input_ct.nii.gz \
  output_dir/masks \
  visualization_output_dir
```

#### 可視化出力
```
visualization_output_dir/
├── slices/                          # 個別スライス画像（約25枚）
│   ├── slice_000.png               # 腹部上部
│   ├── slice_082.png～slice_151.png # 心臓レベル
│   └── slice_188.png               # 腹部下部
├── mip_analysis.png                # 3方向MIP（最大値投影）
└── visceral_fat_distribution.png   # 内臓脂肪分布解析
```

## アルゴリズムの詳細

### EDT（ユークリッド距離変換）ベースの膨張

従来の反復膨張では非等方ボクセルで形状が歪む問題がありました。EDT実装により物理的に正確な距離計算が可能になりました。

```python
# EDTによる等方的膨張（推奨）
from scipy.ndimage import distance_transform_edt

dist_to_heart = distance_transform_edt(~heart_mask, sampling=spacing)
heart_dilated = dist_to_heart <= dilation_radius_mm
shell_mask = heart_dilated & ~heart_mask
```

### 改善効果
- **Shell体積**: 1266ml → 669ml（47.2%減少）
- **物理的精度**: mm単位で正確な15mm膨張
- **Z方向歪み**: 解消（0.75×0.75×1.5mmボクセルでも等方的）

## 統計情報

`eat_pat_analysis.json`に含まれる主要指標：

```json
{
  "volumes_ml": {
    "heart": 391.73,      // 心臓体積
    "shell": 669.20,      // Shell体積  
    "eat_pat": 83.96,     // EAT+PAT体積
    "visceral_fat": 418.46 // 内臓脂肪総量
  },
  "ratios_percent": {
    "fat_fraction_in_shell": 12.5,  // Shell内脂肪割合
    "eat_pat_to_visceral": 20.1,    // EAT+PAT/内臓脂肪比
    "eat_pat_to_heart": 21.4        // EAT+PAT/心臓体積比
  },
  "parameters": {
    "dilation_radius_mm": 15.0,
    "dilation_method": "EDT (Euclidean Distance Transform)"
  }
}
```

## 正常値範囲

| 指標 | 正常範囲 | 注意 |
|------|---------|------|
| EAT+PAT体積 | 50-200ml | >200mlで心血管リスク増加 |
| Shell内脂肪割合 | <30% | >30%で異常脂肪蓄積 |
| EAT+PAT/心臓比 | <25% | 心臓サイズに対する脂肪量 |

## スクリプト一覧

### 主要スクリプト
- `extract_eat_pat_integrated.py`: EAT+PAT抽出メイン処理（EDTベース）
- `visualize_eat_pat_integrated.py`: 統合可視化（4パネル表示）
- `run_tissue_types.sh`: TotalSegmentator tissue_typesタスク実行

### 補助スクリプト
- `extract_eat_pat_totalseg.py`: 基本的なEAT+PAT抽出（旧版）
- `visualize_visceral_fat.py`: 内臓脂肪単独の可視化
- `compare_edt_iteration.py`: EDT vs 反復膨張の比較

## トラブルシューティング

### 1. 内臓脂肪マスクが見つからない
```
[ERROR] Visceral fat mask (torso_fat) not found!
```
**解決策**: TotalSegmentatorをtissue_typesタスクで実行してください。

### 2. 心臓コンポーネントが見つからない
```
FileNotFoundError: No heart(-chambers) mask found
```
**解決策**: TotalSegmentatorの出力ディレクトリを確認し、心臓セグメンテーションが含まれていることを確認。

### 3. Shell体積が大きすぎる
**原因**: 反復膨張を使用している可能性
**解決策**: 最新のEDTベース実装を使用してください。

## 参考文献

1. Commandeur F, et al. "Cardiac adipose tissue: Key mediator in cardiovascular disease." *Eur Heart J* 2023.
2. Wasserthal J, et al. "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images." *Radiol Artif Intell* 2023.

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
TotalSegmentatorの使用には別途ライセンスが必要な場合があります。

## お問い合わせ

問題や質問がある場合は、GitHubのIssueページで報告してください。