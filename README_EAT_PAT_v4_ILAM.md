# EAT+PAT Extraction v4 with ILAM (Inferior Lung-Adjacency Mode)

## 概要

v4 ILAMは、心臓CTスキャンから**EAT (Epicardial Adipose Tissue)** と **PAT (Paracardial Adipose Tissue)** をより正確に抽出するための改良版アルゴリズムです。

### 主な改善点
- **ILAM (Inferior Lung-Adjacency Mode)**: 下部帯域（心横隔角付近）での肺接触ベースの脂肪検出
- **Portal veinとIVCの活用**: より正確な解剖学的境界の決定
- **解剖学的に妥当な抽出**: 横隔膜付近でのcardiophrenic fat padの適切な検出

## 必要な準備

### 1. セグメンテーションデータの準備

TotalSegmentatorで2つのタスクを実行する必要があります：

```bash
# 1. Total task（臓器セグメンテーション）
docker run --rm \
  -v /home/soya/ctrate_ws:/workspace \
  wasserth/totalsegmentator:2.10.0 \
  TotalSegmentator \
  -i /workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_2.nii.gz \
  -o /workspace/outputs/batch_results/valid_1_valid_1_a_2/segmentation_total \
  --task total

# 2. Tissue types task（脂肪セグメンテーション）- 別ディレクトリに出力
docker run --rm \
  -v /home/soya/ctrate_ws:/workspace \
  wasserth/totalsegmentator:2.10.0 \
  TotalSegmentator \
  -i /workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_2.nii.gz \
  -o /workspace/outputs/batch_results/valid_1_valid_1_a_2/segmentation_tissue \
  --task tissue_types -ta <LICENSE_KEY>
```

### 2. セグメンテーションデータの統合

権限エラーを回避するため、新しいディレクトリにマージします：

```bash
# マージディレクトリを作成
mkdir -p /home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation_merged

# tissue_typesの結果をコピー（torso_fat）
cp /home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation_tissue/*.nii.gz \
   /home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation_merged/

# totalタスクの結果をコピー（臓器）
cp /home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation_total/*.nii.gz \
   /home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation_merged/

# heart_chambersがある場合（オプション）
cp /home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation/heart_*.nii.gz \
   /home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation_merged/ 2>/dev/null
```

### 3. 必要なマスクの確認

以下のファイルが必要です：

**必須マスク：**
- `heart_myocardium.nii.gz`（または`heart.nii.gz`）
- `lung_left.nii.gz`と`lung_right.nii.gz`
- `torso_fat.nii.gz`
- `liver.nii.gz`
- `stomach.nii.gz` 
- `spleen.nii.gz`

**推奨マスク（精度向上）：**
- `inferior_vena_cava.nii.gz`
- `portal_vein_and_splenic_vein.nii.gz`
- `pulmonary_vein.nii.gz`
- `aorta.nii.gz`
- `pulmonary_artery.nii.gz`

確認コマンド：
```bash
ls /home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation_merged/ | \
  grep -E "heart|lung|torso_fat|liver|stomach|spleen|vena_cava|portal|vein"
```

## 実行方法

### 基本実行

```bash
python scripts/extract_eat_pat_improved_v4_ilam.py \
  <CT_FILE> \
  <SEGMENTATION_DIR> \
  <OUTPUT_DIR>
```

### 推奨実行（パラメータ調整付き）

```bash
python scripts/extract_eat_pat_improved_v4_ilam.py \
  /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_2.nii.gz \
  /home/soya/ctrate_ws/outputs/batch_results/valid_1_valid_1_a_2/segmentation_merged \
  /home/soya/ctrate_ws/outputs/eat_pat_v4_ilam_valid_1_a_2 \
  --inferior-band-mm 25.0 \
  --lung-contact-mm 1.5 \
  --abdomen-gap-mm 3.0 \
  --max-heart-dist-mm 30.0 \
  --verbose
```

### ILAMパラメータ

| パラメータ | デフォルト | 説明 | 調整指針 |
|-----------|----------|------|---------|
| `--inferior-band-mm` | 25.0 | 下部帯域の高さ[mm] | 肝臓が高い位置→20.0 |
| `--lung-contact-mm` (δ) | 1.5 | 肺接触判定の許容距離[mm] | 横隔膜が薄い→0.5-1.0 |
| `--abdomen-gap-mm` (γ) | 3.0 | 腹部臓器除外の膨張距離[mm] | 腹腔脂肪混入→5.0 |
| `--max-heart-dist-mm` (ρ) | 30.0 | 心臓からの最大距離[mm] | 縦隔脂肪多い→25.0 |
| `--replace-by-contact` | False | 下部で肺接触のみ使用（実験的） | - |

## 可視化

```bash
python scripts/visualize_eat_pat_integrated.py \
  <CT_FILE> \
  <OUTPUT_DIR>/masks \
  <OUTPUT_DIR>/visualization
```

例：
```bash
python scripts/visualize_eat_pat_integrated.py \
  /home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_2.nii.gz \
  /home/soya/ctrate_ws/outputs/eat_pat_v4_ilam_valid_1_a_2/masks \
  /home/soya/ctrate_ws/outputs/eat_pat_v4_ilam_valid_1_a_2/visualization
```

## 出力構造

```
<OUTPUT_DIR>/
├── masks/
│   ├── heart.nii.gz          # 心臓マスク（大血管近位部含む）
│   ├── shell.nii.gz          # Shell（15mm膨張領域）
│   ├── eat_pat.nii.gz        # EAT+PATマスク（ILAM改良版）
│   └── visceral_fat.nii.gz   # 内臓脂肪マスク
├── statistics/
│   └── eat_pat_analysis.json # 統計情報（ILAM設定含む）
└── visualization/
    ├── slices/                # 個別スライス画像
    ├── mip_analysis.png       # 3方向MIP
    └── visceral_fat_distribution.png
```

## 統計情報の確認

```bash
# 全統計情報
cat <OUTPUT_DIR>/statistics/eat_pat_analysis.json | jq .

# ILAMの効果を確認
cat <OUTPUT_DIR>/statistics/eat_pat_analysis.json | jq .parameters.ilam_settings
```

出力例：
```json
{
  "inferior_band_mm": 25.0,
  "lung_contact_mm": 1.5,
  "abdomen_gap_mm": 3.0,
  "max_heart_dist_mm": 30.0,
  "mode": "add",
  "z_range": [125, 162],
  "added_voxels": 5254,
  "ilam_voxels": 8492
}
```

## トラブルシューティング

### 1. 必須マスクが見つからない

```
[ERROR] Required masks missing: ['lungs', 'liver', 'stomach', 'spleen']
```

**解決策：**
1. TotalSegmentatorのtotalタスクを実行
2. segmentation_mergedディレクトリを作成して統合

### 2. Permission denied エラー

```
cp: cannot create regular file: Permission denied
```

**解決策：**
```bash
# 新しいディレクトリにコピー
mkdir -p ~/ctrate_ws/outputs/merged_segmentation
cp <SOURCE>/*.nii.gz ~/ctrate_ws/outputs/merged_segmentation/
# このディレクトリを使用
```

### 3. IVC/Portal veinが検出されない

```
WARNING: IVC/PV/Portal not found, using heart bottom
```

**解決策：**
1. totalタスクの出力を確認：
```bash
ls <SEGMENTATION_DIR>/*vena*.nii.gz
ls <SEGMENTATION_DIR>/*portal*.nii.gz
```

2. ファイルが存在する場合、segmentation_mergedにコピー：
```bash
cp <SEGMENTATION_DIR>/inferior_vena_cava.nii.gz <MERGED_DIR>/
cp <SEGMENTATION_DIR>/portal_vein_and_splenic_vein.nii.gz <MERGED_DIR>/
```

### 4. ILAM効果が大きすぎる

```
⚠ ILAM added >50% volume - consider adjusting parameters
```

**解決策：**
- `--inferior-band-mm`を20.0に減らす
- `--max-heart-dist-mm`を25.0に減らす
- `--abdomen-gap-mm`を5.0に増やす

## アルゴリズムの詳細

### ILAMの動作原理

下部帯域（z_min から inferior_band_mm）において：

```
C2 = torso_fat ∩ (~lungs) ∩ touch(torso_fat, lungs, δ) 
     ∩ (dist_heart ≤ ρ) ∩ ~(dilate(liver|spleen|stomach, γ))

最終EAT+PAT = Shell_based_EAT_PAT ∪ C2
```

### 改善効果の例

- **Shell-based EAT+PAT**: 263,548 voxels
- **ILAM追加分**: 5,254 voxels
- **最終EAT+PAT**: 268,802 voxels（94.21 ml）

## 正常値範囲

| 指標 | 正常範囲 | v4での典型値 |
|-----|---------|------------|
| EAT+PAT体積 | 50-200 ml | 94.21 ml |
| Shell内脂肪割合 | <30% | 20.3% |
| ILAMによる追加 | - | 2-10% |

## バッチ処理例

複数のCTデータを処理する場合：

```bash
#!/bin/bash
# batch_process_v4.sh

BASE_DIR="/home/soya/ctrate_ws"
DATA_DIR="$BASE_DIR/data/CT-RATE-v2/dataset/valid_fixed"

for patient_dir in $DATA_DIR/valid_*/valid_*_*; do
    patient_id=$(basename $patient_dir)
    
    # セグメンテーションディレクトリを準備
    SEG_DIR="$BASE_DIR/outputs/batch_results/${patient_id}/segmentation_merged"
    
    if [ -d "$SEG_DIR" ]; then
        echo "Processing $patient_id..."
        
        # CT画像を探す
        CT_FILE=$(find $patient_dir -name "*.nii.gz" | head -1)
        
        # v4 ILAM実行
        python scripts/extract_eat_pat_improved_v4_ilam.py \
            "$CT_FILE" \
            "$SEG_DIR" \
            "$BASE_DIR/outputs/eat_pat_v4_ilam_${patient_id}" \
            --inferior-band-mm 25.0 \
            --lung-contact-mm 1.5
    fi
done
```

## 参考文献

- lung_ROI_2.md: ILAMアルゴリズムの理論的根拠
- Radiopaedia: 縦隔と心膜の解剖学
- TotalSegmentator論文: セグメンテーション精度

## ライセンス

MITライセンス（研究・教育目的）