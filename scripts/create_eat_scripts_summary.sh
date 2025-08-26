#!/bin/bash
# EATセグメンテーション関連スクリプトをまとめたドキュメント作成

OUTPUT_FILE="/home/soya/ctrate_ws/EAT_SCRIPTS_SUMMARY.md"

cat > "$OUTPUT_FILE" << 'EOF'
# EATセグメンテーション関連スクリプト一覧

生成日時: $(date +"%Y-%m-%d %H:%M:%S")

## 1. コアスクリプト（EAT抽出）

### 1.1 extract_eat_with_totalseg.py
**パス**: `/home/soya/ctrate_ws/scripts/extract_eat_with_totalseg.py`  
**説明**: TotalSegmentatorの心臓マスクとEDTを使用したEAT抽出のメインロジック  
**重要修正**: 心臓全体の外側のみシェル生成（`outside = ~heart`）

```python
EOF

cat /home/soya/ctrate_ws/scripts/extract_eat_with_totalseg.py >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << 'EOF'
```

---

### 1.2 extract_eat_with_totalseg_docker.sh
**パス**: `/home/soya/ctrate_ws/scripts/extract_eat_with_totalseg_docker.sh`  
**説明**: Docker内でEAT抽出を実行（nibabel等の依存関係対応）  
**重要修正**: 心臓全体の外側のみシェル生成

```bash
EOF

cat /home/soya/ctrate_ws/scripts/extract_eat_with_totalseg_docker.sh >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << 'EOF'
```

---

## 2. HARTA関連スクリプト

### 2.1 harta_masks_to_nifti_preserve_geom.py
**パス**: `/home/soya/ctrate_ws/scripts/harta_masks_to_nifti_preserve_geom.py`  
**説明**: HARTA PNG出力から3D NIfTIへの変換  
**重要修正**: load_mask_from_png関数（RGBチャンネル使用）

```python
EOF

cat /home/soya/ctrate_ws/scripts/harta_masks_to_nifti_preserve_geom.py >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << 'EOF'
```

---

### 2.2 run_harta_auto.py
**パス**: `/home/soya/ctrate_ws/scripts/run_harta_auto.py`  
**説明**: HARTA自動セグメンテーション実行

```python
EOF

cat /home/soya/ctrate_ws/scripts/run_harta_auto.py >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << 'EOF'
```

---

### 2.3 nifti_to_dicom_preserve_geom.py
**パス**: `/home/soya/ctrate_ws/scripts/nifti_to_dicom_preserve_geom.py`  
**説明**: NIfTI→DICOM変換（HARTAの入力用）

```python
EOF

cat /home/soya/ctrate_ws/scripts/nifti_to_dicom_preserve_geom.py >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << 'EOF'
```

---

## 3. 可視化スクリプト

### 3.1 visualize_eat_improved.sh
**パス**: `/home/soya/ctrate_ws/scripts/visualize_eat_improved.sh`  
**説明**: 心臓、シェル、EATのPNG生成（改善版）  
**特徴**: アルファ合成と輪郭表示  
**重要修正**: 心臓全体の外側のみシェル生成

```bash
EOF

cat /home/soya/ctrate_ws/scripts/visualize_eat_improved.sh >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << 'EOF'
```

---

### 3.2 visualize_eat_docker.sh
**パス**: `/home/soya/ctrate_ws/scripts/visualize_eat_docker.sh`  
**説明**: 基本的な可視化（旧版）

```bash
EOF

if [ -f /home/soya/ctrate_ws/scripts/visualize_eat_docker.sh ]; then
    cat /home/soya/ctrate_ws/scripts/visualize_eat_docker.sh >> "$OUTPUT_FILE"
else
    echo "# ファイルが見つかりませんでした" >> "$OUTPUT_FILE"
fi

cat >> "$OUTPUT_FILE" << 'EOF'
```

---

## 4. 設定ファイル

### 4.1 .env
**パス**: `/home/soya/ctrate_ws/.env`  
**説明**: TotalSegmentatorライセンスキー等の設定

```bash
EOF

cat /home/soya/ctrate_ws/.env >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << 'EOF'
```

---

## 5. バッチ処理スクリプト

### 5.1 batch_process_all.sh
**パス**: `/home/soya/ctrate_ws/scripts/batch_process_all.sh`  
**説明**: 全体のパイプライン実行（存在する場合）

```bash
EOF

if [ -f /home/soya/ctrate_ws/scripts/batch_process_all.sh ]; then
    cat /home/soya/ctrate_ws/scripts/batch_process_all.sh >> "$OUTPUT_FILE"
else
    echo "# ファイルが見つかりませんでした" >> "$OUTPUT_FILE"
fi

cat >> "$OUTPUT_FILE" << 'EOF'
```

---

## 主要なアルゴリズム

### EAT計算式
```
EAT = HARTA_fat ∧ HU(-190,-30) ∧ Shell(0-6mm) ∧ ¬Myocardium
```

### シェル生成（修正版）
```python
# 重要: 心臓全体の外側のみを計算領域にする
outside = ~heart  # heart = myocardium + all chambers
dist_mm = distance_transform_edt(outside, sampling=spacing)
shell_eat = (dist_mm > 0) & (dist_mm <= shell_thickness_mm)
```

### 主な修正履歴
1. **PNG読み込み修正**: アルファチャンネル → RGBチャンネル
2. **シェル生成修正**: `~myo` → `~heart`（内側シェル除去）
3. **座標系保持**: NIfTI ↔ DICOM変換でaffine行列保持

### 現在の結果
- EAT体積: 約1.93ml（期待値50-200mlより低い）
- シェルボクセル数: 108,931（修正後）
- 心臓ボクセル数: 445,484

EOF

echo "==================================="
echo "EATスクリプトまとめを作成しました："
echo "$OUTPUT_FILE"
echo "==================================="
echo "ファイルサイズ: $(du -h $OUTPUT_FILE | cut -f1)"
echo "行数: $(wc -l < $OUTPUT_FILE) 行"