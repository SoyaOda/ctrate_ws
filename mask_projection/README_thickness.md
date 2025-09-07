# DeepDRR マスク投射 厚み推定版

## 概要

`mask_deepdrr_projection_thickness.py` は、gpt5pro5.mdの推奨方法Aを実装した厚み推定機能付きマスク投射スクリプトです。

## 主要機能

### 1. Beer-Lambert逆変換 + 体積一致自己較正
- **物理的厚み推定**: ピクセル値を物理的な厚み[mm]に正確変換
- **μ_eff自己較正**: 体積保存制約による実効線減弱係数の自動推定
- **体積一致保証**: ∑T·p² = V_mask(3D) による厳密な体積整合性

### 2. 高解像度維持
- **完全解像度保持**: 3070×2500ピクセルの高解像度を維持
- **位置完全整合**: 元のDeepDRR投影と同一幾何による精密位置合わせ

### 3. 包括的可視化
- **4×3レイアウト**: 元スクリプトと同一の可視化構造
- **厚み専用表示**: 厚み画像・プロファイル・ヒストグラム
- **境界検出**: 275.2999mm急激途切れの線状境界検証

## 技術仕様

### 実装方法
- **方法**: gpt5pro5.md推奨方法A
- **逆変換**: G = -ln(I_mask/I_air)
- **自己較正**: μ_eff = (p²∑G) / V_mask(3D)
- **厚み算出**: T = G / μ_eff [mm]

### 入出力
```
入力:
- CT: /workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz
- マスク: /workspace/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz

出力:
- PA_mask_thickness_mm.png: 厚み画像 (3070×2500)
- PA_mask_thickness_mm.npy: 生厚みデータ [mm]
- thickness_info.npy: 厚み推定パラメータ
- mask_deepdrr_boundary_analysis.png: 詳細解析
```

## 実行方法

### Docker実行
```bash
docker run --rm --gpus all \
    -v "/home/soya/ctrate_ws:/workspace" \
    deepdrr-japan-mirror:latest bash -c "
      pip3 install torch torchvision pydicom opencv-python --no-cache-dir > /dev/null 2>&1 &&
      cd /workspace &&
      python3 mask_projection/src/mask_deepdrr_projection_thickness.py
    "
```

### 期待される結果
```
✅ 厚み推定完了: max=X.XXX mm
✅ 体積検証: 誤差 < 1%
✅ 線状境界判定: 成功
```

## 出力ファイル詳細

### 1. PA_mask_thickness_mm.npy
- **形状**: (3070, 2500)
- **データ型**: float32
- **単位**: mm
- **内容**: ピクセル毎の物理的厚み

### 2. thickness_info.npy
```python
{
    'mu_eff': float,              # 実効線減弱係数 [1/mm]
    'volume_3d_mm3': float,       # 3Dマスク体積 [mm³]
    'volume_2d_mm3': float,       # 2D復元体積 [mm³]
    'pixel_pitch_mm': 0.14,       # ピクセルピッチ [mm]
    'max_thickness_mm': float,    # 最大厚み [mm]
    'mean_thickness_mm': float,   # 平均厚み [mm]
    'support_pixels': int         # 非ゼロピクセル数
}
```

### 3. 可視化ファイル
- **CT DRR**: 元のCT投影画像
- **マスク DRR**: マスク投影画像
- **厚み画像**: viridisカラーマップでの厚み表示
- **プロファイル**: 中央行での厚み分布
- **ヒストグラム**: 厚み値の統計分布

## 技術的改良点

### 元スクリプトからの改良
1. **強度の物理的意味化**: Energy値 → 厚み[mm]
2. **体積保存**: 3D体積と2D投影体積の厳密一致
3. **自己較正**: 外部パラメータ不要の自動較正
4. **定量評価**: 体積誤差による精度検証

### gpt5pro5.md方法Aの利点
- **スペクトル非依存**: ポリエネルギでも体積一致保証
- **幾何活用**: DeepDRRの高精度幾何をそのまま利用
- **線形関係**: 厚みと体積の線形関係による学習安定性

## 検証項目

### 1. 体積整合性
- 3D体積と2D復元体積の誤差 < 1%

### 2. 境界再現性  
- 275.2999mm急激途切れの線状境界検出

### 3. 物理的妥当性
- 厚み値の合理的範囲 (0-50mm程度)
- μ_effの医学的妥当性

## 使用場面

### 1. AI学習データ作成
- 物理的に意味のある厚み[mm]ラベル
- 体積一貫性による学習安定化

### 2. 定量解析
- 脂肪組織の厚み分布解析
- 解剖学的構造の定量評価

### 3. 検証・校正
- DeepDRR投影の物理的検証
- マスク品質の定量評価

## 関連ファイル

- `mask_deepdrr_projection.py`: 元の境界検出版
- `gpt5pro5.md`: 実装理論・推奨方法の詳細
- `CLAUDE.md`: プロジェクト全体指針