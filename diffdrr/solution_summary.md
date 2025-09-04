# DiffDRRを使用した幾何学的投影の実装まとめ

## 問題の本質
DiffDRRは医療画像のX線シミュレーションのために設計されており、Beer-Lambert則（I = I₀ × exp(-μt)）が内部的に強制適用される。これにより：
- 厚い領域の値が指数関数的に圧縮される
- 動的範囲が大幅に失われる（67.5mm → 1.6mm）

## 試みたアプローチと結果

### 1. 減衰係数の調整
- **μ = 0.001**: 動的範囲1.6mm（圧縮率41倍）
- **μ = 0.0**: 情報が完全に失われる（すべて1になる）
- **μ = 1e-10**: 数値的アンダーフローで非現実的な値

### 2. 線形近似
- 小さいμで`exp(-μt) ≈ 1 - μt`として近似
- 体積保存は達成（誤差0.00%）
- しかし動的範囲の圧縮は解決せず

### 3. 対数線形化
- 変動係数を5.2% → 63.7%に改善
- しかし動的範囲は依然1.6mmに圧縮

## 結論と推奨事項

### 現在の成果
✅ **体積保存**: 完璧（誤差0.00%）
✅ **空間的整合性**: CT DRRと完全一致
⚠️ **厚み比例性**: 相対的な違いは保持されるが、絶対値は圧縮

### 今後の選択肢

#### オプション1: 現在の実装を使用
- `final_geometric_projection.py`が最も安定
- 体積情報が重要な用途には十分
- 実装済みですぐ使用可能

#### オプション2: DiffDRRのソースコード修正
```bash
# DiffDRRをフォーク
git clone https://github.com/eigenvivek/DiffDRR.git
cd DiffDRR

# コア部分を修正（diffdrr/rendering.py など）
# exp(-integral) を単純な integral に置き換え
```

#### オプション3: 代替実装
1. **PyTorch3D**での完全実装（座標変換の課題あり）
2. **カスタムCUDAカーネル**（最も柔軟だが開発コスト高）
3. **SimpleITK/ITK**のレイキャスティング（CPUベース、遅い）

## 実装ファイル一覧

### 主要な実装
- `final_geometric_projection.py` - 最終的な実用版（体積保存重視）
- `improved_geometric_projection.py` - 動的範囲改善版（CV: 63.7%）
- `test_zero_attenuation.py` - 減衰係数実験

### 出力ディレクトリ
- `/home/soya/ctrate_ws/deepdrr/final_output/` - 最終結果
- `/home/soya/ctrate_ws/deepdrr/improved_output/` - 改善版結果
- `/home/soya/ctrate_ws/deepdrr/zero_attenuation_output/` - 実験結果

## 使用方法

```python
from final_geometric_projection import GeometricProjector

# プロジェクタを初期化
projector = GeometricProjector()

# マスクを投影
result = projector.project_mask(
    mask_path="path/to/mask.nii.gz",
    use_calibration=True,
    attenuation_scale=0.001
)

# 結果を取得
thickness_map = result['thickness']  # 2D厚みマップ
volume_3d = result['volume_3d']      # 元の3D体積
```