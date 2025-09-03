# DiffDRRでバイナリマスクの幾何学的投影を行う際の体積保存問題

## 環境
- DiffDRR v0.5.1
- PyTorch 2.8.0
- CUDA 12.0
- Python 3.12

## 実装目的
3D CTデータとそのセグメンテーションマスク（EAT+PAT領域）から、以下を生成したい：
1. **擬似胸部X線画像（DRR）**: 臨床的にリアルな2D投影画像
2. **定量的2Dマスク**: 各ピクセルが3D体積（mm³）を表し、総和が元の3D体積と一致する

## 現在の実装

### 成功している部分：CT DRRの生成
```python
from diffdrr.drr import DRR
from diffdrr.data import read

# CTボリュームを読み込み（ToCanonicalなしで直接）
ct_subject = read(ct_path)

# DRRモジュール初期化
drr = DRR(
    ct_subject,
    sdd=1800.0,    # Source-to-Detector Distance
    height=512,    # 画像サイズ
    width=512,
    delx=0.7,      # ピクセル間隔 (mm)
    dely=0.7,
).to(device)

# PA view（正面）でDRR生成
rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)

ct_drr = drr(rotations, translations,
             parameterization="euler_angles",
             convention="ZXY")

# 結果：高品質な胸部X線様画像が生成される
```

### 問題：マスクの投影で体積が保存されない

#### 試行1：バイナリマスクをそのまま投影
```python
# マスクデータ（0 or 1のバイナリ）を読み込み
mask_data = nib.load(mask_path).get_fdata()
voxel_volume = 0.342 * 0.342 * 1.5  # mm³

# 3D体積：139,642.68 mm³
original_volume_3d = np.sum(mask_data) * voxel_volume

# DiffDRRで投影
mask_subject = read(mask_nifti_path)
mask_drr = DRR(mask_subject, sdd=1800.0, height=512, width=512, 
               delx=0.7, dely=0.7).to(device)

mask_thickness = mask_drr(rotations, translations,
                          parameterization="euler_angles",
                          convention="ZXY")

# 体積に変換
pixel_area = 0.7 * 0.7  # mm²
volume_map = mask_thickness * pixel_area

# 問題：投影後の体積が366,519.59 mm³（誤差162.47%）
projected_volume_2d = np.sum(volume_map)
```

#### 試行2：マスクをスケーリング
```python
# マスクを1000倍にスケール（物理的減衰を回避）
mask_scaled = mask_data.astype(np.float32) * 1000.0

# DiffDRRで投影後、逆変換
mask_projection = mask_drr(rotations, translations, ...)
mask_thickness_raw = -np.log(np.maximum(mask_projection, 1e-6))
mask_thickness = mask_thickness_raw / 1000.0 * np.mean(voxel_dims[:2])

# 問題：投影後の体積が370.48 mm³（誤差99.73%）
# 厚みマップの範囲が異常に小さい：[0.000, 0.005] mm
```

## 問題の原因分析

1. **DiffDRRの内部処理**
   - DiffDRRは医療画像用のX線減衰シミュレーション
   - 入力を密度/減衰係数として扱い、Beer-Lambert則を適用
   - 出力は減衰後の強度（I = I₀ * exp(-∫μ dx)）
   - バイナリマスクの純粋な幾何学的投影には不適切

2. **期待する動作**
   - マスク領域（値=1）を通過するレイの長さ（厚み）を計算
   - 厚み × ピクセル面積 = 体積
   - 全ピクセルの体積の総和 = 元の3D体積

3. **現在の結果**
   - マスクの投影結果が物理的減衰として計算される
   - 単純な積分（Σ mask * dx）ではなく、exp(-Σ mask * dx)が計算される

## 質問

1. **DiffDRRでバイナリマスクの純粋な幾何学的投影（レイに沿った積分）を実現する方法は？**
   - カスタムレンダラーの実装が必要？
   - 既存のパラメータで対応可能？

2. **代替アプローチの提案**
   - DRRProj（別のDRRライブラリ）の使用？
   - PyTorch3Dでのカスタムレイキャスティング？
   - SimpleITKのRayCastInterpolator？

3. **実装例**
   - 3Dマスクから2D厚みマップを生成する最も効率的な方法
   - 体積保存が保証される投影アルゴリズム
   - DiffDRRと同じカメラパラメータで使用可能な実装

## 要求仕様

```python
def project_mask_to_thickness_map(
    mask_volume: np.ndarray,  # 3Dバイナリマスク
    voxel_spacing: tuple,      # ボクセルサイズ (x, y, z) mm
    camera_params: dict,       # DiffDRRと同じパラメータ
) -> np.ndarray:
    """
    返り値：2D厚みマップ（各ピクセルはレイが通過するマスクの厚み[mm]）
    条件：np.sum(thickness_map * pixel_area) ≈ np.sum(mask_volume) * voxel_volume
    """
    pass
```

これらについて、公式実装を参考にした解決策を教えてください。