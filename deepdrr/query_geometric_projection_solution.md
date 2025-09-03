# DiffDRRでバイナリマスクの幾何学的投影を正確に行う方法

## 背景と要求仕様

3D CTデータとそのセグメンテーションマスク（心臓周囲脂肪）から、以下の3条件を満たす2D投影を生成したい：

1. **空間的整合性**: 2DマスクがCT DRRと位置的に完全に一致
2. **厚み比例性**: 各ピクセル値が対応する3Dマスクの実際の厚みと比例
3. **体積保存**: 2Dマスクの総和が元の3D体積と一致

## 環境
- DiffDRR v0.5.1
- PyTorch 2.8.0
- Python 3.12
- CUDA 12.0

## 現在の実装と問題点

### 実装1: 直接投影方式 (test_mask_projection.py)

```python
#!/usr/bin/env python3
"""
CTとマスクの2D投影テスト
問題：DiffDRRの物理シミュレーションにより体積が保存されない
"""

import torch
from diffdrr.drr import DRR
from diffdrr.data import read
import numpy as np
import nibabel as nib

def generate_ct_and_mask_drrs():
    # パス設定
    ct_path = "/path/to/ct.nii.gz"
    mask_path = "/path/to/mask.nii.gz"
    
    # マスクデータを読み込み（バイナリ: 0 or 1）
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    mask_binary = (mask_data > 0).astype(np.float32)
    
    # 3D体積を計算
    voxel_dims = mask_nifti.header.get_zooms()[:3]
    voxel_volume = np.prod(voxel_dims)  # 0.175 mm³
    total_volume_3d = np.sum(mask_binary) * voxel_volume  # 139,642.69 mm³
    
    # DiffDRRで投影
    mask_subject = read(mask_nifti_path)
    mask_drr = DRR(
        mask_subject,
        sdd=1800.0,
        height=512,
        width=512,
        delx=0.7,
        dely=0.7,
    ).to(device)
    
    # PA viewで投影
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    mask_projection = mask_drr(rotations, translations,
                               parameterization="euler_angles",
                               convention="ZXY")
    
    # 体積計算（問題：物理的減衰が適用される）
    mask_proj_np = mask_projection.squeeze().cpu().numpy()
    pixel_area = 0.7 * 0.7  # 0.49 mm²
    volume_2d = np.sum(mask_proj_np) * pixel_area  # 366,519.59 mm³
    
    print(f"3D Volume: {total_volume_3d:.2f} mm³")
    print(f"2D Volume: {volume_2d:.2f} mm³")
    print(f"Error: {abs(total_volume_3d - volume_2d) / total_volume_3d * 100:.1f}%")
    # Error: 162.5%
```

**問題点：**
- DiffDRRの出力は透過率（I = exp(-∫μ dx)）として計算される
- バイナリマスクが物理的減衰として扱われる
- 結果：体積が2.6倍に増大（誤差162.5%）

### 実装2: キャリブレーション方式 (test_mask_calibration.py)

```python
#!/usr/bin/env python3
"""
体積保存キャリブレーション実装
問題：体積は保存されるが、厚みの変化が失われる
"""

def project_mask_with_volume_calibration(mask_path, ...):
    # マスクを読み込み
    mask_binary = (mask_data > 0).astype(np.float32)
    true_3d_volume = np.sum(mask_binary) * voxel_volume  # 139,642.69 mm³
    
    # DiffDRRで投影
    transmittance = drr(rotations, translations, ...)
    
    # 透過率から相対的な厚みを計算
    # I = exp(-thickness) → thickness = -log(I)
    relative_thickness = -torch.log(torch.clamp(transmittance, min=1e-9))
    relative_thickness_np = relative_thickness.squeeze().cpu().numpy()
    
    # 投影された体積（未調整）
    uncalibrated_volume = np.sum(relative_thickness_np) * pixel_area
    # 1,665,654.62 mm³（約12倍！）
    
    # キャリブレーション係数を計算
    calibration_factor = true_3d_volume / uncalibrated_volume
    # 0.083837
    
    # キャリブレーションを適用
    calibrated_thickness = relative_thickness_np * calibration_factor
    volume_map = calibrated_thickness * pixel_area
    
    calibrated_volume = np.sum(volume_map)  # 139,642.70 mm³
    print(f"Error: {abs(true_3d_volume - calibrated_volume) / true_3d_volume * 100:.4f}%")
    # Error: 0.0000%（完璧な体積保存）
```

**問題点：**
- 体積は完璧に保存される（誤差0.0000%）
- しかし、厚みの変化が大幅に圧縮される

## 検証結果の詳細

### 厚み分布の圧縮問題

```python
# 実際の3D厚み分布（Z軸方向）
true_thickness_z = np.sum(mask_binary, axis=2) * voxel_dims[2]
# 統計：
#   - 範囲: 1.5 - 69.0 mm（67.5 mmの幅）
#   - 46種類の異なる値
#   - 標準偏差: 13.8 mm

# キャリブレーション後の2D厚み
calibrated_thickness = relative_thickness * 0.083837
# 統計：
#   - 範囲: 0.010 - 1.737 mm（1.73 mmの幅）
#   - わずか7種類の値
#   - 標準偏差: 0.137 mm
#   - 圧縮率: 39.1倍
```

### Beer-Lambert則による非線形圧縮

```
厚み → DiffDRR出力の変換：
1.5 mm → exp(-1.5) = 0.223 → キャリブレート後 0.019 mm
10 mm  → exp(-10)  = 0.0000454 → キャリブレート後 0.0038 mm
30 mm  → exp(-30)  = 9.36×10⁻¹⁴ → ほぼゼロ
69 mm  → exp(-69)  = 0 → 完全にゼロ
```

## 根本的な問題

DiffDRRは医療画像用のX線減衰シミュレータとして設計されており、内部で以下の処理を行う：

1. 入力を密度/減衰係数として扱う
2. Siddon's algorithmでレイキャスティング
3. Beer-Lambert則を適用：I = I₀ × exp(-∫μ(s) ds)
4. 出力は透過率（0-1の範囲）

バイナリマスク（0 or 1）に対してこの物理シミュレーションを適用すると：
- 厚い領域（>30mm）はすべて「ほぼゼロ」になる
- 薄い領域の差も指数関数的に圧縮される
- キャリブレーションで体積は合わせられるが、局所的な厚みの精度は回復不可能

## 質問

### 1. DiffDRR内でBeer-Lambert則を回避する方法はあるか？

- カスタムレンダラーの実装？
- 内部パラメータの調整？
- 未文書化のAPIやオプション？

### 2. 純粋な幾何学的投影の実装方法

必要な機能：
```python
def geometric_ray_casting(
    mask_volume: np.ndarray,      # 3Dバイナリマスク
    camera_params: dict,          # DiffDRRと同じパラメータ
    voxel_spacing: tuple,         # ボクセルサイズ
) -> np.ndarray:
    """
    各レイが通過するマスクボクセルの物理的な長さ（mm）を計算
    Beer-Lambert則を使わず、単純な線積分を実行
    """
    # 実装が必要
    pass
```

### 3. 代替ライブラリや手法

以下のアプローチのどれが最適か：

a) **PyTorch3Dを使用**
```python
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points
# カスタムレイキャスティングの実装
```

b) **SimpleITKのRayCastInterpolator**
```python
import SimpleITK as sitk
interpolator = sitk.RayCastInterpolator()
# 幾何学的投影の実装
```

c) **カスタムCUDAカーネル**
```cuda
__global__ void geometric_ray_cast(
    float* volume, float* projection,
    // Siddon's algorithm without Beer-Lambert
)
```

d) **DRRプロジェクト（別ライブラリ）**
- plastimatch
- RTK (Reconstruction Toolkit)
- これらは幾何学的投影をサポートするか？

### 4. ハイブリッドアプローチ

DiffDRRの利点（GPU最適化、微分可能）を活かしつつ、正確な厚み投影を実現する方法：

```python
# 案：マスクを密度エンコードして投影
def encode_thickness_as_density(mask_3d):
    """
    各ボクセルに、その位置のZ方向の厚みを密度として割り当て
    これによりDiffDRRの出力が厚みと相関するようにする
    """
    # 実装方法は？
```

## 要求される解決策

1. **3つの条件をすべて満たす**実装
2. **DiffDRRと同じカメラパラメータ**で動作
3. **GPU対応**で高速処理
4. **検証可能**な精度（厚み誤差<5%）

これらについて、公式実装を参考にした解決策を教えてください。