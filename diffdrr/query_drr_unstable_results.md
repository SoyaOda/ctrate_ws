# DiffDRR実装：DRR生成が不安定で結果が一貫しない問題

## 環境
- Python 3.12  
- DiffDRR v0.5.1
- TorchIO 0.20.21
- PyTorch 2.8.0
- CUDA対応環境

## 問題の概要
DRR生成パイプラインは動作するものの、結果が非常に不安定です：
1. **同じコードで実行ごとに異なる結果**（完全に黒 vs 正常な画像）
2. **同一実行内でもサンプルによって結果が異なる**（サンプル0は成功、サンプル1は失敗）
3. **マスク投影の体積が大幅に異なる**（元の体積の約5倍になることも）

## 現在の完全な実装コード

### generate_drrs.py の主要部分

```python
#!/usr/bin/env python3
import torch
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import tempfile
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.pose import convert
from diffdrr.data import read
from typing import Tuple, Optional

class DRRPipeline:
    def __init__(self, 
                 ct_path: str,
                 mask_path: str,
                 output_dir: str = "./augmented_data",
                 device: str = "cuda"):
        self.ct_path = ct_path
        self.mask_path = mask_path
        self.output_dir = Path(output_dir)
        
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "drr").mkdir(exist_ok=True)
        (self.output_dir / "mask").mkdir(exist_ok=True)
        (self.output_dir / "visualization").mkdir(exist_ok=True)
        
        print("Loading NIfTI data...")
        self.initial_subject = self.load_nifti_data()
        self.augmentation_pipeline = self.get_augmentation_transform()
    
    def load_nifti_data(self) -> tio.Subject:
        subject = tio.Subject(
            ct=tio.ScalarImage(self.ct_path),
            mask=tio.LabelMap(self.mask_path),
        )
        return subject
    
    def get_augmentation_transform(self) -> tio.Compose:
        return tio.Compose([
            # 空間変換
            tio.RandomAffine(
                scales=(0.95, 1.05),
                degrees=10,
                translation=20,
                isotropic=False,
                center='image',
                default_pad_value='minimum',
                p=0.5
            ),
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=20,
                locked_borders=2,
                p=0.3
            ),
            # 強度変換（CTのみ）
            tio.RandomNoise(
                mean=0,
                std=(0, 25),
                include=['ct'],
                p=0.3
            ),
            tio.RandomBlur(
                std=(0, 1),
                include=['ct'],
                p=0.3
            ),
        ])
    
    def setup_drr_generator(self, subject: tio.Subject) -> DRR:
        """
        TorchIO SubjectからDRRジェネレータを正しくセットアップする。
        """
        # Step 1: TorchIOから必要なデータを抽出
        ct_tensor_hu = subject.ct.data.squeeze(0)  # (D, H, W)
        spacing_xyz = subject.ct.spacing  # (sx, sy, sz)
        affine_matrix = subject.ct.affine  # 4x4 numpy array

        # Step 2: HU値を線減弱係数(LAC)に変換
        volume_clipped = torch.clamp(ct_tensor_hu, -1024, 3071)
        mu_water = 0.0192  # [mm^-1]
        volume_lac = (volume_clipped + 1024) * (mu_water / 1000.0)
        
        print(f"\nLAC Volume Statistics (setup_drr_generator):")
        print(f"  HU range: [{ct_tensor_hu.min().item():.1f}, {ct_tensor_hu.max().item():.1f}]")
        print(f"  LAC range: [{volume_lac.min().item():.6f}, {volume_lac.max().item():.6f}]")
        print(f"  Shape: {volume_lac.shape}")
        print(f"  Spacing: {spacing_xyz}")

        # Step 3: read関数と同じ方法でTorchIO Subjectを作成
        volume_4d = volume_lac.unsqueeze(0).to(self.device)  # (1, D, H, W)
        
        diffdrr_subject = tio.Subject(
            volume=tio.ScalarImage(tensor=volume_4d, affine=affine_matrix),
            density=tio.ScalarImage(tensor=volume_4d, affine=affine_matrix),
        )
        
        # read関数が追加する特別な属性を手動で追加
        diffdrr_subject.__dict__['spacing'] = spacing_xyz
        diffdrr_subject.__dict__['reorient'] = torch.eye(4, dtype=torch.float32).to(self.device)
        diffdrr_subject.__dict__['mask'] = None
        diffdrr_subject.__dict__['fiducials'] = None

        # Step 4: DRRジェネレータを初期化
        drr_generator = DRR(
            subject=diffdrr_subject,
            sdd=1800.0,
            height=512,
            width=512,
            delx=0.7,
            dely=0.7,
        ).to(self.device)
        
        return drr_generator
    
    def get_random_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 回転: 各軸±10度の範囲
        rotations = (torch.rand(1, 3) - 0.5) * 20.0  # 度数法
        
        # 平行移動: X,Z軸は±50mm、Y軸は基準位置から±100mm
        translations = torch.zeros(1, 3)
        translations[:, 0] = (torch.rand(1) - 0.5) * 100.0  # X軸
        translations[:, 1] = -100.0 + (torch.rand(1) - 0.5) * 200.0  # Y軸
        translations[:, 2] = (torch.rand(1) - 0.5) * 100.0  # Z軸
        
        return rotations, translations
    
    def setup_mask_drr_generator(self, subject: tio.Subject) -> DRR:
        # マスクデータを抽出
        mask_tensor = subject.mask.data.squeeze(0).float() # (D, H, W)
        spacing_xyz = subject.mask.spacing
        affine_matrix = subject.mask.affine

        # マスクを4Dテンソルに変換
        mask_4d = mask_tensor.unsqueeze(0).to(self.device)
        
        diffdrr_subject = tio.Subject(
            volume=tio.ScalarImage(tensor=mask_4d, affine=affine_matrix),
            density=tio.ScalarImage(tensor=mask_4d, affine=affine_matrix),
        )
        
        diffdrr_subject.__dict__['spacing'] = spacing_xyz
        diffdrr_subject.__dict__['reorient'] = torch.eye(4, dtype=torch.float32).to(self.device)
        diffdrr_subject.__dict__['mask'] = None
        diffdrr_subject.__dict__['fiducials'] = None

        mask_drr_generator = DRR(
            subject=diffdrr_subject,
            sdd=1800.0,
            height=512,
            width=512,
            delx=0.7,
            dely=0.7,
        ).to(self.device)
        
        return mask_drr_generator
```

## 観察された結果のパターン

### パターン1: 完全に失敗（以前の実行）
```
DRR Image Statistics:
  Min: 0.0000
  Max: 0.0000
  Mean: 0.0000
  Std: 0.0000
```
- DRR画像: 完全に黒
- マスク: 完全に0

### パターン2: 部分的成功（最新の実行）
```
# サンプル0（成功）
DRR Image Statistics:
  Min: 5.1339
  Max: 8.5595
  Mean: 7.2315
  Std: 0.5229

# サンプル1（失敗？）
（統計情報なし、画像はほぼ黒）
```
- サンプル0: 胸部X線様の画像が生成される
- サンプル1: ほぼ黒（左下に微かな信号）
- マスク投影: 体積が元の5倍（139642.69 mm³ → 687646.75 mm³）

## 問題の分析

### 1. 不安定性の原因候補

#### A. ランダムな3D拡張の影響
- `RandomAffine`と`RandomElasticDeformation`によりボリュームが大きく変形
- 変形後のボリュームがカメラの視野外に出る可能性

#### B. ランダムなカメラポーズの影響  
- 平行移動範囲が大きすぎる（Y軸: -200mm～0mm）
- ボリュームとカメラの位置関係が不適切になることがある

#### C. 座標系の不整合
- TorchIOとDiffDRRで座標系の解釈が異なる
- `reorient`を単位行列にしているが、実際には変換が必要かもしれない

### 2. マスク投影の体積増大問題

元の3D体積の約5倍になっている原因：
- 投影方向による重複カウント？
- スケーリング係数の誤り？
- マスクのバイナリ値が正しく扱われていない？

## 検証が必要な項目

### 1. 最小構成でのテスト
```python
# 拡張なし、固定カメラポーズで確認
augmented_subject = self.initial_subject  # 拡張をスキップ
rotations = torch.tensor([[0.0, 0.0, 0.0]])  # 回転なし
translations = torch.tensor([[0.0, 0.0, 0.0]])  # 移動なし
```

### 2. 座標系の確認
- read関数で読み込んだ場合の`reorient`値を確認
- TorchIOの座標系とDiffDRRの座標系の関係を明確化

### 3. デバッグ出力の追加
- カメラポーズの具体的な値
- ボリュームの中心座標
- レイキャスティングの成功/失敗の判定

## 質問

1. **DiffDRRにおける座標系とカメラ配置の正確な仕様**
   - ボリュームの中心は常に原点(0,0,0)に配置されるのか？
   - カメラの初期位置と向きは？
   - `translations`の単位と方向の定義は？

2. **安定したDRR生成のベストプラクティス**
   - 推奨されるカメラポーズの範囲
   - 3D拡張後の座標系補正の必要性
   - ボリュームがカメラ視野内にあることを保証する方法

3. **マスク投影の正しい実装**
   - バイナリマスクから厚み（体積）への変換の正しい方法
   - 投影による体積保存の実現方法

4. **DiffDRR内部のデバッグ方法**
   - レイキャスティングの成功/失敗を確認する方法
   - 中間結果（レイの軌跡、交差点）の可視化

これらの問題を解決し、安定して高品質なDRR画像を生成できるようにするための具体的な解決策を教えてください。