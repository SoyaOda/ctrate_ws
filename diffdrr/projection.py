#!/usr/bin/env python3
"""
Step 3: DiffDRRによる2D投影機能
3DボリュームからDRR（Digital Reconstructed Radiograph）を生成する
DiffDRR v0.5.1対応版
"""

import torch
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from typing import Tuple, Optional
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def preprocess_ct_for_diffdrr(ct_tensor_hu):
    """
    CTテンソル(HU)をDiffDRR用の線減弱係数(LAC)に変換する
    
    Args:
        ct_tensor_hu: CTボリューム (HU単位)
        
    Returns:
        torch.Tensor: 線減弱係数に変換されたテンソル
    """
    # 一般的なCT値の範囲にクリッピング
    ct_clipped = torch.clamp(ct_tensor_hu, -1024, 3071)
    
    # HUを線減弱係数(LAC)に変換 (mu = (HU + 1024) / 1000 * 0.02)
    # DiffDRRは内部で水の減弱係数を基準にしているため、HU+1024が一般的
    lac_tensor = (ct_clipped + 1024) * (0.02 / 1000)
    return lac_tensor


class Projector:
    def __init__(self, volume_tensor_hu, spacing_tensor, device="cuda"):
        """
        DiffDRRプロジェクターを初期化する

        Args:
            volume_tensor_hu (torch.Tensor): CTボリューム (HU単位)
            spacing_tensor (torch.Tensor): ボクセル間隔 [x, y, z]
            device (str): "cuda" or "cpu"
        """
        # デバイス設定
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
            
        self.device = torch.device(device)
        self.volume_hu = volume_tensor_hu.to(self.device)
        self.spacing = spacing_tensor.to(self.device)

        # CT値を前処理して線減弱係数(LAC)に変換
        self.volume_lac = preprocess_ct_for_diffdrr(self.volume_hu)
        
        # DiffDRR用のSubjectオブジェクトを作成
        import torchio as tio
        from diffdrr.data import transform_hu_to_density
        
        # HU値を密度に変換
        density = transform_hu_to_density(self.volume_hu.unsqueeze(0), bone_attenuation_multiplier=1.0)
        
        # TorchIOのSubjectオブジェクトを作成
        # affineは単位行列を使用（簡単のため）
        affine = torch.eye(4, dtype=torch.float32)
        
        subject = tio.Subject(
            volume=tio.ScalarImage(tensor=self.volume_hu.unsqueeze(0).cpu(), affine=affine),
            density=tio.ScalarImage(tensor=density.cpu(), affine=affine),
        )

        # DiffDRRのDRRモジュールを初期化
        # 臨床的な胸部レントゲンの設定を反映
        self.drr_module = DRR(
            subject,
            sdd=1800.0,              # Source-to-Detector Distance (線源-検出器間距離)
            height=512,              # 検出器の高さ (ピクセル)
            delx=0.7,                # ピクセル間隔 (mm)
            width=512,               # 検出器の幅 (ピクセル)
            dely=0.7,                # ピクセル間隔 (mm)
            x0=0.0,
            y0=0.0,
        ).to(self.device)
        
        # ピクセル面積を保存（体積計算用）
        self.pixel_area = 0.7 * 0.7  # delx * dely

    def project(self, rotations_xyz, translations_xyz):
        """
        指定されたカメラポーズでDRR画像を生成する

        Args:
            rotations_xyz (torch.Tensor): 回転 (度数法) [batch, 3] (rx, ry, rz)
            translations_xyz (torch.Tensor): 平行移動 (mm) [batch, 3] (tx, ty, tz)

        Returns:
            torch.Tensor: 生成されたDRR画像 [batch, height, width]
        """
        # 度数法からラジアンに変換
        rotations_rad = torch.deg2rad(rotations_xyz).to(self.device)
        translations = translations_xyz.to(self.device)

        # カメラポーズを設定
        transform = RigidTransform(
            rotations_rad,
            translations,
            parameterization="euler_angles",
            convention="XYZ",
        ).to(self.device)

        # DRR画像を生成
        drr_image = self.drr_module(transform)
        
        return drr_image

    def project_mask(self, mask_tensor, rotations_xyz, translations_xyz):
        """
        マスクから体積を反映した2D投影画像を生成する

        Args:
            mask_tensor (torch.Tensor): 3Dマスクボリューム (0 or 1)
            rotations_xyz (torch.Tensor): 回転 (度数法) [batch, 3]
            translations_xyz (torch.Tensor): 平行移動 (mm) [batch, 3]

        Returns:
            tuple: (thickness_map, volume_map)
                - thickness_map: 投影されたマスクの厚みマップ [batch, height, width]
                - volume_map: 体積マップ（各ピクセルの体積）[batch, height, width]
        """
        # マスクをDRRモジュールと同じデバイスに転送
        mask = mask_tensor.to(self.device).float()

        # カメラポーズを設定 (DRR生成時と全く同じ)
        rotations_rad = torch.deg2rad(rotations_xyz).to(self.device)
        translations = translations_xyz.to(self.device)
        transform = RigidTransform(
            rotations_rad,
            translations,
            parameterization="euler_angles",
            convention="XYZ",
        ).to(self.device)

        # マスクを一時的に線減弱係数テンソルとして扱って投影
        # DRRモジュールの内部で線積分が行われ、厚みが計算される
        original_volume = self.drr_module.volume.clone()
        self.drr_module.volume = mask
        thickness_map = self.drr_module(transform)
        self.drr_module.volume = original_volume
        
        # 厚みマップから体積マップに変換
        volume_map = thickness_map * self.pixel_area
        
        return thickness_map, volume_map


def test_projection():
    """
    投影機能をテストする
    """
    from load_data import load_nifti_data
    from augmentation import get_minimal_augmentation
    import torchio as tio
    
    # データパス
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    print("Loading data...")
    subject = load_nifti_data(ct_path, mask_path)
    
    # 軽い拡張を適用
    print("Applying augmentation...")
    augmentation = get_minimal_augmentation()
    augmented_subject = augmentation(subject)
    
    # CTとマスクのテンソルを取得
    ct_tensor = augmented_subject.ct.data.squeeze(0)  # (H, W, D)
    mask_tensor = augmented_subject.mask.data.squeeze(0)  # (H, W, D)
    spacing_tensor = torch.tensor(augmented_subject.ct.spacing, dtype=torch.float32)
    
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Projectorを初期化
    print("Initializing Projector...")
    projector = Projector(
        volume_tensor_hu=ct_tensor,
        spacing_tensor=spacing_tensor,
        device=device
    )
    
    # カメラポーズを生成（わずかな回転と移動）
    print("Setting camera pose...")
    batch_size = 1
    rotations = torch.tensor([[0.0, 3.0, 5.0]])  # 度数法で指定
    translations = torch.tensor([[0.0, 0.0, 20.0]])  # mm単位
    
    print(f"Rotations (deg): {rotations.squeeze().numpy()}")
    print(f"Translations (mm): {translations.squeeze().numpy()}")
    
    # DRRを生成
    print("Generating DRR...")
    drr_image = projector.project(rotations, translations)
    
    # マスクの投影を生成
    print("Projecting mask...")
    thickness_map, volume_map = projector.project_mask(mask_tensor, rotations, translations)
    
    # 統計情報を表示
    print("\n=== DRR Statistics ===")
    print(f"DRR shape: {drr_image.shape}")
    print(f"DRR range: [{drr_image.min():.2f}, {drr_image.max():.2f}]")
    
    print("\n=== Mask Projection Statistics ===")
    print(f"Thickness map shape: {thickness_map.shape}")
    print(f"Thickness range: [{thickness_map.min():.2f}, {thickness_map.max():.2f}] mm")
    print(f"Volume map shape: {volume_map.shape}")
    print(f"Total projected volume: {volume_map.sum():.2f} mm³")
    
    # 元の3Dマスクの体積と比較
    voxel_volume = torch.prod(spacing_tensor)
    original_volume = mask_tensor.sum() * voxel_volume
    print(f"\nOriginal 3D mask volume: {original_volume:.2f} mm³")
    print(f"Volume preservation ratio: {(volume_map.sum() / original_volume * 100):.2f}%")
    
    # 結果を保存
    print("\nSaving results...")
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    drr_dir = output_dir / "drr"
    mask_dir = output_dir / "mask"
    drr_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    
    # 画像として保存
    drr_np = drr_image.squeeze().cpu().detach().numpy()
    thickness_np = thickness_map.squeeze().cpu().detach().numpy()
    
    # 正規化して保存
    drr_normalized = (drr_np - drr_np.min()) / (drr_np.max() - drr_np.min() + 1e-8)
    thickness_normalized = (thickness_np - thickness_np.min()) / (thickness_np.max() - thickness_np.min() + 1e-8)
    
    plt.imsave(drr_dir / "test_drr.png", drr_normalized, cmap='gray')
    plt.imsave(mask_dir / "test_thickness.png", thickness_normalized, cmap='hot')
    
    # NumPy配列として保存
    np.save(drr_dir / "test_drr.npy", drr_np)
    np.save(mask_dir / "test_thickness.npy", thickness_np)
    np.save(mask_dir / "test_volume.npy", volume_map.cpu().numpy())
    
    print("✓ Projection test completed successfully!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # テスト実行
    test_projection()