#!/usr/bin/env python3
"""
Step 3: DiffDRRによる2D投影機能（修正版）
3DボリュームからDRR（Digital Reconstructed Radiograph）を生成する
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from diffdrr.drr import DRR
from diffdrr.pose import convert
from diffdrr.data import read, transform_hu_to_density
import tempfile
import nibabel as nib
from typing import Tuple, Optional


class Projector:
    """DiffDRRを使用した投影クラス"""
    
    def __init__(self, ct_tensor, spacing_tensor, device="cuda"):
        """
        初期化
        
        Args:
            ct_tensor: CTボリューム（HU値）
            spacing_tensor: ボクセル間隔 [x, y, z] mm
            device: 使用デバイス
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        
        self.device = device
        self.spacing = spacing_tensor
        
        # CTデータを一時的にNIfTIファイルとして保存
        # （DiffDRRが期待するSubject形式に変換するため）
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            # NIfTI形式で保存
            affine = np.eye(4)
            affine[0, 0] = spacing_tensor[0]
            affine[1, 1] = spacing_tensor[1]
            affine[2, 2] = spacing_tensor[2]
            
            ct_data = ct_tensor.cpu().numpy()
            img = nib.Nifti1Image(ct_data, affine)
            nib.save(img, tmp.name)
            tmp_path = tmp.name
        
        # DiffDRRのread関数で読み込み（reorient属性を持つSubjectが作成される）
        self.subject = read(tmp_path, bone_attenuation_multiplier=1.0)
        
        # 一時ファイルを削除
        import os
        os.remove(tmp_path)
        
        # DRRモジュールを初期化
        self.drr = DRR(
            self.subject,
            sdd=1800.0,
            height=512,
            delx=0.7,
            width=512,
            dely=0.7,
        ).to(self.device)
        
        # マスク用DRRモジュール（別インスタンス）
        self.drr_mask = None
        
        # ピクセル面積（体積計算用）
        self.pixel_area = 0.7 * 0.7

    def project(self, rotations_deg, translations_mm):
        """
        DRR画像を生成
        
        Args:
            rotations_deg: 回転角度（度）[batch, 3]
            translations_mm: 平行移動（mm）[batch, 3]
            
        Returns:
            DRR画像 [batch, 1, H, W]
        """
        # ラジアンに変換
        rotations_rad = torch.deg2rad(rotations_deg).to(self.device)
        translations = translations_mm.to(self.device)
        
        # RigidTransformを作成
        transform = convert(
            rotations_rad,
            translations,
            parameterization="euler_angles",
            convention="XYZ",
        ).to(self.device)
        
        # DRRを生成
        drr_image = self.drr(transform)
        
        return drr_image
    
    def setup_mask_projector(self, mask_tensor):
        """
        マスク投影用のDRRモジュールをセットアップ
        
        Args:
            mask_tensor: マスクボリューム
        """
        # マスクデータを一時的にNIfTIファイルとして保存
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            # NIfTI形式で保存
            affine = np.eye(4)
            affine[0, 0] = self.spacing[0]
            affine[1, 1] = self.spacing[1]
            affine[2, 2] = self.spacing[2]
            
            # マスクをHU値に変換（脂肪のHU値として-100を使用）
            mask_data = mask_tensor.float().cpu().numpy()
            mask_as_ct = mask_data * -100.0  # 脂肪のHU値（約-100）
            
            img = nib.Nifti1Image(mask_as_ct, affine)
            nib.save(img, tmp.name)
            tmp_path = tmp.name
        
        # DiffDRRのread関数で読み込み
        mask_subject = read(tmp_path, bone_attenuation_multiplier=1.0)
        
        # 一時ファイルを削除
        import os
        os.remove(tmp_path)
        
        # マスク用DRRモジュールを作成
        self.drr_mask = DRR(
            mask_subject,
            sdd=1800.0,
            height=512,
            delx=0.7,
            width=512,
            dely=0.7,
        ).to(self.device)
    
    def project_mask(self, mask_tensor, rotations_deg, translations_mm):
        """
        マスクの投影を生成
        
        Args:
            mask_tensor: マスクボリューム
            rotations_deg: 回転角度（度）[batch, 3]
            translations_mm: 平行移動（mm）[batch, 3]
            
        Returns:
            thickness_map, volume_map
        """
        # マスクプロジェクターをセットアップ（未設定の場合）
        if self.drr_mask is None:
            self.setup_mask_projector(mask_tensor)
        
        # ラジアンに変換
        rotations_rad = torch.deg2rad(rotations_deg).to(self.device)
        translations = translations_mm.to(self.device)
        
        # RigidTransformを作成
        transform = convert(
            rotations_rad,
            translations,
            parameterization="euler_angles",
            convention="XYZ",
        ).to(self.device)
        
        # マスクの投影を生成（厚みマップ）
        thickness_map = self.drr_mask(transform)
        
        # 体積マップに変換
        volume_map = thickness_map * self.pixel_area
        
        return thickness_map, volume_map


def test_projection_v2():
    """投影機能のテスト"""
    from load_data_sitk import load_nifti_data_sitk
    
    # データパス
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    print("Loading data with SimpleITK...")
    ct_tensor, mask_tensor, spacing_tensor = load_nifti_data_sitk(ct_path, mask_path)
    
    print(f"CT shape: {ct_tensor.shape}")
    print(f"Mask shape: {mask_tensor.shape}")
    print(f"Spacing: {spacing_tensor.tolist()}")
    
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Projectorを初期化
    print("\nInitializing Projector...")
    projector = Projector(ct_tensor, spacing_tensor, device=device)
    
    # カメラポーズを設定
    rotations = torch.tensor([[0.0, 5.0, 10.0]], dtype=torch.float32)  # 度
    translations = torch.tensor([[0.0, 0.0, 50.0]], dtype=torch.float32)  # mm
    
    print(f"Rotations (deg): {rotations.squeeze().tolist()}")
    print(f"Translations (mm): {translations.squeeze().tolist()}")
    
    # DRRを生成
    print("\nGenerating DRR...")
    drr_image = projector.project(rotations, translations)
    print(f"DRR shape: {drr_image.shape}")
    print(f"DRR range: [{drr_image.min():.2f}, {drr_image.max():.2f}]")
    
    # マスクの投影を生成
    print("\nProjecting mask...")
    thickness_map, volume_map = projector.project_mask(mask_tensor, rotations, translations)
    print(f"Thickness map shape: {thickness_map.shape}")
    print(f"Thickness range: [{thickness_map.min():.2f}, {thickness_map.max():.2f}] mm")
    
    # 体積の確認
    total_volume = volume_map.sum()
    original_volume = mask_tensor.sum() * torch.prod(spacing_tensor)
    print(f"\nOriginal 3D volume: {original_volume:.2f} mm³")
    print(f"Projected 2D volume: {total_volume:.2f} mm³")
    print(f"Ratio: {(total_volume / original_volume * 100):.2f}%")
    
    # 結果を保存
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    drr_np = drr_image.squeeze().cpu().numpy()
    thickness_np = thickness_map.squeeze().cpu().numpy()
    
    # 画像として保存
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(drr_np, cmap='gray')
    plt.title('DRR Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(thickness_np, cmap='hot')
    plt.title('Mask Thickness Map')
    plt.colorbar(label='Thickness (mm)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_projection_v2.png')
    print(f"\nResults saved to {output_dir}")
    
    print("✓ Test completed successfully!")


if __name__ == "__main__":
    test_projection_v2()