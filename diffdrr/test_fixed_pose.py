#!/usr/bin/env python3
"""
固定カメラポーズでDRR生成をテスト
"""

import torch
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffdrr.drr import DRR
from diffdrr.pose import convert
from typing import Tuple

class FixedPoseDRRTest:
    """固定カメラポーズでDRRをテスト"""
    
    def __init__(self, 
                 ct_path: str,
                 mask_path: str,
                 output_dir: str = "./test_fixed",
                 device: str = "cuda"):
        self.ct_path = ct_path
        self.mask_path = mask_path
        self.output_dir = Path(output_dir)
        
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading NIfTI data...")
        self.subject = self.load_nifti_data()
    
    def load_nifti_data(self) -> tio.Subject:
        subject = tio.Subject(
            ct=tio.ScalarImage(self.ct_path),
            mask=tio.LabelMap(self.mask_path),
        )
        return subject
    
    def setup_drr_generator(self, subject: tio.Subject) -> DRR:
        # TorchIOから必要なデータを抽出
        ct_tensor_hu = subject.ct.data.squeeze(0)  # (D, H, W)
        spacing_xyz = subject.ct.spacing  # (sx, sy, sz)
        affine_matrix = subject.ct.affine  # 4x4 numpy array
        
        # HU値を線減弱係数(LAC)に変換
        volume_clipped = torch.clamp(ct_tensor_hu, -1024, 3071)
        mu_water = 0.0192  # [mm^-1]
        volume_lac = (volume_clipped + 1024) * (mu_water / 1000.0)
        
        print(f"\nVolume Statistics:")
        print(f"  Shape: {volume_lac.shape}")
        print(f"  HU range: [{ct_tensor_hu.min().item():.1f}, {ct_tensor_hu.max().item():.1f}]")
        print(f"  LAC range: [{volume_lac.min().item():.6f}, {volume_lac.max().item():.6f}]")
        print(f"  Spacing: {spacing_xyz}")
        
        # ボリュームの物理的サイズを計算
        volume_size_mm = torch.tensor(volume_lac.shape) * torch.tensor(spacing_xyz)
        print(f"  Physical size (mm): {volume_size_mm.tolist()}")
        print(f"  Center position: {(volume_size_mm / 2).tolist()}")
        
        # DiffDRRに適した形式で4Dテンソルを準備
        volume_4d = volume_lac.unsqueeze(0).to(self.device)  # (1, D, H, W)
        
        # DiffDRRSubjectを直接作成
        diffdrr_subject = tio.Subject(
            volume=tio.ScalarImage(tensor=volume_4d, affine=affine_matrix),
            density=tio.ScalarImage(tensor=volume_4d, affine=affine_matrix),
        )
        
        # DiffDRRが期待する追加属性を設定
        diffdrr_subject.__dict__['spacing'] = spacing_xyz
        diffdrr_subject.__dict__['reorient'] = torch.eye(4, dtype=torch.float32).to(self.device)
        diffdrr_subject.__dict__['mask'] = None
        diffdrr_subject.__dict__['fiducials'] = None
        
        # DRRジェネレータを初期化
        drr_generator = DRR(
            subject=diffdrr_subject,
            sdd=1800.0,      # Source-to-detector distance
            height=512,      # 検出器の高さ（ピクセル）
            width=512,       # 検出器の幅（ピクセル）
            delx=0.7,        # ピクセルサイズX（mm）
            dely=0.7,        # ピクセルサイズY（mm）
        ).to(self.device)
        
        return drr_generator
    
    def test_poses(self):
        """複数の固定カメラポーズでテスト"""
        drr_generator = self.setup_drr_generator(self.subject)
        
        # テストする固定ポーズのリスト
        test_poses = [
            # (name, rotations, translations)
            ("frontal_center", [0, 0, 0], [0, 0, 0]),
            ("frontal_up", [0, 0, 0], [0, -100, 0]),  # カメラを上に移動
            ("frontal_down", [0, 0, 0], [0, 100, 0]),  # カメラを下に移動
            ("lateral", [0, 90, 0], [0, 0, 0]),  # 横から
            ("oblique", [15, 30, 0], [0, 0, 0]),  # 斜めから
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (name, rot, trans) in enumerate(test_poses):
            if idx >= 6:
                break
            
            print(f"\nTesting pose: {name}")
            print(f"  Rotation (deg): {rot}")
            print(f"  Translation (mm): {trans}")
            
            # 固定ポーズを設定
            rotations = torch.tensor([rot], dtype=torch.float32)
            translations = torch.tensor([trans], dtype=torch.float32)
            
            # RigidTransformを作成
            rotations_rad = torch.deg2rad(rotations).to(self.device)
            translations = translations.to(self.device)
            transform = convert(
                rotations_rad,
                translations,
                parameterization="euler_angles",
                convention="XYZ",
            ).to(self.device)
            
            # DRRを生成
            drr_image = drr_generator(transform)
            
            # 統計情報を出力
            print(f"  DRR Stats - Min: {drr_image.min().item():.4f}, "
                  f"Max: {drr_image.max().item():.4f}, "
                  f"Mean: {drr_image.mean().item():.4f}")
            
            # 可視化
            drr_np = drr_image.squeeze().cpu().numpy()
            
            # 正規化（ガンマ補正でコントラストを改善）
            drr_normalized = (drr_np - drr_np.min()) / (drr_np.max() - drr_np.min() + 1e-8)
            drr_gamma = np.power(drr_normalized, 0.5)  # ガンマ補正
            
            axes[idx].imshow(drr_gamma, cmap='gray')
            axes[idx].set_title(f'{name}\nMin:{drr_image.min():.3f} Max:{drr_image.max():.3f}')
            axes[idx].axis('off')
        
        # 最後の軸を隠す
        if idx < 5:
            axes[5].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fixed_pose_tests.png", dpi=150)
        plt.show()
        print(f"\nResults saved to {self.output_dir}/fixed_pose_tests.png")

def main():
    """メイン関数"""
    CT_NIFTI_PATH = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    MASK_NIFTI_PATH = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    tester = FixedPoseDRRTest(
        ct_path=CT_NIFTI_PATH,
        mask_path=MASK_NIFTI_PATH,
        output_dir="./test_fixed"
    )
    
    tester.test_poses()

if __name__ == "__main__":
    main()