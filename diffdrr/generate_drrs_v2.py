#!/usr/bin/env python3
"""
改良版DRR生成パイプライン
よりリアルな胸部X線画像を生成
"""

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
import os


class ImprovedDRRPipeline:
    """改良版DRR生成パイプライン"""
    
    def __init__(self, 
                 ct_path: str,
                 mask_path: str,
                 output_dir: str = "./drr_output_v2",
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
    
    def load_nifti_data(self) -> tio.Subject:
        subject = tio.Subject(
            ct=tio.ScalarImage(self.ct_path),
            mask=tio.LabelMap(self.mask_path),
        )
        return subject
    
    def setup_drr_generator_with_read(self, ct_path: str) -> DRR:
        """
        readを使って正しくDRRジェネレータをセットアップ
        """
        # DiffDRRのread関数で読み込み（正しい前処理が適用される）
        diffdrr_subject = read(
            ct_path, 
            bone_attenuation_multiplier=2.5  # 骨を強調
        )
        
        # DRRジェネレータを初期化
        drr_generator = DRR(
            subject=diffdrr_subject,
            sdd=1800.0,      # 標準的な胸部X線の距離
            height=512,
            width=512,
            delx=0.7,
            dely=0.7,
        ).to(self.device)
        
        return drr_generator
    
    def generate_sample_simple(self, sample_id: int):
        """
        シンプルなDRR生成（拡張なし）
        """
        print(f"\nGenerating sample {sample_id}...")
        
        # 元のCTファイルを直接使用
        drr_generator = self.setup_drr_generator_with_read(self.ct_path)
        
        # 固定カメラポーズ（正面から、適切な位置）
        rotations = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        translations = torch.tensor([[0.0, -100.0, 0.0]], dtype=torch.float32)  # Y軸を調整
        
        # 小さなランダム変動を追加
        if sample_id > 0:
            rotations += (torch.rand(1, 3) - 0.5) * 5.0  # ±2.5度
            translations[:, 0] += (torch.rand(1) - 0.5) * 20.0  # X: ±10mm
            translations[:, 1] += (torch.rand(1) - 0.5) * 20.0  # Y: ±10mm
            translations[:, 2] += (torch.rand(1) - 0.5) * 20.0  # Z: ±10mm
        
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
        
        print(f"  DRR Stats - Min: {drr_image.min().item():.4f}, "
              f"Max: {drr_image.max().item():.4f}, "
              f"Mean: {drr_image.mean().item():.4f}")
        
        # 結果を保存
        self.save_results_xray_style(drr_image, sample_id)
    
    def save_results_xray_style(self, drr_image: torch.Tensor, sample_id: int):
        """
        胸部X線スタイルで結果を保存
        """
        # TensorをNumpy配列に変換
        drr_np = drr_image.squeeze().cpu().numpy()
        
        # X線物理に基づく処理
        # 1. 対数変換（Lambert-Beer則）
        drr_np_safe = np.maximum(drr_np, 1e-6)  # 0を避ける
        drr_log = -np.log(drr_np_safe / np.max(drr_np_safe))
        
        # 2. ヒストグラム平坦化
        from skimage import exposure
        drr_equalized = exposure.equalize_adapthist(
            (drr_log - drr_log.min()) / (drr_log.max() - drr_log.min()),
            clip_limit=0.03  # コントラスト制限
        )
        
        # 3. ガンマ補正
        gamma = 1.2
        drr_gamma = np.power(drr_equalized, gamma)
        
        # 保存
        plt.imsave(
            self.output_dir / "drr" / f"sample_{sample_id:04d}.png",
            drr_gamma,
            cmap='gray'
        )
        
        # 可視化
        if sample_id < 5:
            self.create_visualization_simple(drr_gamma, sample_id)
    
    def create_visualization_simple(self, drr: np.ndarray, sample_id: int):
        """
        シンプルな可視化
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.imshow(drr, cmap='gray')
        ax.set_title(f'Sample {sample_id:04d} - Chest X-ray Style DRR')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualization" / f"sample_{sample_id:04d}.png", dpi=150)
        plt.close()
    
    def generate_dataset(self, num_samples: int = 5):
        """
        データセットを生成
        """
        print(f"\n=== Generating {num_samples} chest X-ray style DRR samples ===")
        print(f"Output directory: {self.output_dir}")
        
        for i in range(num_samples):
            self.generate_sample_simple(i)
        
        print(f"\n✓ Successfully generated {num_samples} samples")
        print(f"  - DRR images: {self.output_dir}/drr/")
        print(f"  - Visualizations: {self.output_dir}/visualization/")


def main():
    """メイン関数"""
    CT_NIFTI_PATH = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    MASK_NIFTI_PATH = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    # scikit-imageをインストール
    import subprocess
    try:
        from skimage import exposure
    except ImportError:
        print("Installing scikit-image...")
        subprocess.check_call(["pip", "install", "scikit-image"])
    
    pipeline = ImprovedDRRPipeline(
        ct_path=CT_NIFTI_PATH,
        mask_path=MASK_NIFTI_PATH,
        output_dir="./drr_output_v2"
    )
    
    pipeline.generate_dataset(num_samples=3)


if __name__ == "__main__":
    main()